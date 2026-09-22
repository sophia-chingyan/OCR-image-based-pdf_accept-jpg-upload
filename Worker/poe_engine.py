"""
Poe OCR Engine Implementation
==============================
Uses Poe's OpenAI-compatible external API (https://api.poe.com) to perform
OCR + layout analysis through a vision-capable bot hosted on Poe (Claude,
GPT, Gemini, etc.) in a single request per page — an alternative to
gemini_engine.py's direct-to-Gemini approach for anyone who already has a
Poe subscription / API key.

Why you might pick this over the default "gemini" engine:
- Reuse an existing Poe subscription / points balance instead of a
  separate Google AI Studio key
- Swap the underlying model by bot name (POE_MODEL) without touching code
  — any vision-capable bot on Poe works, not just Gemini

Model selection:
Poe has no platform-wide default model — every request must name a bot,
and which bots you can call depends on your account. Set the POE_MODEL
environment variable (or ocr.poe_model_name in config.yaml) to a
vision-capable bot name from your Poe account, e.g. "Claude-Sonnet-4.5" or
"GPT-5" — confirm the exact name at https://poe.com first. load() fails
fast with an actionable error if neither is set, rather than guessing.

Rate-limiting:
Poe's external API enforces a flat 500 requests/minute ceiling per account
(no daily-quota concept the way Gemini's free tier has — usage instead
draws down your Poe points balance). The engine self-throttles to stay
within the configured RPM, clamped to that 500 ceiling.

Persistent per-page caching:
Same contract as GeminiOCREngine — the worker can pre-populate this
engine's in-memory cache with a previously saved page result via
`prime_page_cache_from_dict`, and read back the just-computed result via
`export_last_page_result`, so a paused/resumed job doesn't re-spend Poe
points on pages already OCR'd.

Known limitation vs. gemini_engine.py:
This engine does not implement Gemini's diagram-tiling fallback. A page
that comes back with zero text blocks is treated as image-only (genuinely
blank or decorative) rather than retried as a tiled 2×2 grid.
"""

from __future__ import annotations
import os
import io
import json
import time
import base64
import logging
import threading
from typing import List, Dict, Any, Optional
from collections import deque

from ocr_engine import (
    OCREngine, TextBlock, TextLine, LayoutBlock, BBox,
    TextDirection, LayoutType
)
from settings import poe_model

logger = logging.getLogger(__name__)

POE_API_BASE = "https://api.poe.com/v1"

# ── Language detection helpers ────────────────────────────────────────────────
CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x3000, 0x303F),
]
HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGE = (0x30A0, 0x30FF)
HANGUL_RANGE   = (0xAC00, 0xD7AF)
TRAD_CHARS = set("繁體傳統國語臺灣")


def _in_range(char: str, lo: int, hi: int) -> bool:
    return lo <= ord(char) <= hi


def _detect_lang_from_text(text: str) -> str:
    has_hiragana = any(_in_range(c, *HIRAGANA_RANGE) for c in text)
    has_katakana = any(_in_range(c, *KATAKANA_RANGE) for c in text)
    has_hangul   = any(_in_range(c, *HANGUL_RANGE)   for c in text)
    has_cjk      = any(
        any(_in_range(c, lo, hi) for lo, hi in CJK_RANGES)
        for c in text
    )
    has_trad = any(c in TRAD_CHARS for c in text)
    if has_hangul:
        return "korean"
    if has_hiragana or has_katakana:
        return "japan"
    if has_cjk:
        return "ch_tra" if has_trad else "ch_sim"
    return "en"


def _coerce_str(val) -> str:
    """Safely coerce any value to a string (handles list, None, etc.)."""
    if val is None:
        return ""
    if isinstance(val, list):
        return str(val[0]) if val else ""
    return str(val)


def _coerce_bbox(val) -> list:
    """Safely coerce bbox value to a 4-element list of numbers."""
    if not val:
        return [0, 0, 0, 0]
    if isinstance(val, list):
        if len(val) >= 4:
            try:
                return [float(v) for v in val[:4]]
            except (TypeError, ValueError):
                pass
        return [0, 0, 0, 0]
    if isinstance(val, dict):
        try:
            return [float(val.get("x0", 0)), float(val.get("y0", 0)),
                    float(val.get("x1", 0)), float(val.get("y1", 0))]
        except (TypeError, ValueError):
            return [0, 0, 0, 0]
    return [0, 0, 0, 0]


def _coerce_lines(val) -> list:
    """Coerce a block's 'lines' array into [{'text': str, 'bbox': list}, ...]."""
    if not isinstance(val, list):
        return []
    out = []
    for ln in val:
        if not isinstance(ln, dict):
            continue
        t = _coerce_str(ln.get("text"))
        if not t:
            continue
        out.append({"text": t, "bbox": _coerce_bbox(ln.get("bbox"))})
    return out


# ── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    """
    Sliding-window rate limiter. Blocks the calling thread until a request
    can be made without exceeding `max_rpm` requests in any 60-second window.
    """
    def __init__(self, max_rpm: int):
        self.max_rpm = max_rpm
        self.calls: deque = deque()
        self.lock = threading.Lock()

    def wait(self):
        """Block until a request can be issued without exceeding the limit."""
        while True:
            sleep_for = 0.0
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] > 60.0:
                    self.calls.popleft()
                if len(self.calls) >= self.max_rpm:
                    sleep_for = 60.0 - (now - self.calls[0]) + 0.1
                else:
                    return
            if sleep_for > 0:
                logger.info(f"Rate limit reached — sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)

    def record(self):
        """Record a successful request. Call AFTER the API call succeeds."""
        with self.lock:
            self.calls.append(time.monotonic())


# ── Language hint labels ──────────────────────────────────────────────────────
LANGUAGE_HINT_LABELS: Dict[str, str] = {
    "zh_tra_h": "Traditional Chinese (horizontal text)",
    "zh_tra_v": "Traditional Chinese (vertical text)",
    "zh_sim_h": "Simplified Chinese (horizontal text)",
    "zh_sim_v": "Simplified Chinese (vertical text)",
    "ja_h":     "Japanese (horizontal text, including hiragana, katakana, and kanji)",
    "ja_v":     "Japanese (vertical text, including hiragana, katakana, and kanji)",
    "ko_h":     "Korean (horizontal text)",
    "ko_v":     "Korean (vertical text)",
}

# ── OCR prompt — same schema gemini_engine.py uses, so both engines produce
#    output structure_analysis.py / pdf_assembly.py can consume identically ──
_OCR_PROMPT_BASE = """You are an OCR engine. Read this PDF page image and return ONLY a JSON object describing every text region you can see.

Return this exact JSON structure:
{
  "direction": "horizontal" or "vertical",
  "blocks": [
    {
      "text": "the full recognised text content of this block",
      "type": "heading" | "paragraph" | "list-item" | "footnote" | "page-number" | "caption",
      "bbox": [x0, y0, x1, y1],
      "lines": [
        { "text": "the text of one visual line", "bbox": [x0, y0, x1, y1] }
      ]
    }
  ]
}

Rules:
- Extract ALL text completely and without any omissions. Reproduce every character verbatim — do NOT paraphrase, summarise, or add any text that is not literally visible in the image. If a block is long, output the full text in a single block rather than truncating it.
- Preserve the original paragraph structure exactly as it appears on the page, including all punctuation, line breaks within blocks, and spacing that is meaningful to the text.
- "direction" indicates the dominant text flow on this page. Vertical text (typical in CJK literature) flows top-to-bottom, right-to-left.
- "bbox" is in pixel coordinates of THIS image, where (0,0) is top-left and values are integers.
- "lines" breaks the block into its individual VISUAL lines. For horizontal text each row of text is one line. For VERTICAL text each top-to-bottom column is one line. Each entry has the line's own "text" and a tight "bbox" in the same pixel coordinate system. The concatenation of all line texts, in order, MUST equal the block "text". If a block is a single visual line, or you cannot determine reliable per-line boxes, return an empty "lines" array for that block.
- Classify each region:
  * "heading": large title text, chapter/section headers
  * "paragraph": ordinary body text
  * "list-item": bullet or numbered list entry
  * "footnote": small text typically at the bottom of the page
  * "page-number": isolated page number, usually at top or bottom corner/center
  * "caption": text describing a figure or image
- Preserve the original language and script — do NOT translate.
- For vertical text, output the text in natural reading order (top-to-bottom within each column, columns ordered right-to-left).
- Order "blocks" in natural reading order for the page.
- If the page has no text (e.g. a full-page illustration), return {"direction":"horizontal","blocks":[]}.
- Return ONLY the JSON object, no commentary, no markdown fences, no explanations, no introductions, and no conclusions.
"""


class PoeOCREngine(OCREngine):
    """
    OCR engine backed by Poe's OpenAI-compatible external API.

    A single request per page (image + prompt, to /chat/completions) returns
    OCR text + layout classification + direction detection + per-line
    bounding boxes, using the same JSON schema gemini_engine.py asks for.
    """

    def __init__(self, config: dict):
        self.config      = config
        # POE_MODEL (env) wins over ocr.poe_model_name (config.yaml) so the
        # bot can be switched from the host's variables UI without a code
        # change. Resolves to "" if neither is set — load() then fails fast.
        self.model_name  = poe_model(config.get("poe_model_name"))
        self.api_key     = os.environ.get("POE_API_KEY", "").strip()
        self.max_retries = int(config.get("max_retries", 8))
        self.timeout_s   = int(config.get("request_timeout_s", 180))

        configured_rpm = int(config.get("rpm_limit", 60))
        # Poe's external API enforces a flat 500 requests/minute ceiling per
        # account regardless of plan — clamp so a value copied from the
        # Gemini section of config.yaml (e.g. 2000 on its paid tier) can't
        # cause a 429 storm when this engine is selected instead.
        self.rpm_limit = max(1, min(configured_rpm, 500))

        self._client        = None
        self._loaded        = False
        self._rate_limiter  = RateLimiter(self.rpm_limit)

        self._retry_backoff_base_s = float(config.get("retry_backoff_base_s", 4.0))
        self._retry_backoff_cap_s  = float(config.get("retry_backoff_cap_s", 120.0))

        self._page_cache: Dict[int, dict] = {}
        self._last_page_result: Optional[dict] = None
        self._primed_next_result: Optional[dict] = None
        self._language_hints: List[str] = []

    def load(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "POE_API_KEY environment variable is not set. "
                "Get a key at https://poe.com/api_key and add it to your "
                "host's environment variables (Railway: service → Variables)."
            )
        if not self.model_name:
            raise RuntimeError(
                "No Poe bot configured for OCR. Set the POE_MODEL environment "
                "variable (or ocr.poe_model_name in config.yaml) to a "
                "vision-capable bot name from your Poe account — e.g. "
                "'Claude-Sonnet-4.5' or 'GPT-5' are examples only; confirm "
                "the exact name at https://poe.com, since availability "
                "depends on your account."
            )
        import httpx
        logger.info(f"Initialising Poe client (model={self.model_name}, rpm={self.rpm_limit})…")
        self._client = httpx.Client(base_url=POE_API_BASE, timeout=self.timeout_s)
        self._loaded = True
        logger.info("Poe client ready.")

    # ── OCREngine interface ──────────────────────────────────────────────────

    def detect_language(self, page_image) -> str:
        result = self._analyse_page(page_image)
        all_text = "".join(_coerce_str(b.get("text")) for b in result.get("blocks", []))
        return _detect_lang_from_text(all_text)

    def detect_direction(self, page_image) -> TextDirection:
        result = self._analyse_page(page_image)
        d = _coerce_str(result.get("direction", "horizontal")).lower()
        return "vertical" if d == "vertical" else "horizontal"

    def recognize(self, page_image, direction: TextDirection) -> List[TextBlock]:
        result = self._analyse_page(page_image)
        blocks: List[TextBlock] = []
        for b in result.get("blocks", []):
            text = _coerce_str(b.get("text")).strip()
            if not text:
                continue
            bbox_raw = _coerce_bbox(b.get("bbox"))
            x0, y0, x1, y1 = bbox_raw
            bbox = BBox(x0, y0, x1, y1)
            lang = _detect_lang_from_text(text)

            tb_lines: List[TextLine] = []
            for ln in (b.get("lines") or []):
                if not isinstance(ln, dict):
                    continue
                lt = _coerce_str(ln.get("text")).strip()
                if not lt:
                    continue
                lx0, ly0, lx1, ly1 = _coerce_bbox(ln.get("bbox"))
                tb_lines.append(TextLine(text=lt, bbox=BBox(lx0, ly0, lx1, ly1)))

            if tb_lines:
                line_count = len(tb_lines)
            else:
                line_count = max(1, text.count('\n') + 1)

            if direction == "vertical":
                estimated_font_size = bbox.width / line_count
            else:
                estimated_font_size = bbox.height / line_count

            blocks.append(TextBlock(
                text=text,
                bbox=bbox,
                language=lang,
                font_size_estimate=estimated_font_size,
                confidence=1.0,
                direction=direction,
                lines=tb_lines,
            ))
        return blocks

    def get_layout(self, page_image) -> List[LayoutBlock]:
        result = self._analyse_page(page_image)
        layout_blocks: List[LayoutBlock] = []
        valid_types = {
            "heading", "paragraph", "list-item", "footnote",
            "page-number", "caption", "image"
        }
        for b in result.get("blocks", []):
            text = _coerce_str(b.get("text")).strip()
            if not text:
                continue
            raw_type = b.get("type")
            type_str = _coerce_str(raw_type).lower().strip()
            block_type: LayoutType = type_str if type_str in valid_types else "unknown"
            bbox_raw = _coerce_bbox(b.get("bbox"))
            x0, y0, x1, y1 = bbox_raw
            layout_blocks.append(LayoutBlock(
                block_type=block_type,
                bbox=BBox(x0, y0, x1, y1),
            ))
        return layout_blocks

    def health_check(self) -> bool:
        return self._loaded

    def set_language_hints(self, hints: List[str]) -> None:
        self._language_hints = [h for h in (hints or []) if h in LANGUAGE_HINT_LABELS]

    def _build_prompt(self) -> str:
        base = _OCR_PROMPT_BASE
        if not self._language_hints:
            return base
        labels = ", ".join(LANGUAGE_HINT_LABELS[h] for h in self._language_hints)
        hint_line = (
            f"- Language priority: This document is expected to primarily contain {labels}."
            " Prioritize recognition of these languages and scripts,"
            " but still recognize all text accurately.\n"
        )
        return base + hint_line

    def reset_page_cache(self):
        self._page_cache.clear()

    # ── Persistent cache hooks ───────────────────────────────────────────────

    def prime_page_cache_from_dict(self, cached_result: dict) -> None:
        if not isinstance(cached_result, dict):
            return
        self._primed_next_result = self._normalise_result(cached_result)

    def export_last_page_result(self) -> Optional[dict]:
        if self._last_page_result is None:
            return None
        return {
            "direction": _coerce_str(self._last_page_result.get("direction", "horizontal")),
            "blocks": [
                {
                    "text":  _coerce_str(b.get("text")),
                    "type":  _coerce_str(b.get("type")),
                    "bbox":  list(_coerce_bbox(b.get("bbox"))),
                    "lines": [
                        {
                            "text": _coerce_str(ln.get("text")),
                            "bbox": list(_coerce_bbox(ln.get("bbox"))),
                        }
                        for ln in (b.get("lines") or [])
                        if isinstance(ln, dict)
                    ],
                }
                for b in self._last_page_result.get("blocks", [])
                if isinstance(b, dict)
            ],
        }

    # ── Internal: single API call per page, cached ──────────────────────────

    def _analyse_page(self, page_image) -> dict:
        """
        Run one Poe API call for this page, cache the result, and return the
        parsed dict {"direction": ..., "blocks": [...]}.

        Reusing the cached result across detect_direction / recognize /
        get_layout means each PDF page costs exactly ONE Poe request.
        """
        self._assert_loaded()

        cache_key = id(page_image)
        if cache_key in self._page_cache:
            self._last_page_result = self._page_cache[cache_key]
            return self._last_page_result

        if self._primed_next_result is not None:
            result = self._primed_next_result
            self._primed_next_result = None
            self._page_cache[cache_key] = result
            self._last_page_result = result
            return result

        jpeg_bytes, scale = self._image_to_jpeg(page_image)
        result = self._call_poe_with_retry(jpeg_bytes)

        if scale < 1.0:
            inv_scale = 1.0 / scale
            self._upscale_bboxes(result, inv_scale)

        non_empty_blocks = [
            b for b in result.get("blocks", [])
            if _coerce_str(b.get("text")).strip()
        ]
        if not non_empty_blocks:
            h, w = page_image.shape[:2]
            logger.info(
                f"Poe returned 0 text blocks for this page (image {w}×{h}px). "
                f"Treated as image-only (genuinely blank or decorative) — "
                f"this engine has no tiling fallback for sparse/diagram pages."
            )

        self._page_cache[cache_key] = result
        self._last_page_result = result
        return result

    def _image_to_jpeg(self, page_image) -> tuple[bytes, float]:
        """
        Convert OpenCV BGR ndarray to JPEG bytes for the API call.
        Returns (jpeg_bytes, scale_factor).
        """
        from PIL import Image
        import numpy as np
        h, w = page_image.shape[:2]
        max_side = 2048
        scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            import cv2
            page_image = cv2.resize(page_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = page_image[:, :, ::-1]
        pil = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue(), scale

    @staticmethod
    def _upscale_bboxes(result: dict, inv_scale: float) -> None:
        """Multiply all block and line bboxes in-place by inv_scale."""
        for b in result.get("blocks", []):
            bbox = b.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                b["bbox"] = [v * inv_scale for v in bbox[:4]]
            for ln in (b.get("lines") or []):
                lb = ln.get("bbox")
                if isinstance(lb, list) and len(lb) >= 4:
                    ln["bbox"] = [v * inv_scale for v in lb[:4]]

    def _call_poe_with_retry(self, jpeg_bytes: bytes) -> dict:
        import httpx

        RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
        max_attempts = self.max_retries
        backoff_base = self._retry_backoff_base_s
        backoff_cap  = self._retry_backoff_cap_s

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_prompt()},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        attempt = 0
        last_exc: Exception = RuntimeError("no attempts made")
        last_code: Optional[int] = None

        while attempt < max_attempts:
            attempt += 1
            self._rate_limiter.wait()
            try:
                response = self._client.post("/chat/completions", json=payload, headers=headers)
            except httpx.HTTPError as e:
                last_exc, last_code = e, None
                if attempt >= max_attempts:
                    break
                wait = min(backoff_cap, backoff_base * attempt)
                logger.warning(f"Poe call failed: {e} — retry {attempt}/{max_attempts} in {wait:.1f}s")
                time.sleep(wait)
                continue

            if response.status_code >= 400:
                last_code = response.status_code
                last_exc = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                if response.status_code in RETRYABLE_CODES and attempt < max_attempts:
                    wait = self._retry_delay_seconds(response, attempt, backoff_base, backoff_cap)
                    logger.warning(
                        f"Poe API transient error (HTTP {response.status_code}) — "
                        f"retry {attempt}/{max_attempts} in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                # Non-retryable (400/401/403/404 …) or retries exhausted: fail fast.
                raise RuntimeError(
                    f"Poe API error (HTTP {response.status_code}): {response.text[:500]}"
                )

            self._rate_limiter.record()
            data = response.json()
            try:
                text = (data["choices"][0]["message"]["content"] or "").strip()
            except (KeyError, IndexError, TypeError):
                logger.warning(f"Unexpected Poe response shape: {str(data)[:300]}")
                text = ""
            return self._parse_response(text)

        code_str = f"HTTP {last_code} " if last_code else ""
        raise RuntimeError(
            f"Poe API failed after {attempt} attempts ({code_str}— last error: {last_exc})"
        )

    @staticmethod
    def _retry_delay_seconds(response, attempt: int, base: float, cap: float) -> float:
        """Honor a server Retry-After header if present, else exponential
        backoff with full jitter, capped at `cap`."""
        import random
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(cap, max(0.5, float(retry_after)))
            except ValueError:
                pass
        ceiling = min(cap, base * (2 ** (attempt - 1)))
        return random.uniform(base, max(base, ceiling))

    def _parse_response(self, text: str) -> dict:
        """Tolerantly parse the bot's JSON response."""
        if not text:
            return {"direction": "horizontal", "blocks": []}

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    logger.warning("Could not parse Poe response as JSON.")
                    return {"direction": "horizontal", "blocks": []}
            else:
                logger.warning("Poe response did not contain JSON.")
                return {"direction": "horizontal", "blocks": []}

        return self._normalise_result(data)

    def _normalise_result(self, data: Any) -> dict:
        """Defensively normalise a result dict (from API or from cache)."""
        if not isinstance(data, dict):
            return {"direction": "horizontal", "blocks": []}

        blocks = data.get("blocks", [])
        if not isinstance(blocks, list):
            blocks = []

        clean_blocks = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            clean_blocks.append({
                "text":  _coerce_str(b.get("text")),
                "type":  _coerce_str(b.get("type")),
                "bbox":  _coerce_bbox(b.get("bbox")),
                "lines": _coerce_lines(b.get("lines")),
            })

        return {
            "direction": _coerce_str(data.get("direction", "horizontal")),
            "blocks":    clean_blocks,
        }

    def _assert_loaded(self):
        if not self._loaded:
            raise RuntimeError("PoeOCREngine.load() must be called before use.")
