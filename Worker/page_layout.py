"""
Page Layout
===========
Places a page's OCR text where it sits on the original page, so the clean
PDF's text page mirrors the scan: same columns/rows, same line breaks, same
indents, headings and page numbers in the same spots.

`layout_page()` turns a StructuredPage's block and per-line boxes into a
list of `Glyphs` draw calls in PDF points (top-left origin). The renderers
(ReportLab canvas, PyMuPDF) only draw them, so both produce the same page.

Vertical text (CJK, top-to-bottom columns, right-to-left) follows book
typesetting:
- characters are stacked down each column from the column's own top, so
  indented first columns stay indented;
- small punctuation (，。、．) moves to the upper right of its cell;
- brackets, dashes, ellipses and the long-vowel mark are turned 90°;
- runs of Latin letters/digits (e.g. "(enlightenment)") are set sideways,
  except numbers/letters of at most two characters (a chapter number "2"),
  which stand upright across the column (縱中橫).
The Unicode text is never altered, so copy/search still give normal text.

When a page's boxes can't be trusted (missing, degenerate, off-page, or the
per-line texts don't add up to the block text), `layout_page()` returns None
and the caller falls back to the reflowed layout — no OCR text is ever lost.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from statistics import median
from typing import Callable, List, Optional, Tuple

from structure_analysis import StructuredPage, StructuredElement

logger = logging.getLogger(__name__)

# measure(text, size) → advance width in points
Measure = Callable[[str, float], float]
# ink(char) → glyph ink box in em units, baseline origin, y down (or None)
Ink = Callable[[str], Optional[Tuple[float, float, float, float]]]

# Font metrics approximations shared by the CJK and Noto fonts we embed.
_ASCENT = 0.88          # baseline sits this far below the em box top
_EM_CENTER = 0.38       # em box centre, above the baseline

# Vertical-writing character classes.
_SHIFT_PUNCT = set("，。、．")      # ：；！？ stay centred (TW style)
_ROTATE_CHARS = set("「」『』（）《》〈〉【】〔〕〖〗｛｝［］—―‐…‥～〜－ー＝")
_UPRIGHT_LIMIT = 0x2000  # below this (Latin, Greek, Cyrillic…) → sideways

# Characters that may not start a column/row (禁則): they hang at the end
# of the previous one instead.
_NO_LINE_START = set("，。、．：；！？」』）》〉】〕〗,.;:!?)]}")

# Box sanity limits.
_PAGE_TOLERANCE = 0.03   # boxes may poke this fraction past the page edge
_MIN_BOX_PT = 2.0


@dataclass
class Glyphs:
    """One draw call: `text` with its baseline origin at (x, y), in points."""
    text: str
    x: float
    y: float
    size: float
    rotate: int = 0      # 0, or 90 = turned clockwise (reads top-to-bottom)


Rect = Tuple[float, float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Box validation
# ─────────────────────────────────────────────────────────────────────────────

def _to_pt(bbox, px_to_pt: float, page_w: float, page_h: float) -> Optional[Rect]:
    """Pixel BBox → clamped point rect, or None when it is unusable."""
    if bbox is None:
        return None
    x0, y0 = min(bbox.x0, bbox.x1) * px_to_pt, min(bbox.y0, bbox.y1) * px_to_pt
    x1, y1 = max(bbox.x0, bbox.x1) * px_to_pt, max(bbox.y0, bbox.y1) * px_to_pt
    tol_w, tol_h = page_w * _PAGE_TOLERANCE, page_h * _PAGE_TOLERANCE
    if x0 < -tol_w or y0 < -tol_h or x1 > page_w + tol_w or y1 > page_h + tol_h:
        return None
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(page_w, x1), min(page_h, y1)
    if x1 - x0 < _MIN_BOX_PT or y1 - y0 < _MIN_BOX_PT:
        return None
    return (x0, y0, x1, y1)


def _squash(text: str) -> str:
    return "".join(text.split())


def _element_boxes(el: StructuredElement, px_to_pt, page_w, page_h):
    """
    (block rect, [(line text, line rect)…]) for an element. Line boxes are
    used only if every one is valid and their texts add up to the block text;
    otherwise the line list is empty and the block is wrapped inside its box.
    Returns None when the element has no usable box at all.
    """
    lines = []
    for ln in el.lines or []:
        t = (ln.text or "").strip()
        if not t:
            continue
        r = _to_pt(ln.bbox, px_to_pt, page_w, page_h)
        if r is None:
            lines = []
            break
        lines.append((t, r))
    if lines and _squash("".join(t for t, _ in lines)) != _squash(el.text or ""):
        lines = []

    block = _to_pt(el.bbox, px_to_pt, page_w, page_h)
    if block is None and lines:
        block = (min(r[0] for _, r in lines), min(r[1] for _, r in lines),
                 max(r[2] for _, r in lines), max(r[3] for _, r in lines))
    if block is None:
        return None
    return block, lines


# ─────────────────────────────────────────────────────────────────────────────
# Vertical text
# ─────────────────────────────────────────────────────────────────────────────

def _column_items(text: str, measure: Measure) -> List[Tuple[str, str, float]]:
    """
    Split one column's text into (kind, text, advance at size 1) items:
    'up' upright character, 'shift' small punctuation, 'side' sideways run,
    'tcy' short upright number/word set across the column.
    """
    items: List[Tuple[str, str, float]] = []
    side = ""

    def flush():
        nonlocal side
        s = side.strip()
        if s and len(s) <= 2 and s.isalnum():
            items.append(("tcy", s, 1.0))
        elif s:
            items.append(("side", s, measure(s, 1.0)))
        side = ""

    for ch in text:
        if ch.isspace():
            if side:
                side += ch
            continue
        if ord(ch) < _UPRIGHT_LIMIT:
            side += ch
            continue
        flush()
        if ch in _ROTATE_CHARS:
            items.append(("side", ch, measure(ch, 1.0)))
        elif ch in _SHIFT_PUNCT:
            items.append(("shift", ch, 1.0))
        else:
            items.append(("up", ch, 1.0))
    flush()
    return items


def _units(items) -> float:
    return sum(adv for _, _, adv in items)


def _place_column(items, xc: float, top: float, size: float, step: float,
                  measure: Measure, ink: Optional[Ink], out: List[Glyphs]) -> None:
    """Stack `items` down a column centred on xc, starting at `top`."""
    y = top
    for kind, text, adv in items:
        if kind == "side":
            out.append(Glyphs(text, xc - _EM_CENTER * size, y, size, rotate=90))
            y += adv * step
            continue
        w = measure(text, size)
        if kind == "tcy" and w > size:
            # Two digits must still fit across the column.
            glyph_size = size * size / w
            out.append(Glyphs(text, xc - size / 2, y + _ASCENT * size, glyph_size))
            y += step
            continue
        x = xc - w / 2
        base = y + _ASCENT * size
        if kind == "shift":
            # Vertical punctuation sits in the upper right of its cell. Fonts
            # place it differently (bottom-left in CN style, centred in TW
            # style), so move its actual ink there when we can measure it.
            box = ink(text) if ink else None
            if box:
                ix, iy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                x = xc + 0.22 * size - ix * size
                base = y + 0.28 * size - iy * size
            else:
                x += size * 0.5
                base -= size * 0.5
        out.append(Glyphs(text, x, base, size))
        y += step


def _layout_vertical(lines, block: Rect, text: str, measure: Measure,
                     ink: Optional[Ink], out: List[Glyphs]) -> None:
    if not lines:
        lines = _wrap_vertical(text, block, measure)
    cols = [(_column_items(t, measure), r) for t, r in lines]
    widths = [r[2] - r[0] for _, r in cols]
    # Size from full columns: column width, and height / characters.
    fits = [(r[3] - r[1]) / _units(it) for it, r in cols if _units(it) >= 4]
    size = median(widths)
    if fits:
        size = min(size, median(fits))
    for items, (x0, y0, x1, y1) in cols:
        units = _units(items)
        if not units:
            continue
        col_size = min(size, (y1 - y0) / units) if units * size > (y1 - y0) * 1.05 else size
        _place_column(items, (x0 + x1) / 2, y0, col_size, col_size, measure, ink, out)


def _source_lines(text: str) -> List[str]:
    """The block's own line breaks (the OCR keeps them), if it has any."""
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def _wrap_vertical(text: str, block: Rect, measure: Measure):
    """
    Right-to-left columns filling `block` when the OCR gave no line boxes:
    one column per line break in the OCR text when it has them (the page's
    own columns), otherwise wrapped to fit.
    """
    x0, y0, x1, y1 = block
    w, h = x1 - x0, y1 - y0
    src = _source_lines(text)
    if len(src) > 1:
        units = max(_units(_column_items(t, measure)) for t in src) or 1.0
        pitch = w / len(src)
        size = min(pitch * 0.9, h / units)
        return [(t, (x1 - (i + 1) * pitch, y0, x1 - i * pitch,
                     y0 + _units(_column_items(t, measure)) * size))
                for i, t in enumerate(src)]
    items = _column_items(text.replace("\n", ""), measure)
    units = _units(items) or 1.0
    size = min(w, h / units)                       # fits as a single column?
    if size < min(w, h) * 0.5:
        size = (w * h / (units * 1.3)) ** 0.5
    while True:
        per_col = max(1.0, h / size)
        cols, cur, acc = [], [], 0.0
        for it in items:
            if cur and acc + it[2] > per_col + 1e-6 and it[1][0] not in _NO_LINE_START:
                cols.append(cur)
                cur, acc = [], 0.0
            cur.append(it)
            acc += it[2]
        if cur:
            cols.append(cur)
        if (len(cols) - 1) * size * 1.3 + size <= w * 1.05 or size < 3:
            break
        size *= 0.95
    pitch = size * 1.3 if len(cols) > 1 else w
    lines = []
    for i, col in enumerate(cols):
        cx1 = x1 - i * pitch
        lines.append(("".join(t for _, t, _ in col), (cx1 - min(pitch, w), y0, cx1, y0 + _units(col) * size)))
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Horizontal text
# ─────────────────────────────────────────────────────────────────────────────

def _layout_horizontal(lines, block: Rect, text: str, measure: Measure,
                       out: List[Glyphs]) -> None:
    if not lines:
        lines = _wrap_horizontal(text, block, measure)
    heights = [r[3] - r[1] for _, r in lines]
    size = median(heights) * 0.85
    # Full rows set the size: width / text length.
    fits = [(r[2] - r[0]) / measure(t, 1.0) for t, r in lines
            if measure(t, 1.0) > 0 and (r[2] - r[0]) > (block[2] - block[0]) * 0.8]
    if fits:
        size = min(size, median(fits))
    for t, (x0, y0, x1, y1) in lines:
        row_size = size
        tw = measure(t, row_size)
        if tw > (x1 - x0) * 1.02 and tw > 0:
            row_size *= (x1 - x0) / tw
        out.append(Glyphs(t, x0, (y0 + y1) / 2 + _EM_CENTER * row_size, row_size))


def _wrap_horizontal(text: str, block: Rect, measure: Measure):
    """
    Rows filling `block` when the OCR gave no line boxes: one row per line
    break in the OCR text when it has them, otherwise wrapped to fit.
    """
    x0, y0, x1, y1 = block
    w, h = x1 - x0, y1 - y0
    src = _source_lines(text)
    if len(src) > 1:
        lead = h / len(src)
        return [(t, (x0, y0 + i * lead, x1, y0 + (i + 1) * lead)) for i, t in enumerate(src)]
    text = " ".join(text.split("\n"))
    one = measure(text, 1.0) or 1.0
    size = min(h * 0.85, w / one)                  # fits as a single row?
    if size < h * 0.4:
        size = (w * h / (one * 1.3)) ** 0.5
    while True:
        rows = _wrap_rows(text, size, w, measure)
        if (len(rows) - 1) * size * 1.3 + size <= h * 1.05 or size < 3:
            break
        size *= 0.95
    lead = size * 1.3 if len(rows) > 1 else h
    return [(t, (x0, y0 + i * lead, x1, y0 + (i + 1) * lead)) for i, t in enumerate(rows)]


def _wrap_rows(text: str, size: float, width: float, measure: Measure) -> List[str]:
    rows, cur, cur_w = [], "", 0.0
    for ch in text:
        cw = measure(ch, size)
        if cur and cur_w + cw > width and ch not in _NO_LINE_START:
            # Prefer breaking at the last space for space-separated scripts
            # (but don't leave a short row in CJK text with a stray space).
            cut = cur.rfind(" ")
            if cut > len(cur) * 0.5 and not ch.isspace():
                rows.append(cur[:cut])
                cur = cur[cut + 1:]
                cur_w = measure(cur, size)
            else:
                rows.append(cur)
                cur, cur_w = "", 0.0
            if ch.isspace():
                continue
        cur += ch
        cur_w += cw
    if cur.strip():
        rows.append(cur)
    return [r.strip() for r in rows if r.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def layout_page(page: StructuredPage, px_to_pt: float, page_w: float,
                page_h: float, measure: Measure,
                ink: Optional[Ink] = None) -> Optional[List[Glyphs]]:
    """
    Draw calls placing every OCR element of `page` at its original position,
    or None when the page's boxes are unusable (caller should reflow).
    """
    try:
        return _layout_page(page, px_to_pt, page_w, page_h, measure, ink)
    except Exception as e:  # never let one odd page break the whole PDF
        logger.warning(f"Page {page.page_number}: layout failed ({e}) — "
                       f"using reflowed layout for this page")
        return None


def _layout_page(page, px_to_pt, page_w, page_h, measure, ink):
    out: List[Glyphs] = []
    any_text = False
    for el in page.elements:
        text = (el.text or "").strip()
        if not text:
            continue
        any_text = True
        boxes = _element_boxes(el, px_to_pt, page_w, page_h)
        if boxes is None:
            logger.warning(f"Page {page.page_number}: element without a usable box "
                           f"({text[:20]!r}) — using reflowed layout for this page")
            return None
        block, lines = boxes
        vertical = el.direction == "vertical"
        # Direction is detected per page; on a vertical page, a block whose
        # lines (or, without lines, whose box) are wide and flat — a page
        # number, a running header — is horizontal text.
        if vertical:
            if lines:
                vertical = (median(r[3] - r[1] for _, r in lines)
                            >= median(r[2] - r[0] for _, r in lines))
            else:
                vertical = (block[2] - block[0]) <= (block[3] - block[1]) * 1.2
        if vertical:
            _layout_vertical(lines, block, text, measure, ink, out)
        else:
            _layout_horizontal(lines, block, text, measure, out)
    return out if any_text else []
