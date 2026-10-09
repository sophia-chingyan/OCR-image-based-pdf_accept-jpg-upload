"""
Font fallback chains for PDF assembly
=====================================
Every output PDF must show the OCR text correctly whatever its script, so
no renderer relies on a single font. Instead each one gets an ordered
*chain* of embeddable fonts and picks, per character, the first font in the
chain that has a glyph for it. Consecutive characters that resolve to the
same font are emitted as one run.

The chain's head depends on the document's dominant language (so Han
characters get the right regional glyph style and the body text gets a
book-like face); after that come the Noto fonts for every other script,
so a Thai word or a Russian name inside a Chinese page still renders.

Fonts are discovered on disk — the Docker image installs fonts-arphic-uming,
fonts-wqy-zenhei and fonts-noto-core under /usr/share/fonts. Extra
directories can be added with the FONT_DIRS environment variable
(os.pathsep-separated). Only TrueType-outline fonts (.ttf/.ttc) are used,
because those are what ReportLab can embed.

Two backends share the discovery and run-splitting logic:
- ReportLabFontChain — registers TTFont objects (embedded, subsetted).
- FitzFontChain      — fitz.Font objects for the PyMuPDF renderers.
"""

from __future__ import annotations
import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts"]

# (file name, face name inside a .ttc or None) — head of the chain per language.
_PRIMARY = {
    "ch_tra": [("uming.ttc", "AR PL UMing TW"), ("wqy-zenhei.ttc", "WenQuanYi Zen Hei")],
    "ch_sim": [("uming.ttc", "AR PL UMing CN"), ("wqy-zenhei.ttc", "WenQuanYi Zen Hei")],
    "japan":  [("wqy-zenhei.ttc", "WenQuanYi Zen Hei"), ("uming.ttc", "AR PL UMing TW")],
    "korean": [("wqy-zenhei.ttc", "WenQuanYi Zen Hei"), ("uming.ttc", "AR PL UMing TW")],
}
_DEFAULT_PRIMARY = [("NotoSerif-Regular.ttf", None), ("NotoSans-Regular.ttf", None)]
# CJK fonts that close the chain for non-CJK documents.
_CJK_TAIL = [("wqy-zenhei.ttc", "WenQuanYi Zen Hei"), ("uming.ttc", "AR PL UMing TW")]

FontSpec = Tuple[str, Optional[str]]          # (path, ttc face name)

# Font used first for Latin/Greek/Cyrillic and other non-CJK letters even
# when a CJK font heads the chain: CJK fonts' Latin glyphs are wide and
# monospaced, so Western words inside Chinese text are set the way books do.
_WESTERN_FONT = "NotoSerif-Regular.ttf"
# Below this code point a character is "western" (Latin … Cyrillic … Thai …);
# CJK punctuation, dashes and ellipses (U+2000 and up) stay with the CJK font.
_WESTERN_LIMIT = 0x2000


def _font_dirs() -> List[Path]:
    extra = [d for d in os.environ.get("FONT_DIRS", "").split(os.pathsep) if d]
    return [Path(d) for d in extra + _DEFAULT_FONT_DIRS]


@lru_cache(maxsize=None)
def _font_files() -> dict:
    """File name → full path for every .ttf/.ttc under the font dirs (first wins)."""
    found: dict = {}
    for d in _font_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in (".ttf", ".ttc") and p.name not in found:
                found[p.name] = str(p)
    return found


def _noto_script_files() -> List[str]:
    """
    Regular-weight Noto fonts for all other scripts: Sans before Serif before
    the rest (Kufi, Naskh, Looped…), so the plainest face wins per script.
    """
    names = [n for n in _font_files() if n.startswith("Noto") and n.endswith("-Regular.ttf")
             and "Emoji" not in n and "Mono" not in n]

    def rank(n: str) -> tuple:
        if n.startswith("NotoSans"):
            return (0, n)
        if n.startswith("NotoSerif"):
            return (1, n)
        return (2, n)
    return sorted(names, key=rank)


@lru_cache(maxsize=None)
def font_specs(language: str) -> Tuple[FontSpec, ...]:
    """Ordered (path, ttc face) chain for `language`, de-duplicated, existing files only."""
    files = _font_files()
    wanted: List[Tuple[str, Optional[str]]] = []
    wanted += _PRIMARY.get(language, _DEFAULT_PRIMARY)
    wanted += [("NotoSans-Regular.ttf", None), ("NotoSerif-Regular.ttf", None)]
    wanted += [(n, None) for n in _noto_script_files()]
    wanted += _CJK_TAIL
    specs: List[FontSpec] = []
    for name, face in wanted:
        path = files.get(name)
        if path and (path, face) not in specs:
            specs.append((path, face))
    if not specs:
        logger.warning("No embeddable fonts found in %s", [str(d) for d in _font_dirs()])
    return tuple(specs)


def _ttc_index(path: str, face: Optional[str]) -> int:
    """Subfont index of `face` inside a .ttc (0 for .ttf or when not found)."""
    if not face or not path.lower().endswith(".ttc"):
        return 0
    from reportlab.pdfbase.ttfonts import TTFontFile
    i = 0
    while True:
        try:
            f = TTFontFile(path, subfontIndex=i)
        except Exception:
            return 0
        fam = getattr(f, "familyName", b"")
        if isinstance(fam, bytes):
            fam = fam.decode("latin-1", "replace")
        if fam == face:
            return i
        i += 1


class _FontChain:
    """
    Backend-neutral per-character font selection. Subclasses supply
    `_count()`, `_has_glyph(i, ch)`; fonts are loaded lazily, so a document
    in one script only ever loads the first font or two.
    """

    def __init__(self) -> None:
        self._char_cache: dict = {}
        self._western: Optional[int] = None   # index of _WESTERN_FONT, if any

    def _set_western(self, paths: List[Optional[str]]) -> None:
        for i, path in enumerate(paths):
            if path and Path(path).name == _WESTERN_FONT:
                self._western = i
                return

    def _count(self) -> int:
        raise NotImplementedError

    def _has_glyph(self, i: int, ch: str) -> bool:
        raise NotImplementedError

    def index_for(self, ch: str) -> int:
        """Index of the first font that has `ch` (0 when none does)."""
        idx = self._char_cache.get(ch)
        if idx is None:
            idx = 0
            order = list(range(self._count()))
            if self._western is not None and ord(ch) < _WESTERN_LIMIT:
                order.insert(0, self._western)
            for i in order:
                try:
                    if self._has_glyph(i, ch):
                        idx = i
                        break
                except Exception as e:
                    logger.debug("font %d unusable: %s", i, e)
            self._char_cache[ch] = idx
        return idx

    def runs(self, text: str) -> List[Tuple[int, str]]:
        """
        Split `text` into (font index, substring) runs. Whitespace stays with
        the run before it, so spaces between words don't fragment runs.
        """
        out: List[Tuple[int, str]] = []
        for ch in text:
            if out and ch.isspace():
                i = out[-1][0]
            else:
                i = self.index_for(ch)
            if out and out[-1][0] == i:
                out[-1] = (i, out[-1][1] + ch)
            else:
                out.append((i, ch))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# ReportLab backend
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _reportlab_font(path: str, face: Optional[str]):
    """
    Register (once per process) and return the TTFont for one font file/face.
    ReportLab's font registry is global, so each file/face gets its own
    unique name and is shared by every job.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    name = "OCR-" + Path(path).stem + (f"-{face}" if face else "")
    name = "".join(c if c.isalnum() or c == "-" else "_" for c in name)
    font = TTFont(name, path, subfontIndex=_ttc_index(path, face))
    pdfmetrics.registerFont(font)
    logger.info("ReportLab font registered: %s", name)
    return font


class ReportLabFontChain(_FontChain):
    """TTFont chain for ReportLab. Fonts are embedded (subsetted) when used."""

    def __init__(self, language: str) -> None:
        super().__init__()
        self.specs = font_specs(language)
        if not self.specs:
            raise RuntimeError("no embeddable TrueType fonts available")
        self.primary = self.name(0)
        self._set_western([path for path, _face in self.specs])

    def _count(self) -> int:
        return len(self.specs)

    def _load(self, i: int):
        return _reportlab_font(*self.specs[i])

    def _has_glyph(self, i: int, ch: str) -> bool:
        return ord(ch) in self._load(i).face.charToGlyph

    def name(self, i: int) -> str:
        return self._load(i).fontName

    def markup(self, text: str, escape: Callable[[str], str]) -> str:
        """Escaped Paragraph markup with <font> tags around non-primary runs."""
        parts = []
        for i, run in self.runs(text):
            safe = escape(run)
            parts.append(safe if i == 0 else f'<font name="{self.name(i)}">{safe}</font>')
        return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PyMuPDF backend
# ─────────────────────────────────────────────────────────────────────────────

class FitzFontChain(_FontChain):
    """
    fitz.Font chain: PyMuPDF's built-in CJK font for the language first
    (embedded by PyMuPDF), then the .ttf files from the shared chain. .ttc
    files are skipped — fitz.Font can't pick a subfont — the built-in font
    already covers CJK.
    """

    def __init__(self, builtin_name: str, language: str) -> None:
        super().__init__()
        self._sources: List[dict] = [{"fontname": builtin_name}]
        for path, _face in font_specs(language):
            if path.lower().endswith(".ttf"):
                self._sources.append({"fontfile": path})
        self._fonts: List[Optional[object]] = [None] * len(self._sources)
        self._set_western([src.get("fontfile") for src in self._sources])

    def _count(self) -> int:
        return len(self._sources)

    def font(self, i: int):
        if self._fonts[i] is None:
            import fitz
            self._fonts[i] = fitz.Font(**self._sources[i])
        return self._fonts[i]

    def _has_glyph(self, i: int, ch: str) -> bool:
        return self.font(i).has_glyph(ord(ch)) != 0

    def font_for(self, ch: str):
        return self.font(self.index_for(ch))

    def text_length(self, text: str, fontsize: float) -> float:
        return sum(self.font(i).text_length(run, fontsize=fontsize)
                   for i, run in self.runs(text))

    def append(self, tw, pos, text: str, fontsize: float):
        """Append `text` to TextWriter `tw` run by run; returns the end point."""
        import fitz
        point = fitz.Point(pos)
        for i, run in self.runs(text):
            _, point = tw.append(point, run, font=self.font(i), fontsize=fontsize)
        return point
