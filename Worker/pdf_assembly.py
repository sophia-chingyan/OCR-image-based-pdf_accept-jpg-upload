"""
PDF Assembly
============
Two output modes:

Clean PDF (方案 A) — assemble_clean_pdf()
   Re-renders OCR text into a cleanly typeset PDF using ReportLab.
   Layout: for each source page, emit (1) a full-page image of the original
   scan at 300 DPI, followed by (2) a text-only OCR page (no images).

Searchable PDF (方案 B) — assemble_searchable_pdf()
   Keeps the original scanned pages pixel-for-pixel, and overlays an
   INVISIBLE OCR text layer using PyMuPDF render_mode=3 so the text is
   selectable, copyable, and searchable without altering the appearance.
   Per-line bboxes (horizontal) and per-character-column distribution
   (vertical CJK) give tight selection alignment.

Font selection:
- Traditional Chinese → MSung-Light (ReportLab) / china-t (PyMuPDF)
- Simplified Chinese  → STSong-Light / china-s
- Japanese            → HeiseiMin-W3 / japan
- Korean              → HYSMyeongJo-Medium / korea
"""

from __future__ import annotations
import io
import re
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import fitz  # PyMuPDF

from structure_analysis import DocumentStructure, StructuredPage

logger = logging.getLogger(__name__)

# DPI settings for clean PDF scan-page embedding
_SCAN_DPI = 300   # original scan pages rasterised at this DPI


# ─────────────────────────────────────────────────────────────────────────────
# Font selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_fitz_font_name(language: str) -> str:
    lang_map = {
        "ch_tra": "china-t",
        "ch_sim": "china-s",
        "japan":  "japan",
        "korean": "korea",
    }
    result = lang_map.get(language, "china-t")
    logger.info(f"PyMuPDF font for language '{language}': {result}")
    return result


def _register_best_font(pdfmetrics, UnicodeCIDFont, language: str = "ch_tra") -> str:
    if language == "ch_sim":
        candidates = ["STSong-Light", "MSung-Light", "HeiseiMin-W3", "HYSMyeongJo-Medium"]
    elif language == "japan":
        candidates = ["HeiseiMin-W3", "MSung-Light", "STSong-Light", "HYSMyeongJo-Medium"]
    elif language == "korean":
        candidates = ["HYSMyeongJo-Medium", "MSung-Light", "STSong-Light", "HeiseiMin-W3"]
    else:
        candidates = ["MSung-Light", "STSong-Light", "HeiseiMin-W3", "HYSMyeongJo-Medium"]

    for fname in candidates:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(fname))
            logger.info(f"ReportLab CID font registered: {fname} (language={language})")
            return fname
        except Exception:
            continue
    logger.warning(f"No CJK CID font registered for language={language}, falling back to Helvetica")
    return "Helvetica"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: rasterise one source page to JPEG bytes
# ─────────────────────────────────────────────────────────────────────────────

def _rasterise_source_page_jpeg(src_doc: fitz.Document, page_num: int, dpi: int) -> Optional[bytes]:
    """
    Rasterise page_num of src_doc at dpi and return JPEG bytes.
    Returns None on any error so callers can skip gracefully.
    """
    try:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = src_doc[page_num].get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("jpeg")
    except Exception as e:
        logger.warning(f"Could not rasterise source page {page_num} at {dpi} DPI: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 方案 A: Clean PDF — reflowed text with proper typography
# ─────────────────────────────────────────────────────────────────────────────

def assemble_clean_pdf(
    structure: DocumentStructure,
    output_path: Path,
    source_pdf_path: Optional[Path] = None,
) -> None:
    """
    Re-render OCR text into a cleanly typeset PDF.

    For each source page the output contains two consecutive pages:
      1. The original scan rasterised at 300 DPI (full-frame image).
      2. A text-only OCR page with typeset paragraphs/headings (no images).

    Tries ReportLab first; falls back to PyMuPDF on any failure.
    """
    logger.info(f"Assembling clean PDF: {output_path}")
    try:
        _assemble_clean_pdf_reportlab(structure, output_path, source_pdf_path)
        logger.info(f"Clean PDF written (ReportLab): {output_path} "
                     f"({output_path.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"ReportLab build failed ({e}), falling back to PyMuPDF renderer")
        try:
            _assemble_clean_pdf_pymupdf(structure, output_path, source_pdf_path)
            logger.info(f"Clean PDF written (PyMuPDF fallback): {output_path} "
                         f"({output_path.stat().st_size/1024:.1f} KB)")
        except Exception as e2:
            logger.error(f"PyMuPDF fallback also failed ({e2}), writing minimal PDF")
            _write_minimal_pdf(output_path, structure.title or "Untitled",
                               f"PDF assembly error: {e}")


def _assemble_clean_pdf_reportlab(
    structure: DocumentStructure,
    output_path: Path,
    source_pdf_path: Optional[Path],
) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    cjk_font = _register_best_font(pdfmetrics, UnicodeCIDFont,
                                    language=structure.dominant_language)
    logger.info(f"Clean PDF using font: {cjk_font}")

    styles = getSampleStyleSheet()

    def _style(name, parent_name="Normal", **kw):
        parent = styles.get(parent_name, styles["Normal"])
        return ParagraphStyle(name, parent=parent, fontName=cjk_font, **kw)

    s_title = _style("T", "Title",   fontSize=18, leading=24, spaceAfter=12, alignment=TA_CENTER)
    s_h1    = _style("H1","Heading1",fontSize=16, leading=22, spaceBefore=14, spaceAfter=8)
    s_h2    = _style("H2","Heading2",fontSize=14, leading=19, spaceBefore=10, spaceAfter=6)
    s_h3    = _style("H3","Heading3",fontSize=12, leading=17, spaceBefore=8,  spaceAfter=4)
    s_body  = _style("B", fontSize=11, leading=18, firstLineIndent=22,
                     spaceBefore=2, spaceAfter=2, alignment=TA_JUSTIFY)
    # Rest of a paragraph that began on the previous source page — no
    # first-line indent, so it doesn't read as a new paragraph.
    s_cont  = _style("BC", "B", firstLineIndent=0)
    s_fn    = _style("FN",fontSize=9,  leading=13, textColor="#555555")
    s_pn    = _style("PN",fontSize=9,  leading=12, textColor="#888888", alignment=TA_CENTER)
    s_cap   = _style("C", fontSize=10, leading=14, textColor="#666666", alignment=TA_CENTER)
    s_li    = _style("LI",fontSize=11, leading=18, leftIndent=20, bulletIndent=10)
    s_auth  = _style("A", fontSize=12, leading=18, alignment=TA_CENTER)
    hs = {1: s_h1, 2: s_h2, 3: s_h3}

    A4_W, A4_H = A4  # 595.28 pt × 841.89 pt
    SIDE    = 25 * mm
    TOP     = 20 * mm
    # Frame dimensions — scan images are sized to fill this area
    frame_w = A4_W - 2 * SIDE
    frame_h = A4_H - 2 * TOP

    # Open source PDF for scan page rasterisation (optional)
    src_doc: Optional[fitz.Document] = None
    if source_pdf_path is not None:
        try:
            src_doc = fitz.open(str(source_pdf_path))
        except Exception as e:
            logger.warning(f"Could not open source PDF for scan embedding: {e}")

    try:
        story: list = []

        # ── Title page ───────────────────────────────────────────────────────
        if structure.title:
            story.append(Spacer(1, 40 * mm))
            story.append(Paragraph(_esc(structure.title), s_title))
            if structure.author:
                story.append(Spacer(1, 5 * mm))
                story.append(Paragraph(_esc(structure.author), s_auth))

        has_any_page = False
        page_items = _prepare_clean_text(structure.pages)

        for page, items in zip(structure.pages, page_items):
            pno = page.page_number

            # ── 1. Original scan page ────────────────────────────────────────
            if src_doc is not None and 0 <= pno < src_doc.page_count:
                scan_jpeg = _rasterise_source_page_jpeg(src_doc, pno, _SCAN_DPI)
                if scan_jpeg is not None:
                    if story:
                        story.append(PageBreak())
                    story.append(RLImage(io.BytesIO(scan_jpeg),
                                         width=frame_w, height=frame_h,
                                         kind="proportional"))
                    has_any_page = True

            # ── 2. OCR text-only page ────────────────────────────────────────
            text_items: list = []
            for el in items:
                t = el.text
                safe = _esc(t)
                try:
                    if el.element_type == "heading":
                        if el.href:
                            safe = f'<a href="{_esc(el.href)}" color="blue">{safe}</a>'
                        text_items.append(Paragraph(safe, hs.get(min(el.level, 3), s_h3)))
                    elif el.element_type == "paragraph":
                        if el.href:
                            safe = f'<a href="{_esc(el.href)}" color="blue">{safe}</a>'
                        text_items.append(Paragraph(safe, s_cont if el.continued else s_body))
                    elif el.element_type == "list-item":
                        bullet_safe = f"\u2022 {safe}"
                        if el.href:
                            bullet_safe = f'<a href="{_esc(el.href)}" color="blue">{bullet_safe}</a>'
                        text_items.append(Paragraph(bullet_safe, s_li))
                    elif el.element_type == "footnote":
                        text_items.append(Paragraph(safe, s_fn))
                    elif el.element_type == "page-number":
                        text_items.append(Paragraph(safe, s_pn))
                    elif el.element_type == "caption":
                        text_items.append(Paragraph(safe, s_cap))
                    else:
                        text_items.append(Paragraph(safe, s_body))
                except Exception as e:
                    logger.warning(f"Skipping element: {e} — text: {t[:50]!r}")

            if text_items:
                story.append(PageBreak())
                story.extend(text_items)
                has_any_page = True

        # ── Fallback: nothing produced ───────────────────────────────────────
        if not has_any_page:
            if story:
                story.append(Spacer(1, 10 * mm))
            story.append(Paragraph(
                "[ No body text was extracted from this PDF ]", s_body
            ))

        if not story:
            story.append(Paragraph(_esc(structure.title or "Untitled"), s_title))
            story.append(Spacer(1, 10 * mm))
            story.append(Paragraph(
                "[ No text content could be extracted from this PDF ]", s_body
            ))

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=SIDE, rightMargin=SIDE,
            topMargin=TOP,   bottomMargin=TOP,
            title=structure.title or "Untitled",
            author=structure.author or "",
        )
        doc.build(story)

    finally:
        if src_doc is not None:
            try:
                src_doc.close()
            except Exception:
                pass


def _assemble_clean_pdf_pymupdf(
    structure: DocumentStructure,
    output_path: Path,
    source_pdf_path: Optional[Path],
) -> None:
    """
    PyMuPDF fallback for clean PDF assembly.
    Emits: title page → for each source page: scan page (300 DPI) + text-only page.
    """
    doc = fitz.open()
    fitz_font_name = _get_fitz_font_name(structure.dominant_language)
    font = fitz.Font(fitz_font_name)

    title  = structure.title or "Untitled"
    author = structure.author or ""

    # ── Title page ───────────────────────────────────────────────────────────
    title_page = doc.new_page(width=595, height=842)
    try:
        tw = fitz.TextWriter(title_page.rect)
        tw.append(pos=(72, 200), text=title[:100], font=font, fontsize=20)
        if author:
            tw.append(pos=(72, 240), text=author[:100], font=font, fontsize=14)
        tw.write_text(title_page)
    except Exception:
        title_page.insert_text((72, 200), title[:100], fontsize=20)
        if author:
            title_page.insert_text((72, 240), author[:100], fontsize=14)

    # Open source PDF for scan rasterisation
    src_doc: Optional[fitz.Document] = None
    if source_pdf_path is not None:
        try:
            src_doc = fitz.open(str(source_pdf_path))
        except Exception as e:
            logger.warning(f"PyMuPDF fallback: could not open source PDF: {e}")

    try:
        has_any_content = False
        page_items = _prepare_clean_text(structure.pages)

        for struct_page, text_elements in zip(structure.pages, page_items):
            pno = struct_page.page_number

            # ── 1. Original scan page ────────────────────────────────────────
            if src_doc is not None and 0 <= pno < src_doc.page_count:
                scan_jpeg = _rasterise_source_page_jpeg(src_doc, pno, _SCAN_DPI)
                if scan_jpeg is not None:
                    scan_page = doc.new_page(width=595, height=842)
                    try:
                        scan_page.insert_image(scan_page.rect, stream=scan_jpeg)
                        has_any_content = True
                    except Exception as e:
                        logger.warning(f"PyMuPDF: could not insert scan image for page {pno}: {e}")

            # ── 2. OCR text-only page ────────────────────────────────────────
            if not text_elements:
                continue

            text_page   = doc.new_page(width=595, height=842)
            y_cursor    = 60.0
            margin_left = 50.0
            max_width   = 495.0

            for el in text_elements:
                text = el.text

                if el.element_type == "heading":
                    fs = 16 if el.level == 1 else 14 if el.level == 2 else 12
                    y_cursor += 8
                elif el.element_type == "footnote":
                    fs = 9
                elif el.element_type == "page-number":
                    fs = 8
                elif el.element_type == "caption":
                    fs = 10
                else:
                    fs = 11

                lines = _wrap_text_fitz(text, font, fs, max_width)
                for line in lines:
                    if y_cursor > 790:
                        text_page = doc.new_page(width=595, height=842)
                        y_cursor = 60.0
                    try:
                        tw = fitz.TextWriter(text_page.rect)
                        tw.append(pos=(margin_left, y_cursor), text=line,
                                  font=font, fontsize=fs)
                        tw.write_text(text_page)
                    except Exception:
                        try:
                            text_page.insert_text((margin_left, y_cursor),
                                                  line[:200], fontsize=fs)
                        except Exception:
                            pass
                    y_cursor += fs * 1.5
                y_cursor += 4
                has_any_content = True

        if not has_any_content:
            fallback_page = doc[0]
            try:
                tw = fitz.TextWriter(fallback_page.rect)
                tw.append(pos=(72, 300),
                          text="[ No text content could be extracted ]",
                          font=font, fontsize=12)
                tw.write_text(fallback_page)
            except Exception:
                fallback_page.insert_text((72, 300),
                                           "[ No text content could be extracted ]",
                                           fontsize=12)

        doc.save(str(output_path), garbage=4, deflate=True)

    finally:
        doc.close()
        if src_doc is not None:
            try:
                src_doc.close()
            except Exception:
                pass


def _wrap_text_fitz(text: str, font, fontsize: float, max_width: float) -> List[str]:
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            lines.append("")
            continue
        current = ""
        for char in raw_line:
            test = current + char
            try:
                w = font.text_length(test, fontsize=fontsize)
            except Exception:
                w = len(test) * fontsize * 0.6
            if w > max_width and current:
                lines.append(current)
                current = char
            else:
                current = test
        if current:
            lines.append(current)
    return lines if lines else [""]


# ─────────────────────────────────────────────────────────────────────────────
# Clean-PDF text preparation: join visual lines, stitch cross-page paragraphs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CleanItem:
    """One typeset element on a clean-PDF text page."""
    element_type: str
    text: str
    level: int = 1
    href: Optional[str] = None
    continued: bool = False  # tail of a paragraph begun on the previous page


# Characters that end a sentence. A paragraph whose text ends with none of
# these at the bottom of a page continues on the next page.
_SENTENCE_END = set("。！？!?…；;：:.")
# Closing quotes/brackets that may follow a sentence-ending character.
_CLOSERS = set("」』”’）)】〉》\"'")
# Elements that sit between the body text of two pages and are skipped when
# looking for the paragraph that runs across the page break.
_MARGIN_TYPES = ("page-number", "footnote")


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF
            or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF
            or 0x20000 <= o <= 0x2FA1F)


def _join_visual_lines(text: str) -> str:
    """
    Join the visual lines (rows, or columns in vertical text) of one OCR
    block into continuous text. The OCR keeps the page's line breaks, but in
    the reflowed clean PDF they are just line-wrap points: rendered as-is
    they become stray spaces between CJK characters (ReportLab) or short
    broken lines (PyMuPDF). A space is kept only between two non-CJK words.
    """
    parts = [ln.strip() for ln in text.split("\n")]
    out = ""
    for part in parts:
        if not part:
            continue
        if out and not _is_cjk(out[-1]) and not _is_cjk(part[0]):
            out += " "
        out += part
    return out


def _ends_sentence(text: str) -> bool:
    t = text.rstrip()
    while t and t[-1] in _CLOSERS:
        t = t[:-1]
    return bool(t) and t[-1] in _SENTENCE_END


def _first_sentence_end(text: str) -> int:
    """Index just past the first sentence end (and its closing quotes), or -1."""
    for i, ch in enumerate(text):
        if ch in _SENTENCE_END:
            j = i + 1
            while j < len(text) and (text[j] in _SENTENCE_END or text[j] in _CLOSERS):
                j += 1
            return j
    return -1


def _prepare_clean_text(pages: List[StructuredPage]) -> List[List[_CleanItem]]:
    """
    Build the per-page text items for the clean PDF.

    The clean PDF puts each source page's text on its own page, right after
    that page's scan. A paragraph that runs across a page break used to be
    cut in two there: the sentence split mid-way, the scan of the next page
    sat between the halves, and the second half was indented as if it were a
    new paragraph. Now the unfinished sentence is completed on the page where
    it starts (the next page's text up to its first sentence end is moved
    back), and whatever remains of that paragraph is marked `continued` so it
    is typeset without a first-line indent.
    """
    result: List[List[_CleanItem]] = []
    open_para: Optional[_CleanItem] = None  # unfinished paragraph from the previous page
    prev_pno: Optional[int] = None

    for page in pages:
        items: List[_CleanItem] = []
        for el in page.elements:
            text = _join_visual_lines(el.text or "")
            if not text:
                continue
            items.append(_CleanItem(
                element_type=el.element_type,
                text=text,
                level=el.level,
                href=el.href,
            ))

        # A page skipped by OCR breaks the chain — don't stitch across it.
        if open_para is not None and prev_pno is not None and page.page_number == prev_pno + 1:
            first = next((it for it in items if it.element_type not in _MARGIN_TYPES), None)
            if first is not None and first.element_type == "paragraph" and not first.href:
                cut = _first_sentence_end(first.text)
                if cut < 0:
                    cut = len(first.text)
                open_para.text += first.text[:cut]
                rest = first.text[cut:].lstrip()
                if rest:
                    first.text = rest
                    first.continued = True
                else:
                    items.remove(first)

        result.append(items)

        # Remember this page's last body paragraph if it is left unfinished.
        last = next((it for it in reversed(items) if it.element_type not in _MARGIN_TYPES), None)
        if last is not None and last.element_type == "paragraph" and not last.href \
                and not _ends_sentence(last.text):
            open_para = last
        elif last is not None or not page.is_image_only:
            open_para = None
        prev_pno = page.page_number

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 方案 B: Searchable PDF — original scan + invisible OCR text layer
# ─────────────────────────────────────────────────────────────────────────────

def assemble_searchable_pdf(
    structure: DocumentStructure,
    source_pdf_path: Path,
    output_path: Path,
    dpi: int = 400,
) -> None:
    """
    Produce a "searchable PDF": the original scanned pages are preserved
    exactly, with an INVISIBLE OCR text layer overlaid on top so the text
    becomes selectable / copyable / searchable while the document still looks
    identical to the original scan.

    Per-line bboxes (horizontal text) and per-character-column distribution
    (vertical CJK text) give tight selection alignment that matches each
    visible text line rather than the whole paragraph block.

    Coordinate systems
    -------------------
    OCR bboxes on each StructuredElement / TextLine are in *pixel* space at
    the rasterization DPI (default 400). PDF content is in *points* (72/inch).
    We convert pixels → points with the factor 72/dpi.

    Robustness
    ----------
    If the overlay step fails, we fall back to copying the original PDF so
    the user still gets a valid, viewable file.
    """
    source_pdf_path = Path(source_pdf_path)
    logger.info(f"Assembling searchable PDF: {output_path}")
    try:
        _assemble_searchable_pdf_impl(structure, source_pdf_path, output_path, dpi)
        logger.info(f"Searchable PDF written: {output_path} "
                    f"({output_path.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"Searchable overlay failed ({e}); "
                       f"falling back to a copy of the original PDF")
        try:
            shutil.copyfile(str(source_pdf_path), str(output_path))
        except Exception as e2:
            logger.error(f"Could not copy original PDF ({e2}); writing minimal PDF")
            _write_minimal_pdf(output_path, structure.title or "Untitled",
                               f"Searchable PDF assembly error: {e}")


def _assemble_searchable_pdf_impl(
    structure: DocumentStructure,
    source_pdf_path: Path,
    output_path: Path,
    dpi: int,
) -> None:
    doc = fitz.open(str(source_pdf_path))

    try:
        font = fitz.Font(_get_fitz_font_name(structure.dominant_language))
    except Exception:
        font = fitz.Font("china-t")

    px_to_pt  = 72.0 / float(dpi if dpi else 400)
    page_map  = {p.page_number: p for p in structure.pages}
    overlaid  = 0

    for pno in range(doc.page_count):
        sp = page_map.get(pno)
        if sp is None:
            continue
        page      = doc[pno]
        page_rect = page.rect

        for el in sp.elements:
            text = (el.text or "").strip()
            if not text:
                continue
            direction = el.direction or "horizontal"

            # Prefer per-line overlay for tight selection alignment.
            used_lines = False
            for ln in (el.lines or []):
                lt = (ln.text or "").strip()
                if not lt or ln.bbox is None:
                    continue
                rect = _clamp_px_rect(ln.bbox, px_to_pt, page_rect)
                if rect is None:
                    continue
                if direction == "vertical":
                    ok = _overlay_vertical_line(page, rect, lt, font)
                else:
                    ok = _overlay_horizontal_line(page, rect, lt, font)
                if ok:
                    overlaid += 1
                    used_lines = True

            if used_lines:
                continue  # avoid duplicating text with a block-level pass

            # Fall back to block-level overlay: single-line blocks, or when
            # the model returned no usable per-line boxes.
            if el.bbox is None:
                continue
            rect = _clamp_px_rect(el.bbox, px_to_pt, page_rect)
            if rect is None:
                continue
            if _overlay_invisible_text(page, rect, text, font):
                overlaid += 1

    logger.info(f"Searchable PDF: overlaid {overlaid} text segment(s) "
                f"across {doc.page_count} page(s)")
    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()


def _clamp_px_rect(bbox, px_to_pt: float, page_rect):
    """Pixel-space BBox → clamped PDF-point fitz.Rect, or None if degenerate."""
    r = fitz.Rect(
        bbox.x0 * px_to_pt,
        bbox.y0 * px_to_pt,
        bbox.x1 * px_to_pt,
        bbox.y1 * px_to_pt,
    )
    r.normalize()
    r = r & page_rect
    if r.is_empty or r.width <= 1 or r.height <= 1:
        return None
    return r


def _overlay_horizontal_line(page, rect, text: str, font) -> bool:
    """
    Invisible single-line overlay sized to the line box, shrunk to fit width.
    Selection highlight tracks each individual text line precisely.
    """
    try:
        text = text.strip()
        if not text:
            return False
        fs = max(3.0, min(rect.height, 48.0))
        try:
            w = font.text_length(text, fontsize=fs)
            if w > rect.width and w > 0:
                fs = max(3.0, fs * (rect.width / w))
        except Exception:
            pass
        # Baseline slightly above the box bottom edge.
        baseline = rect.y1 - rect.height * 0.18
        tw = fitz.TextWriter(page.rect)
        tw.append(pos=(rect.x0, baseline), text=text, font=font, fontsize=fs)
        tw.write_text(page, render_mode=3)   # invisible but selectable
        return True
    except Exception as e:
        logger.debug(f"horizontal line overlay failed: {e}")
        return False


def _overlay_vertical_line(page, rect, text: str, font) -> bool:
    """
    Invisible per-character overlay distributed down a vertical column.
    Each character is placed at equal steps so drag-selection in CJK
    vertical text lands on the correct individual glyphs.
    """
    try:
        chars = [c for c in text if not c.isspace()]
        n = len(chars)
        if n == 0:
            return False
        step = rect.height / n
        fs   = max(3.0, min(rect.width, step, 48.0))
        tw   = fitz.TextWriter(page.rect)
        x    = rect.x0
        y    = rect.y0 + fs
        for c in chars:
            if y > page.rect.y1:
                break
            tw.append(pos=(x, y), text=c, font=font, fontsize=fs)
            y += step
        tw.write_text(page, render_mode=3)
        return True
    except Exception as e:
        logger.debug(f"vertical line overlay failed: {e}")
        return False


def _overlay_invisible_text(page, rect, text: str, font) -> bool:
    """
    Block-level fallback: lay text into rect, wrapping to width and scaling
    font so the lines roughly fill the box height. render_mode=3 = invisible.
    Used when per-line boxes are unavailable.
    """
    try:
        width  = max(rect.width, 1.0)
        height = rect.height

        nominal = 12.0
        nlines  = max(1, len(_wrap_text_fitz(text, font, nominal, width)))
        line_factor = 1.2
        target_fs   = height / (nlines * line_factor) if height > 0 else nominal
        fs = max(3.0, min(target_fs, 48.0))

        lines = _wrap_text_fitz(text, font, fs, width)

        tw = fitz.TextWriter(page.rect)
        y  = rect.y0 + fs
        for line in lines:
            if not line:
                y += fs * line_factor
                continue
            if y > page.rect.y1:
                break
            tw.append(pos=(rect.x0, y), text=line, font=font, fontsize=fs)
            y += fs * line_factor

        tw.write_text(page, render_mode=3)
        return True
    except Exception as e:
        logger.debug(f"block overlay failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_CTRL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _esc(text: str) -> str:
    text = _CTRL_CHAR_RE.sub('', text)
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def _write_minimal_pdf(output_path: Path, title: str, message: str) -> None:
    """Write a bare-minimum valid PDF using only PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 200), title[:100], fontsize=16)
    page.insert_text((72, 240), message[:500], fontsize=10)
    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()
