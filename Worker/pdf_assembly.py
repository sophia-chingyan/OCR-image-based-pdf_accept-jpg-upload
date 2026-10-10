"""
PDF Assembly
============
Two output modes:

Clean PDF (方案 A) — assemble_clean_pdf()
   Re-renders OCR text into a clean PDF using ReportLab (PyMuPDF fallback).
   For each source page, emit (1) a full-page image of the original scan at
   300 DPI, followed by (2) a text-only page of that page's OCR text laid
   out like the original: same page size, columns/rows, line breaks and
   positions, with vertical CJK set as vertical text. The printed lines are
   found in the page image (scan_layout.py) and the OCR text is poured into
   them (page_layout.py) — the OCR model's own boxes aren't trusted. A page
   whose text doesn't fit its scanned lines gets a reflowed text page.

Searchable PDF (方案 B) — assemble_searchable_pdf()
   Keeps the original scanned pages pixel-for-pixel, and overlays an
   INVISIBLE OCR text layer using PyMuPDF render_mode=3 so the text is
   selectable, copyable, and searchable without altering the appearance.
   The text is placed on the printed lines found in the page image, the
   same layout as the clean PDF's text pages, so selection lands on the
   printed characters; pages where that layout is rejected fall back to the
   OCR model's per-line / per-block boxes.

Font selection (see fonts.py):
Every renderer uses an embedded font *chain* and picks, per character, the
first font that has the glyph, so text in any script renders and stays
copyable/searchable. The chain's head follows the dominant language:
- Traditional Chinese → AR PL UMing TW (ReportLab) / china-t (PyMuPDF)
- Simplified Chinese  → AR PL UMing CN / china-s
- Japanese            → WenQuanYi Zen Hei / japan
- Korean              → WenQuanYi Zen Hei / korea
- anything else       → Noto Serif / china-t
followed by the Noto fonts for every other script.
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

from fonts import FitzFontChain, ReportLabFontChain
from page_layout import join_visual_lines as _join_visual_lines, layout_page
from scan_layout import detect_lines
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


def _fitz_font_chain(language: str) -> FitzFontChain:
    return FitzFontChain(_get_fitz_font_name(language), language)


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


_LAYOUT_DPI = 150   # scan resolution for detecting the printed lines


def _scan_of(src_doc, page) -> tuple:
    """
    (ScanLayout or None, points per scan pixel) for a structured page: its
    printed lines detected in the source page image.
    """
    scale = 72.0 / _LAYOUT_DPI
    if src_doc is None or not (0 <= page.page_number < src_doc.page_count):
        return None, scale
    try:
        import numpy as np
        pix = src_doc[page.page_number].get_pixmap(dpi=_LAYOUT_DPI, colorspace=fitz.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
        return detect_lines(gray, page.direction == "vertical"), scale
    except Exception as e:
        logger.warning(f"Page {page.page_number}: scan line detection failed ({e})")
        return None, scale


# ─────────────────────────────────────────────────────────────────────────────
# 方案 A: Clean PDF — reflowed text with proper typography
# ─────────────────────────────────────────────────────────────────────────────

def assemble_clean_pdf(
    structure: DocumentStructure,
    output_path: Path,
    source_pdf_path: Optional[Path] = None,
    dpi: int = 400,
) -> None:
    """
    Re-render OCR text into a cleanly typeset PDF.

    For each source page the output contains two consecutive pages, both
    the size of the original page:
      1. The original scan rasterised at 300 DPI (full-frame image).
      2. A text-only page with the OCR text placed as on the original
         (reflowed when the page's OCR boxes are unusable).

    `dpi` is the rasterisation DPI the OCR boxes are measured at.

    Tries ReportLab first; falls back to PyMuPDF on any failure.
    """
    logger.info(f"Assembling clean PDF: {output_path}")
    try:
        _assemble_clean_pdf_reportlab(structure, output_path, source_pdf_path, dpi)
        logger.info(f"Clean PDF written (ReportLab): {output_path} "
                     f"({output_path.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"ReportLab build failed ({e}), falling back to PyMuPDF renderer")
        try:
            _assemble_clean_pdf_pymupdf(structure, output_path, source_pdf_path, dpi)
            logger.info(f"Clean PDF written (PyMuPDF fallback): {output_path} "
                         f"({output_path.stat().st_size/1024:.1f} KB)")
        except Exception as e2:
            logger.error(f"PyMuPDF fallback also failed ({e2}), writing minimal PDF")
            _write_minimal_pdf(output_path, structure.title or "Untitled",
                               f"PDF assembly error: {e}")


def _page_size_pt(page: StructuredPage, dpi: int, src_doc) -> tuple:
    """Size in points of the original page (falls back to A4)."""
    if page.width_px > 0 and page.height_px > 0:
        return page.width_px * 72.0 / dpi, page.height_px * 72.0 / dpi
    if src_doc is not None and 0 <= page.page_number < src_doc.page_count:
        r = src_doc[page.page_number].rect
        return r.width, r.height
    return 595.28, 841.89


def _assemble_clean_pdf_reportlab(
    structure: DocumentStructure,
    output_path: Path,
    source_pdf_path: Optional[Path],
    dpi: int = 400,
) -> None:
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab import rl_config
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Frame, Paragraph

    # Store the scan JPEGs as binary streams: ASCII85 (ReportLab's default)
    # is slow in pure Python and makes the file 25% bigger.
    rl_config.useA85 = 0

    # Embedded TrueType fonts with per-character fallback. Raises when no
    # font files exist, which sends assemble_clean_pdf to the PyMuPDF path.
    chain = ReportLabFontChain(structure.dominant_language)
    body_font = chain.primary
    logger.info(f"Clean PDF using font chain of {len(chain.specs)} font(s)")

    def _mk(text: str) -> str:
        return chain.markup(text, _esc)

    # Styles for the reflowed fallback layout (pages without usable boxes).
    styles = getSampleStyleSheet()

    def _style(name, parent_name="Normal", **kw):
        parent = styles.get(parent_name, styles["Normal"])
        return ParagraphStyle(name, parent=parent, fontName=body_font, **kw)

    s_h1    = _style("H1","Heading1",fontSize=16, leading=22, spaceBefore=14, spaceAfter=8)
    s_h2    = _style("H2","Heading2",fontSize=14, leading=19, spaceBefore=10, spaceAfter=6)
    s_h3    = _style("H3","Heading3",fontSize=12, leading=17, spaceBefore=8,  spaceAfter=4)
    s_body  = _style("B", fontSize=11, leading=18, firstLineIndent=22,
                     spaceBefore=2, spaceAfter=2, alignment=TA_JUSTIFY)
    s_fn    = _style("FN",fontSize=9,  leading=13, textColor="#555555")
    s_pn    = _style("PN",fontSize=9,  leading=12, textColor="#888888", alignment=TA_CENTER)
    s_cap   = _style("C", fontSize=10, leading=14, textColor="#666666", alignment=TA_CENTER)
    s_li    = _style("LI",fontSize=11, leading=18, leftIndent=20, bulletIndent=10)
    hs = {1: s_h1, 2: s_h2, 3: s_h3}

    def _reflow_flowables(items) -> list:
        out: list = []
        for el in items:
            t = el.text
            safe = _mk(t)
            try:
                if el.element_type == "list-item":
                    safe = _mk(f"\u2022 {t}")
                if el.href:
                    safe = f'<a href="{_esc(el.href)}" color="blue">{safe}</a>'
                if el.element_type == "heading":
                    out.append(Paragraph(safe, hs.get(min(el.level, 3), s_h3)))
                elif el.element_type == "list-item":
                    out.append(Paragraph(safe, s_li))
                elif el.element_type == "footnote":
                    out.append(Paragraph(safe, s_fn))
                elif el.element_type == "page-number":
                    out.append(Paragraph(safe, s_pn))
                elif el.element_type == "caption":
                    out.append(Paragraph(safe, s_cap))
                else:
                    out.append(Paragraph(safe, s_body))
            except Exception as e:
                logger.warning(f"Skipping element: {e} — text: {t[:50]!r}")
        return out

    def _draw_reflow(c, flowables: list, w: float, h: float) -> None:
        """Typeset flowables into page-sized frames, adding pages as needed."""
        margin = min(w, h) * 0.08

        def _frame():
            return Frame(margin, margin, w - 2 * margin, h - 2 * margin,
                         leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

        frame, fresh = _frame(), True
        while flowables:
            f0 = flowables[0]
            if frame.add(f0, c):
                flowables.pop(0)
                fresh = False
                continue
            # Doesn't fit the rest of this page: split it across pages.
            parts = frame.split(f0, c)
            if len(parts) > 1 and frame.add(parts[0], c):
                flowables[0:1] = parts[1:]
            elif fresh:                           # can't fit even an empty page
                logger.warning("Reflow: dropping an element too large for a page")
                flowables.pop(0)
                continue
            c.showPage()
            c.setPageSize((w, h))
            frame, fresh = _frame(), True

    def _draw_glyphs(c, glyphs, h: float) -> None:
        for g in glyphs:
            y = h - g.y                           # top-left → ReportLab origin
            if g.rotate:
                c.saveState()
                c.translate(g.x, y)
                c.rotate(-g.rotate)               # clockwise
                chain.draw(c, g.text, 0, 0, g.size)
                c.restoreState()
            else:
                chain.draw(c, g.text, g.x, y, g.size)

    measure = chain.text_width

    # Open source PDF for scan page rasterisation (optional)
    src_doc: Optional[fitz.Document] = None
    if source_pdf_path is not None:
        try:
            src_doc = fitz.open(str(source_pdf_path))
        except Exception as e:
            logger.warning(f"Could not open source PDF for scan embedding: {e}")

    try:
        c = Canvas(str(output_path))
        c.setTitle(structure.title or "Untitled")
        c.setAuthor(structure.author or "")
        started = False

        def _new_page(w: float, h: float) -> None:
            nonlocal started
            if started:
                c.showPage()
            c.setPageSize((w, h))
            started = True

        # ── Title page ───────────────────────────────────────────────────────
        if structure.title:
            w, h = 595.28, 841.89
            _new_page(w, h)
            for text, size, y in ((structure.title, 18, h * 0.3),
                                  (structure.author, 12, h * 0.3 + 30)):
                if text:
                    chain.draw(c, text, (w - measure(text, size)) / 2, h - y, size)

        page_items = _prepare_clean_text(structure.pages)
        for page, items in zip(structure.pages, page_items):
            pno = page.page_number
            w, h = _page_size_pt(page, dpi, src_doc)

            # ── 1. Original scan page ────────────────────────────────────────
            if src_doc is not None and 0 <= pno < src_doc.page_count:
                scan_jpeg = _rasterise_source_page_jpeg(src_doc, pno, _SCAN_DPI)
                if scan_jpeg is not None:
                    _new_page(w, h)
                    c.drawImage(ImageReader(io.BytesIO(scan_jpeg)), 0, 0, w, h)

            # ── 2. OCR text page, laid out like the original ─────────────────
            if not items:
                continue
            _new_page(w, h)
            scan, scale = _scan_of(src_doc, page)
            glyphs = layout_page(page, scan, scale, measure, chain.ink)
            if glyphs:
                _draw_glyphs(c, glyphs, h)
            else:
                _draw_reflow(c, _reflow_flowables(items), w, h)

        if not started:
            _new_page(595.28, 841.89)
            msg = "[ No text content could be extracted from this PDF ]"
            chain.draw(c, msg, 72, 841.89 - 300, 12)
        c.save()

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
    dpi: int = 400,
) -> None:
    """
    PyMuPDF fallback for clean PDF assembly — same page pairs as the
    ReportLab renderer: title page → for each source page: scan page +
    text page laid out like the original (reflowed when it can't be).
    """
    doc = fitz.open()
    font = _fitz_font_chain(structure.dominant_language)
    measure = font.text_length

    title  = structure.title or "Untitled"
    author = structure.author or ""

    # ── Title page ───────────────────────────────────────────────────────────
    title_page = doc.new_page(width=595, height=842)
    try:
        tw = fitz.TextWriter(title_page.rect)
        font.append(tw, (72, 200), title[:100], 20)
        if author:
            font.append(tw, (72, 240), author[:100], 14)
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
            w, h = _page_size_pt(struct_page, dpi, src_doc)

            # ── 1. Original scan page ────────────────────────────────────────
            if src_doc is not None and 0 <= pno < src_doc.page_count:
                scan_jpeg = _rasterise_source_page_jpeg(src_doc, pno, _SCAN_DPI)
                if scan_jpeg is not None:
                    scan_page = doc.new_page(width=w, height=h)
                    try:
                        scan_page.insert_image(scan_page.rect, stream=scan_jpeg)
                        has_any_content = True
                    except Exception as e:
                        logger.warning(f"PyMuPDF: could not insert scan image for page {pno}: {e}")

            # ── 2. OCR text page, laid out like the original ─────────────────
            if not text_elements:
                continue
            text_page = doc.new_page(width=w, height=h)
            has_any_content = True
            scan, scale = _scan_of(src_doc, struct_page)
            glyphs = layout_page(struct_page, scan, scale, measure, font.ink)
            if glyphs:
                _draw_glyphs_fitz(text_page, glyphs, font)
            else:
                _reflow_fitz(doc, text_page, text_elements, font, w, h)

        if not has_any_content:
            fallback_page = doc[0]
            try:
                tw = fitz.TextWriter(fallback_page.rect)
                font.append(tw, (72, 300),
                            "[ No text content could be extracted ]", 12)
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


def _draw_glyphs_fitz(page, glyphs, font: FitzFontChain, render_mode: int = 0) -> None:
    """
    Draw page_layout Glyphs; rotated runs each get their own morph.
    render_mode 3 draws them invisible (searchable PDF text layer).
    """
    # Written in order (upright text is flushed before each rotated run) so
    # the text extracts in reading order.
    upright = fitz.TextWriter(page.rect)
    pending = False
    for g in glyphs:
        if g.rotate:
            if pending:
                upright.write_text(page, render_mode=render_mode)
                upright, pending = fitz.TextWriter(page.rect), False
            tw = fitz.TextWriter(page.rect)
            font.append(tw, (g.x, g.y), g.text, g.size)
            tw.write_text(page, morph=(fitz.Point(g.x, g.y), fitz.Matrix(-g.rotate)),
                          render_mode=render_mode)
        else:
            font.append(upright, (g.x, g.y), g.text, g.size)
            pending = True
    if pending:
        upright.write_text(page, render_mode=render_mode)


def _reflow_fitz(doc, text_page, text_elements, font: FitzFontChain,
                 w: float, h: float) -> None:
    """Reflowed fallback layout (pages whose OCR boxes are unusable)."""
    margin = min(w, h) * 0.08
    max_width = w - 2 * margin
    y_cursor = margin + 11
    for el in text_elements:
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
        for line in _wrap_text_fitz(el.text, font, fs, max_width):
            if y_cursor > h - margin:
                text_page = doc.new_page(width=w, height=h)
                y_cursor = margin + fs
            try:
                tw = fitz.TextWriter(text_page.rect)
                font.append(tw, (margin, y_cursor), line, fs)
                tw.write_text(text_page)
            except Exception:
                try:
                    text_page.insert_text((margin, y_cursor), line[:200], fontsize=fs)
                except Exception:
                    pass
            y_cursor += fs * 1.5
        y_cursor += 4


def _wrap_text_fitz(text: str, font, fontsize: float, max_width: float) -> List[str]:
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            lines.append("")
            continue
        current = ""
        width = 0.0
        for char in raw_line:
            try:
                cw = font.text_length(char, fontsize=fontsize)
            except Exception:
                cw = fontsize * 0.6
            if width + cw > max_width and current:
                lines.append(current)
                current, width = char, cw
            else:
                current += char
                width += cw
        if current:
            lines.append(current)
    return lines if lines else [""]


# ─────────────────────────────────────────────────────────────────────────────
# Clean-PDF text preparation: join each block's visual lines
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CleanItem:
    """One typeset element on a clean-PDF text page."""
    element_type: str
    text: str
    level: int = 1
    href: Optional[str] = None


def _prepare_clean_text(pages: List[StructuredPage]) -> List[List[_CleanItem]]:
    """
    Build the per-page text items for the clean PDF. Each source page's OCR
    text stays on its own text page (right after that page's scan), in the
    order the OCR returned it — the same text the searchable PDF overlays on
    that page; nothing is moved between pages.
    """
    result: List[List[_CleanItem]] = []
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
        result.append(items)
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

    font = _fitz_font_chain(structure.dominant_language)

    px_to_pt  = 72.0 / float(dpi if dpi else 400)
    page_map  = {p.page_number: p for p in structure.pages}
    overlaid  = 0

    for pno in range(doc.page_count):
        sp = page_map.get(pno)
        if sp is None:
            continue
        page      = doc[pno]
        page_rect = page.rect

        # Preferred: text placed on the printed lines found in the scan.
        scan, scale = _scan_of(doc, sp)
        glyphs = layout_page(sp, scan, scale, font.text_length, font.ink)
        if glyphs:
            _draw_glyphs_fitz(page, glyphs, font, render_mode=3)
            overlaid += len(glyphs)
            continue

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
        font.append(tw, (rect.x0, baseline), text, fs)
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
            tw.append(pos=(x, y), text=c, font=font.font_for(c), fontsize=fs)
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
            font.append(tw, (rect.x0, y), line, fs)
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
