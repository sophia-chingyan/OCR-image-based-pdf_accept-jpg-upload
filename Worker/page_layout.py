"""
Page Layout
===========
Places a page's OCR text where it is printed on the original page, so the
clean PDF's text page mirrors the scan: same columns/rows, same line
breaks, same indents, headings and section titles in the same spots.

The page geometry comes from the scan itself (scan_layout.detect_lines),
not from the OCR model's bounding boxes, which are often missing or in the
wrong coordinate space. The OCR text is then poured into the detected
lines:
- heading elements (typed so, or short with no sentence punctuation) go to
  the visibly larger (heading) lines, in order — this also fixes models
  that list a mid-page section title first — and page numbers under
  vertical columns go to those short footer lines;
- each body element (paragraph) gets the run of lines whose capacity best
  matches its length, preferring to end where the next line is indented,
  and its characters are shared out in proportion to line capacity, so
  any estimation drift is reset at every paragraph;
- a line never starts with closing punctuation (禁則).

Vertical text (CJK, top-to-bottom columns, right-to-left) follows book
typesetting:
- small punctuation (，。、．) moves to the upper right of its cell;
- brackets, dashes, ellipses and the long-vowel mark are turned 90°;
- runs of Latin letters/digits (e.g. "(enlightenment)") are set sideways,
  except numbers/letters of at most two characters (a chapter number "2"),
  which stand upright across the column (縱中橫).
The Unicode text is never altered, so copy/search still give normal text.

`layout_page()` returns `Glyphs` draw calls in PDF points (top-left
origin), or None when the scan has no usable line structure or the text
doesn't fit it plausibly — the caller then reflows the text instead.
"""

from __future__ import annotations
import logging
import unicodedata
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from scan_layout import ScanLayout, ScanLine
from structure_analysis import StructuredPage

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

# A page's text must fill its detected lines to within this ratio.
_FIT_RANGE = (0.65, 1.5)
# Ink thickness of a line relative to its font size.
_CJK_INK = 0.92


@dataclass
class Glyphs:
    """One draw call: `text` with its baseline origin at (x, y), in points."""
    text: str
    x: float
    y: float
    size: float
    rotate: int = 0      # 0, or 90 = turned clockwise (reads top-to-bottom)



# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF
            or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF
            or 0x20000 <= o <= 0x2FA1F)


def join_visual_lines(text: str) -> str:
    """
    Join the visual lines of one OCR block into continuous text. A space is
    kept only between two non-CJK words, so CJK text gets no stray spaces.
    The text is NFC-normalised: OCR models sometimes return CJK
    compatibility ideographs (e.g. U+F967 for 不), which many fonts lack
    and which searches for the ordinary character don't find.
    """
    out = ""
    text = unicodedata.normalize("NFC", text)
    for part in (ln.strip() for ln in text.split("\n")):
        if not part:
            continue
        if out and not is_cjk(out[-1]) and not is_cjk(part[0]):
            out += " "
        out += part
    return out


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



# ─────────────────────────────────────────────────────────────────────────────
# Horizontal text
# ─────────────────────────────────────────────────────────────────────────────

def _row_items(text: str, measure: Measure) -> List[Tuple[str, str, float]]:
    """
    Tokens of horizontal text as ('w', token, width at size 1): whole words
    (with their trailing space) for space-separated scripts, single
    characters for CJK, so rows only break where the script allows it.
    """
    items: List[Tuple[str, str, float]] = []
    word = ""
    for ch in text:
        if is_cjk(ch):
            if word:
                items.append(("w", word, measure(word, 1.0)))
                word = ""
            items.append(("w", ch, measure(ch, 1.0)))
        else:
            word += ch
            if ch.isspace():
                items.append(("w", word, measure(word, 1.0)))
                word = ""
    if word:
        items.append(("w", word, measure(word, 1.0)))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Pouring text into the scanned lines
# ─────────────────────────────────────────────────────────────────────────────

def _ranges(units: List[float], lines: List[ScanLine], caps: List[float],
            scan: ScanLayout) -> Optional[List[Tuple[int, int]]]:
    """
    Contiguous line ranges, one per element, whose capacity best matches
    each element's length. Ending a range where the next line is indented
    (a new paragraph) or the line is short (a paragraph's last line) is
    preferred. The last element takes the remaining lines.
    """
    if len(units) > len(lines):
        return None
    full = max(caps)
    out, start = [], 0
    for i, u in enumerate(units):
        left = len(units) - i - 1                 # elements still to place
        if left == 0:
            out.append((start, len(lines) - 1))
            break
        best, acc = None, 0.0
        for e in range(start, len(lines) - left):
            acc += caps[e]
            score = abs(acc - u)
            # Paragraph-end hints only break near-ties; they must never pull
            # a range far short of (or past) the element's length.
            if score <= full * 0.2:
                nxt = lines[e + 1]
                if scan.is_indented(nxt) or scan.is_heading(nxt):
                    score -= full * 0.08
                if caps[e] < full * 0.85:
                    score -= full * 0.04
            if best is None or score < best[0]:
                best = (score, e)
            if acc > u * 1.5 + full:
                break
        e = best[1]
        out.append((start, e))
        start = e + 1
    return out


def _distribute(items, caps: List[float]) -> List[list]:
    """Share items over lines in proportion to line capacity, with 禁則."""
    total_units = sum(it[2] for it in items) or 1.0
    total_cap = sum(caps) or 1.0
    bounds, acc = [], 0.0
    for c in caps:
        acc += c
        bounds.append(acc * total_units / total_cap)
    per = [[] for _ in caps]
    pos, li = 0.0, 0
    for it in items:
        mid = pos + it[2] / 2
        while li < len(caps) - 1 and mid > bounds[li]:
            li += 1
        per[li].append(it)
        pos += it[2]
    # 禁則: closing punctuation hangs at the end of the previous line.
    for i in range(1, len(per)):
        while per[i] and per[i][0][1][:1] in _NO_LINE_START and per[i - 1]:
            per[i - 1].append(per[i].pop(0))
    return per


_SENTENCE_PUNCT = set("，。、；：！？,.;:!?")


def _heading_like(el, text: str) -> bool:
    """
    A heading element — typed so by the OCR, or (models often don't type
    them) a short text without sentence punctuation. Page numbers aren't.
    """
    if el.element_type == "page-number" or text.strip().isdigit():
        return False
    if el.element_type == "heading":
        return True
    return len(text) <= 20 and not (set(text) & _SENTENCE_PUNCT)


def _place_vertical_line(ln: ScanLine, items, scan: ScanLayout, scale: float,
                         measure: Measure, ink: Optional[Ink], out: List[Glyphs]) -> None:
    if not items:
        return
    step = scan.line_pitch(ln) * scale
    size = min(ln.thickness * scale / _CJK_INK, step)
    units = sum(it[2] for it in items)
    length = ln.length * scale
    if units > 1 and (units - 1) * step + size > length * 1.08:
        step = max((length - size) / (units - 1), size * 0.6)
    xc = (ln.x0 + ln.x1) / 2 * scale
    _place_column(items, xc, ln.y0 * scale, size, step, measure, ink, out)


def _place_horizontal_line(ln: ScanLine, items, scale: float, measure: Measure,
                           out: List[Glyphs]) -> None:
    text = "".join(it[1] for it in items).strip()
    if not text:
        return
    width = measure(text, 1.0) or 1.0
    height_size = ln.thickness * scale / _CJK_INK
    # The printed row is exactly as long as its text, so size from width —
    # never wider than the row, and not taller than its ink suggests.
    size = min(ln.length * scale / width, height_size * 1.3)
    out.append(Glyphs(text, ln.x0 * scale, ln.y1 * scale - 0.12 * size, size))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def layout_page(page: StructuredPage, scan: Optional[ScanLayout], scale: float,
                measure: Measure, ink: Optional[Ink] = None) -> Optional[List[Glyphs]]:
    """
    Draw calls placing every OCR element of `page` in the lines of its scan
    (`scale` = PDF points per scan pixel), or None when the text can't be
    placed plausibly (caller should reflow).
    """
    try:
        return _layout_page(page, scan, scale, measure, ink)
    except Exception as e:  # never let one odd page break the whole PDF
        logger.warning(f"Page {page.page_number}: layout failed ({e}) — "
                       f"using reflowed layout for this page")
        return None


def _layout_page(page, scan, scale, measure, ink):
    elements = [(el, join_visual_lines(el.text or "")) for el in page.elements]
    elements = [(el, t) for el, t in elements if t]
    if not elements:
        return []
    if scan is None or not scan.lines:
        return None
    vertical = scan.vertical
    tokens = _column_items if vertical else _row_items

    lines = scan.lines
    if vertical:
        caps = [float(scan.capacity(ln)) for ln in lines]
    else:
        # Horizontal (often proportional) text: capacity in em units from a
        # page-wide em size, calibrated so the text fills the rows.
        # Heading lines are set larger, so they hold fewer em per length.
        total_len = sum(ln.length / scan.size_scale(ln) for ln in lines)
        total_units = sum(sum(it[2] for it in tokens(t, measure)) for _, t in elements)
        em = total_len / total_units if total_units else 1.0
        # The em this implies must look like the printed type: rows whose
        # ink is far thinner or thicker than that aren't this text's rows.
        if not 0.5 <= em / scan.body_thickness <= 1.8:
            logger.info(f"Page {page.page_number}: text doesn't fit the scanned rows "
                        f"(em/row {em / scan.body_thickness:.2f}) — using reflowed layout")
            return None
        caps = [ln.length / (scan.size_scale(ln) * em) for ln in lines]

    units_all = [sum(it[2] for it in tokens(t, measure)) for _, t in elements]
    ratio = sum(units_all) / (sum(caps) or 1.0)
    if not (_FIT_RANGE[0] <= ratio <= _FIT_RANGE[1]):
        logger.info(f"Page {page.page_number}: text doesn't fit the scanned lines "
                    f"(ratio {ratio:.2f}) — using reflowed layout")
        return None

    groups: List[Tuple[List[int], List[int]]] = []   # (element idx, line idx)
    body_l, body_e = list(range(len(lines))), list(range(len(elements)))

    def pair(line_ids, elem_ids):
        """Give each element its own line, in order, when the counts agree."""
        nonlocal body_l, body_e
        if line_ids and len(line_ids) == len(elem_ids):
            groups.extend(([e], [l]) for e, l in zip(elem_ids, line_ids))
            body_l = [i for i in body_l if i not in set(line_ids)]
            body_e = [i for i in body_e if i not in set(elem_ids)]

    # Page numbers under vertical columns, then headings, go to their own
    # lines; everything else is body text.
    page_nums = [i for i, (el, t) in enumerate(elements)
                 if el.element_type == "page-number" or t.strip().isdigit()]
    pair([i for i, ln in enumerate(lines) if scan.is_footer(ln)], page_nums)
    # Page numbers with no line of their own never join the body text (they
    # would shift every paragraph); they go to the bottom centre instead.
    loose_nums = [i for i in page_nums if i in body_e]
    body_e = [i for i in body_e if i not in set(loose_nums)]
    pair([i for i in body_l if scan.is_heading(lines[i])],
         [i for i in body_e if _heading_like(*elements[i])])

    if body_e and not body_l:
        return None

    def fill(body_groups):
        """Lines' items for the fixed groups plus `body_groups`, or None on overflow."""
        per_line: dict = {}
        for eis, lis in groups + body_groups:
            items = [it for ei in eis for it in tokens(elements[ei][1], measure)]
            for li, its in zip(lis, _distribute(items, [caps[i] for i in lis])):
                # A line that would have to squeeze in far more than it
                # holds means this split doesn't match the scan.
                if sum(it[2] for it in its) > caps[li] * 1.12 + 1:
                    return None
                per_line[li] = its
        return per_line

    per_line = None
    if body_e:
        # Preferred: each paragraph in its own run of lines.
        ranges = _ranges([units_all[i] for i in body_e], [lines[i] for i in body_l],
                         [caps[i] for i in body_l], scan)
        if ranges is not None:
            per_line = fill([([ei], body_l[a:b + 1]) for ei, (a, b) in zip(body_e, ranges)])
        if per_line is None:
            # Paragraph lengths don't line up with the scan (e.g. the OCR
            # split or ordered them differently): spread all body text over
            # all body lines instead — still in the page's own lines.
            per_line = fill([(body_e, body_l)])
    else:
        per_line = fill([])
    if per_line is None:
        logger.info(f"Page {page.page_number}: a line overflows its scanned "
                    f"length — using reflowed layout")
        return None

    # Emit line by line in the scan's reading order, so the text extracts
    # (copy, search) in page order.
    out: List[Glyphs] = []
    if loose_nums:
        size = scan.pitch * scale * 0.8
        bottom = max(ln.y1 for ln in lines) * scale + size * 2
        mid = (min(ln.x0 for ln in lines) + max(ln.x1 for ln in lines)) / 2 * scale
        for i in loose_nums:
            t = elements[i][1]
            out.append(Glyphs(t, mid - measure(t, size) / 2, bottom, size))
    for li in sorted(per_line):
        ln = lines[li]
        if vertical and scan.is_footer(ln):
            # A page number under the columns is set horizontally, centred
            # on its mark and as tall as it.
            t = "".join(it[1] for it in per_line[li])
            size = min((ln.y1 - ln.y0) * scale / 0.72, scan.pitch * scale)
            xc = (ln.x0 + ln.x1) / 2 * scale
            out.append(Glyphs(t, xc - measure(t, size) / 2, ln.y1 * scale, size))
        elif vertical:
            _place_vertical_line(lines[li], per_line[li], scan, scale, measure, ink, out)
        else:
            _place_horizontal_line(lines[li], per_line[li], scale, measure, out)
    return out
