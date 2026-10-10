"""
Scan Layout
===========
Finds the text lines of a scanned page directly in its image, so the clean
PDF can put the OCR text exactly where it is printed. OCR models return the
text reliably but their bounding boxes and reading order often are not
(Gemini, for one, tends to answer in its own 0–1000 box space and to drop
per-line boxes), so the page geometry comes from the pixels instead.

`detect_lines()` works on a grayscale rasterisation of the page:
- vertical pages: ink is projected onto the x axis to find the columns,
  each column is split at large vertical gaps (a page number, a title) and
  the pieces are read right-to-left, top-to-bottom;
- horizontal pages: the same on the transposed image (rows, top-to-bottom).
  A page with side-by-side text columns is rejected (returns None) — its
  reading order can't be recovered from projections.

The character pitch along a line comes from the autocorrelation of the ink
profile of the longest lines; with it each line gets a capacity in
characters, which is how page_layout pours the OCR text into the lines.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_INK_LEVEL = 140          # gray value below which a pixel is ink
_MIN_LINES = 1


@dataclass
class ScanLine:
    """One printed line (a column in vertical text), in image pixels."""
    x0: int
    y0: int
    x1: int
    y1: int
    vertical: bool

    @property
    def thickness(self) -> int:          # across the line
        return (self.x1 - self.x0) if self.vertical else (self.y1 - self.y0)

    @property
    def length(self) -> int:             # along the reading direction
        return (self.y1 - self.y0) if self.vertical else (self.x1 - self.x0)

    @property
    def start(self) -> int:              # where reading starts (indent)
        return self.y0 if self.vertical else self.x0


@dataclass
class ScanLayout:
    lines: List[ScanLine]                # reading order
    pitch: float                         # body character pitch, pixels
    body_thickness: float                # median line thickness, pixels
    vertical: bool

    def line_pitch(self, ln: ScanLine) -> float:
        """Pitch of one line — larger for visibly larger (heading) type."""
        if ln.thickness > self.body_thickness * 1.2:
            return self.pitch * ln.thickness / self.body_thickness
        return self.pitch

    def size_scale(self, ln: ScanLine) -> float:
        """Type size of a line relative to the body text (1.0 for body)."""
        return ln.thickness / self.body_thickness if self.is_heading(ln) else 1.0

    def capacity(self, ln: ScanLine) -> int:
        """Characters (em units) that fit the printed length of the line."""
        p = self.line_pitch(ln)
        return max(1, int(round((ln.length - ln.thickness) / p)) + 1)

    def is_heading(self, ln: ScanLine) -> bool:
        return ln.thickness > self.body_thickness * 1.2

    @property
    def text_start(self) -> int:
        return min(ln.start for ln in self.lines if not self.is_heading(ln)) \
            if any(not self.is_heading(ln) for ln in self.lines) else 0

    def is_footer(self, ln: ScanLine) -> bool:
        """
        A short mark well below where the text lines start — a page number
        or running footer under the columns of a vertical page.
        """
        if not self.vertical:
            return False
        longest = max(l.length for l in self.lines)
        return ln.start > self.text_start + longest * 0.75 and ln.length < longest * 0.2

    def is_indented(self, ln: ScanLine) -> bool:
        return ln.start > self.text_start + self.pitch * 0.5


def _runs(mask: np.ndarray) -> List[List[int]]:
    """[start, end] (inclusive) of the True runs of a 1-D mask."""
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return [[int(s), int(e)] for s, e in zip(starts, ends)]


def _bands(ink: np.ndarray) -> List[List[int]]:
    """Column bands across axis 1 (x), merged and with slivers removed."""
    # A text line puts ink on many pixels along it; a small mark (a page
    # number, a stray dot) only on a few. Ignoring the faint positions keeps
    # such marks from bridging two neighbouring lines into one band.
    prof = ink.sum(axis=0)
    busy = prof[prof > 0]
    if not len(busy):
        return []
    runs = _runs(prof > max(1.0, float(np.percentile(busy, 95)) * 0.08))
    if not runs:
        return []
    med = float(np.median([e - s + 1 for s, e in runs]))
    merged: List[List[int]] = []
    for r in runs:
        # Only tiny gaps (an offset punctuation mark) join two runs; the gap
        # between neighbouring lines — small in horizontal text — must not.
        if merged and r[0] - merged[-1][1] <= med * 0.2:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    med = float(np.median([e - s + 1 for s, e in merged]))
    # Slivers (stray marks) and very wide bands (pictures, scan borders) are
    # not text lines.
    return [b for b in merged if med * 0.5 <= b[1] - b[0] + 1 <= med * 4]


def _pitch(ink: np.ndarray, segs, thick: float) -> Optional[float]:
    """Character pitch along axis 0 from the autocorrelation of long lines."""
    # CJK cells are about square: the pitch is a little over the line
    # thickness. Take the FIRST strong autocorrelation peak, not the highest
    # one, which can be a multiple of the pitch.
    lo, hi = max(1, int(thick * 0.9)), int(thick * 1.8) + 1
    found = []
    for x0, x1, y0, y1 in sorted(segs, key=lambda s: s[2] - s[3])[:6]:
        prof = ink[y0:y1 + 1, x0:x1 + 1].sum(axis=1).astype(float)
        if len(prof) < hi * 4:
            continue
        prof -= prof.mean()
        ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
        win = ac[lo:hi]
        if not len(win) or win.max() <= 0:
            continue
        peaks = [i for i in range(1, len(win) - 1)
                 if win[i] >= win[i - 1] and win[i] >= win[i + 1]
                 and win[i] >= win.max() * 0.8]
        found.append(lo + (peaks[0] if peaks else int(np.argmax(win))))
    return float(np.median(found)) if found else None


def detect_lines(gray: np.ndarray, vertical: bool) -> Optional[ScanLayout]:
    """
    Text lines of a page image (uint8 grayscale, H×W) in reading order, or
    None when no clean line structure is found.
    """
    ink = gray < _INK_LEVEL
    if not vertical:
        ink = ink.T                       # rows become columns
    bands = _bands(ink)
    if len(bands) < _MIN_LINES:
        return None
    thick = float(np.median([b[1] - b[0] + 1 for b in bands]))

    # Split every band at large gaps along the line (≈ 2 character cells).
    segs = []                             # (x0, x1, y0, y1) in `ink` space
    multi = 0
    for x0, x1 in bands:
        rows = _runs(ink[:, x0:x1 + 1].any(axis=1))
        pieces: List[List[int]] = []
        for r in rows:
            if pieces and r[0] - pieces[-1][1] <= thick * 2.2:
                pieces[-1][1] = r[1]
            else:
                pieces.append(r)
        pieces = [p for p in pieces if p[1] - p[0] + 1 >= thick * 0.5]
        multi += len(pieces) > 1
        for p0, p1 in pieces:
            # Each piece's own extent across the line: a page number under
            # one column must not widen that column (it would read as a
            # larger, heading-size line).
            across = _runs(ink[p0:p1 + 1, x0:x1 + 1].any(axis=0))
            segs.append((x0 + across[0][0], x0 + across[-1][1], p0, p1))
    if not segs:
        return None
    if not vertical and multi > len(bands) * 0.3:
        logger.info("Scan layout: side-by-side text columns — not handled")
        return None

    pitch = _pitch(ink, segs, thick) or thick * 1.1

    # Reading order: vertical → bands right-to-left; horizontal → rows
    # top-to-bottom (the transposed x axis); then along the line.
    segs.sort(key=lambda s: ((-s[0]) if vertical else s[0], s[2]))
    lines = []
    for x0, x1, y0, y1 in segs:
        if vertical:
            lines.append(ScanLine(x0, y0, x1 + 1, y1 + 1, True))
        else:                             # back from transposed space
            lines.append(ScanLine(y0, x0, y1 + 1, x1 + 1, False))
    return ScanLayout(lines=lines, pitch=pitch, body_thickness=thick, vertical=vertical)
