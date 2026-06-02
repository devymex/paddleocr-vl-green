"""Minimal OTSL-v1.0 → HTML table converter.

Ported from PaddleX's pipeline utilities, with `pydantic` replaced by
`dataclasses` so this module has no third-party dependencies.

OTSL tags:
  <fcel>  filled cell (with text following the tag)
  <ecel>  empty cell
  <lcel>  left-merged cell (span from cell to the left)
  <ucel>  up-merged cell (span from cell above)
  <xcel>  cross merge (both directions)
  <nl>    newline (end of row)
"""

from __future__ import annotations

import html
import itertools
import re
from dataclasses import dataclass
from typing import List

OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"

_TAGS = (OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL)
_TAG_GROUP = "(?:" + "|".join(_TAGS) + ")"
_OTSL_FIND = re.compile(_TAG_GROUP + ".*?(?=" + _TAG_GROUP + "|$)", flags=re.DOTALL)
_TAG_SPLIT = re.compile("(" + "|".join(_TAGS) + ")")


@dataclass
class TableCell:
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    text: str = ""
    row_span: int = 1
    col_span: int = 1


def _extract_tokens_and_text(s: str):
    tokens = _TAG_SPLIT.findall(s)
    parts = [p for p in _TAG_SPLIT.split(s) if p.strip()]
    return tokens, parts


def _pad_to_sqr(otsl: str) -> str:
    otsl = otsl.strip()
    if OTSL_NL not in otsl:
        return otsl + OTSL_NL
    lines = otsl.split(OTSL_NL)
    rows = []
    for line in lines:
        if not line:
            continue
        cells = _OTSL_FIND.findall(line)
        if not cells:
            continue
        min_len = 0
        for i, c in enumerate(cells):
            if c.startswith(OTSL_FCEL):
                min_len = i + 1
        rows.append({"cells": cells, "total": len(cells), "min": min_len})
    if not rows:
        return OTSL_NL
    global_min = max(r["min"] for r in rows)
    max_total = max(r["total"] for r in rows)
    best_w = max(global_min, max_total)
    best_cost = float("inf")
    for w in range(global_min, max(global_min, max_total) + 1):
        cost = sum(abs(r["total"] - w) for r in rows)
        if cost < best_cost:
            best_cost = cost
            best_w = w
    out_lines = []
    for r in rows:
        cs = r["cells"]
        if len(cs) > best_w:
            cs = cs[:best_w]
        else:
            cs = cs + [OTSL_ECEL] * (best_w - len(cs))
        out_lines.append("".join(cs))
    return OTSL_NL.join(out_lines) + OTSL_NL


def _parse(texts, tokens):
    split_rows = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == OTSL_NL)
        if not x
    ]
    cells: List[TableCell] = []
    if split_rows:
        max_cols = max(len(r) for r in split_rows)
        for r in split_rows:
            while len(r) < max_cols:
                r.append(OTSL_ECEL)
        # rebuild flat texts that align with the padded grid
        new_texts = []
        ti = 0
        for r in split_rows:
            for tok in r:
                new_texts.append(tok)
                if ti < len(texts) and texts[ti] == tok:
                    ti += 1
                    if ti < len(texts) and texts[ti] not in _TAGS:
                        new_texts.append(texts[ti])
                        ti += 1
            new_texts.append(OTSL_NL)
            if ti < len(texts) and texts[ti] == OTSL_NL:
                ti += 1
        texts = new_texts

    def count_right(c, r, which):
        span = 0
        ci = c
        while ci < len(split_rows[r]) and split_rows[r][ci] in which:
            ci += 1
            span += 1
        return span

    def count_down(c, r, which):
        span = 0
        ri = r
        while ri < len(split_rows) and c < len(split_rows[ri]) and split_rows[ri][c] in which:
            ri += 1
            span += 1
        return span

    r_idx = 0
    c_idx = 0
    i = 0
    while i < len(texts):
        text = texts[i]
        if text in (OTSL_FCEL, OTSL_ECEL):
            cell_text = ""
            right_offset = 1
            if text == OTSL_FCEL and i + 1 < len(texts) and texts[i + 1] not in _TAGS:
                cell_text = texts[i + 1]
                right_offset = 2
            next_right = texts[i + right_offset] if i + right_offset < len(texts) else ""
            next_bottom = ""
            if r_idx + 1 < len(split_rows) and c_idx < len(split_rows[r_idx + 1]):
                next_bottom = split_rows[r_idx + 1][c_idx]
            col_span, row_span = 1, 1
            if next_right in (OTSL_LCEL, OTSL_XCEL):
                col_span += count_right(c_idx + 1, r_idx, (OTSL_LCEL, OTSL_XCEL))
            if next_bottom in (OTSL_UCEL, OTSL_XCEL):
                row_span += count_down(c_idx, r_idx + 1, (OTSL_UCEL, OTSL_XCEL))
            cells.append(TableCell(
                start_row=r_idx, end_row=r_idx + row_span,
                start_col=c_idx, end_col=c_idx + col_span,
                text=cell_text.strip(), row_span=row_span, col_span=col_span,
            ))
        if text in (OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL):
            c_idx += 1
        elif text == OTSL_NL:
            r_idx += 1
            c_idx = 0
        i += 1
    return cells, split_rows


def _export_html(cells: List[TableCell], n_rows: int, n_cols: int) -> str:
    if not cells or n_rows == 0 or n_cols == 0:
        return ""
    # default grid: empty TableCell for every position
    grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    for c in cells:
        for i in range(min(c.start_row, n_rows), min(c.end_row, n_rows)):
            for j in range(min(c.start_col, n_cols), min(c.end_col, n_cols)):
                grid[i][j] = c  # type: ignore[index]
    body = ""
    for i in range(n_rows):
        body += "<tr>"
        for j in range(n_cols):
            c = grid[i][j]
            if c is None:
                body += "<td></td>"
                continue
            if c.start_row != i or c.start_col != j:
                continue
            content = html.escape(c.text)
            tag = "td"
            attrs = ""
            if c.row_span > 1:
                attrs += f' rowspan="{c.row_span}"'
            if c.col_span > 1:
                attrs += f' colspan="{c.col_span}"'
            body += f"<{tag}{attrs}>{content}</{tag}>"
        body += "</tr>"
    return f"<table>{body}</table>"


def convert_otsl_to_html(otsl: str) -> str:
    """Convert an OTSL-v1.0 string to an HTML <table>...</table> string.

    Returns an empty string if the input cannot be parsed.
    """
    if not otsl:
        return ""
    otsl = _pad_to_sqr(otsl)
    tokens, mixed = _extract_tokens_and_text(otsl)
    cells, rows = _parse(mixed, tokens)
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=0)
    return _export_html(cells, n_rows, n_cols)
