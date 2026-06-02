from __future__ import annotations

LAYOUT_INPUT_SIZE = 800
LAYOUT_SCORE_THRESHOLD = 0.5

# Class id list for PP-DocLayoutV3 (from inference.yml `label_list`).
LAYOUT_LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer", "footer_image",
    "footnote", "formula_number", "header", "header_image", "image",
    "inline_formula", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote",
]

# Labels treated as pure image content (no VL recognition).
IMAGE_LABELS = {"image", "header_image", "footer_image", "chart", "seal"}

# Labels skipped in HTML output altogether.
DROP_LABELS: set = set()

# VLM image preprocessing pixel bounds (from PaddleX defaults).
VL_MIN_PIXELS = 112896
VL_MAX_PIXELS = 1003520
VL_MAX_NEW_TOKENS = 4096
