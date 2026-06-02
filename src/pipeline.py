from __future__ import annotations

from typing import List

import numpy as np

from .otsl_to_html import convert_otsl_to_html

from .config import IMAGE_LABELS
from .layout import LayoutDetector
from .postprocess import crop_margin, normalize_formula, prompt_for_block
from .recognizer import VLRecognizer


def run_pipeline(
    img_bgr: np.ndarray,
    layout: LayoutDetector,
    vl: VLRecognizer,
) -> List[dict]:
    """Run layout detection + VL recognition on a BGR image array.

    Returns a list of ``{"label": str, "content": str}`` dicts in reading order.
    """
    boxes = layout(img_bgr)
    blocks: List[dict] = []

    for det in boxes:
        label = det["label"]
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        crop = img_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        if label in IMAGE_LABELS:
            blocks.append({"label": label, "content": label})
            continue

        query = prompt_for_block(label)

        if "formula" in label and label != "formula_number":
            trimmed = crop_margin(crop)
            if trimmed.shape[0] > 2 and trimmed.shape[1] > 2:
                crop = trimmed

        text = vl.recognise(crop, query)

        if label == "table":
            html_str = convert_otsl_to_html(text)
            if html_str:
                text = html_str
        else:
            text = normalize_formula(text, label)

        blocks.append({"label": label, "content": text})

    return blocks
