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
    enable_layout: bool = True,
) -> List[dict]:
    """Run layout detection + VL recognition on a BGR image array.

    Args:
        img_bgr: Input image in BGR format.
        layout: LayoutDetector instance for region detection.
        vl: VLRecognizer instance for content recognition.
        enable_layout: If True, perform layout detection and process each region separately.
                       If False, recognize the entire image as a single text block.

    Returns:
        A list of ``{"label": str, "content": str}`` dicts in reading order.
        If ``enable_layout=False``, returns a single block with label "text".
    """
    blocks: List[dict] = []

    if not enable_layout:
        # Recognize the entire image as a single text block
        query = prompt_for_block("text")
        text = vl.recognise(img_bgr, query)
        text = normalize_formula(text, "text")
        blocks.append({"label": "text", "content": text})
        return blocks

    # Layout-enabled pipeline: detect regions and recognize each
    boxes = layout(img_bgr)

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
