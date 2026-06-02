from __future__ import annotations

import cv2
import numpy as np


def crop_margin(img: np.ndarray) -> np.ndarray:
    """Trim white margins around a (formula) crop. Mirrors PaddleX's implementation."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    mn, mx = int(gray.min()), int(gray.max())
    if mn == mx:
        return img
    lut = np.zeros(256, dtype=np.uint8)
    for v in range(mn, mx + 1):
        lut[v] = int((v - mn) / (mx - mn) * 255)
    data = cv2.LUT(gray, lut)
    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


def prompt_for_block(label: str) -> str:
    if label == "table":
        return "Table Recognition:"
    if label == "chart":
        return "Chart Recognition:"
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    if label == "seal":
        return "Seal Recognition:"
    return "OCR:"


def normalize_formula(s: str, label: str) -> str:
    """Apply the same ``\\(...\\)`` → ``$...$`` normalisation as PaddleX."""
    if not s:
        return s
    if ("\\(" in s and "\\)" in s) or ("\\[" in s and "\\]" in s):
        s = s.replace("$", "")
        s = (s.replace("\\(", " $ ")
              .replace("\\)", " $")
              .replace("\\[\\[", "\\[")
              .replace("\\]\\]", "\\]")
              .replace("\\[", " $$ ")
              .replace("\\]", " $$ "))
        if label == "formula_number":
            s = s.replace("$", "")
    return s
