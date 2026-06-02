from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from .config import LAYOUT_INPUT_SIZE, LAYOUT_LABELS, LAYOUT_SCORE_THRESHOLD


class LayoutDetector:
    def __init__(self, onnx_path: str, providers=None):
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

    def __call__(self, img_bgr: np.ndarray, score_thr: float = LAYOUT_SCORE_THRESHOLD):
        """Run layout detection. Returns list of dicts {label, score, bbox=(x1,y1,x2,y2)}."""
        h0, w0 = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        outputs = self.sess.run(None, {
            "image": x,
            "im_shape": np.array([[LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE]], dtype=np.float32),
            "scale_factor": np.array([[LAYOUT_INPUT_SIZE / h0,
                                       LAYOUT_INPUT_SIZE / w0]], dtype=np.float32),
        })
        dets = outputs[0]  # (N, 7): [cls_id, score, x1, y1, x2, y2, mask_idx]
        keep = dets[dets[:, 1] >= score_thr]  # type: ignore[index]
        boxes = []
        for d in keep:
            cls, score, x1, y1, x2, y2 = d[0], d[1], d[2], d[3], d[4], d[5]
            x1 = max(0, min(w0 - 1, float(x1)))
            y1 = max(0, min(h0 - 1, float(y1)))
            x2 = max(0, min(w0, float(x2)))
            y2 = max(0, min(h0, float(y2)))
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            boxes.append({
                "label": LAYOUT_LABELS[int(cls)],
                "score": float(score),
                "bbox": (x1, y1, x2, y2),
            })
        boxes = _filter_overlap(boxes)
        boxes = _sort_reading_order(boxes)
        return boxes


def _iou_small(a, b) -> float:
    """Overlap area / area of smaller box."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    small = min(area_a, area_b)
    return inter / small if small > 0 else 0.0


def _filter_overlap(boxes):
    """Drop heavily-overlapping smaller boxes (approximation of PaddleX's filter)."""
    boxes = [b for b in boxes if b["label"] != "reference"]
    if not boxes:
        return boxes
    n = len(boxes)
    drop = set()
    areas = [
        (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
        for b in boxes
    ]
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            ratio = _iou_small(boxes[i]["bbox"], boxes[j]["bbox"])
            li, lj = boxes[i]["label"], boxes[j]["label"]
            if li == "inline_formula" or lj == "inline_formula":
                if ratio > 0.5:
                    if li == "inline_formula":
                        drop.add(i); break
                    if lj == "inline_formula":
                        drop.add(j); continue
            if ratio > 0.7:
                labels = {li, lj}
                if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                    if "table" not in labels or labels <= {"table", "image", "seal", "chart"}:
                        continue
                if areas[i] >= areas[j]:
                    drop.add(j)
                else:
                    drop.add(i); break
    return [b for i, b in enumerate(boxes) if i not in drop]


def _sort_reading_order(boxes):
    """Simple top-to-bottom, left-to-right ordering. Boxes whose vertical
    centers are within ~30 px are grouped on the same row."""
    if not boxes:
        return boxes
    items = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        items.append((y1, x1, b))
    items.sort(key=lambda t: (t[0], t[1]))
    rows = []
    for y1, x1, b in items:
        placed = False
        for row in rows:
            ry = row[0]
            if abs(y1 - ry) < 30:
                row[1].append(b)
                placed = True
                break
        if not placed:
            rows.append([y1, [b]])
    out = []
    for _, row in rows:
        row.sort(key=lambda b: b["bbox"][0])
        out.extend(row)
    return out
