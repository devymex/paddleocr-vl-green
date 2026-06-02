from __future__ import annotations

import argparse
import html as html_lib
import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoTokenizer

from model import (
    PaddleOCRVLConfig,
    PaddleOCRVLForConditionalGeneration,
    PaddleOCRVLImageProcessor,
    PaddleOCRVLProcessor,
    convert_otsl_to_html,
)

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Layout detection (ONNX)                                                    #
# --------------------------------------------------------------------------- #


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
    # row clustering
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


# --------------------------------------------------------------------------- #
#  Block cropping & post-processing                                           #
# --------------------------------------------------------------------------- #


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


def _prompt_for_block(label: str) -> str:
    if label == "table":
        return "Table Recognition:"
    if label == "chart":
        return "Chart Recognition:"
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    if label == "seal":
        return "Seal Recognition:"
    return "OCR:"


_LATEX_PAREN_RE = re.compile(r"\\[\(\)\[\]]")


def _normalize_formula(s: str, label: str) -> str:
    """Apply the same `\\(...\\)` → `$...$` normalisation as PaddleX."""
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


# --------------------------------------------------------------------------- #
#  VL recognition (HF transformers)                                           #
# --------------------------------------------------------------------------- #


class VLRecognizer:
    def __init__(self, model_path: str, device: Optional[str] = None, dtype=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        self.dtype = dtype
        # Load classes from the vendored local package so behaviour does not
        # depend on which paddleocr_vl support transformers has built-in.
        config = PaddleOCRVLConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_processor = PaddleOCRVLImageProcessor.from_pretrained(model_path)
        local_path = Path(model_path)
        chat_template_path = local_path / "chat_template.jinja"
        chat_template = (
            chat_template_path.read_text(encoding="utf-8")
            if local_path.is_dir() and chat_template_path.exists() else None
        )
        self.processor = PaddleOCRVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
        ).to(device).eval()  # type: ignore[union-attr]

    @torch.inference_mode()
    def recognise(
        self,
        image_bgr: np.ndarray,
        query: str,
        min_pixels: int = VL_MIN_PIXELS,
        max_pixels: int = VL_MAX_PIXELS,
        max_new_tokens: int = VL_MAX_NEW_TOKENS,
    ) -> str:
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        }]
        prompt = self.processor.apply_chat_template(  # type: ignore[call-arg]
            messages, tokenize=False, add_generation_prompt=True
        )
        size = dict(self.processor.image_processor.size)  # type: ignore[attr-defined]
        size["shortest_edge"] = min_pixels
        size["longest_edge"] = max_pixels
        model_inputs = self.processor(
            images=[image],
            text=[prompt],
            return_tensors="pt",
            images_kwargs={"size": size},
        )
        model_inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in model_inputs.items()
        }
        out_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        in_len = model_inputs["input_ids"].shape[1]
        trimmed = out_ids[0][in_len:]
        return self.processor.batch_decode(
            [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]


# --------------------------------------------------------------------------- #
#  HTML assembly (mirrors test.py)                                            #
# --------------------------------------------------------------------------- #


TITLE_LABELS = {"doc_title", "title"}
HEADING_LABELS = {"paragraph_title", "section_title", "heading"}
TABLE_LABELS = {"table"}
FORMULA_LABELS = {"formula"}
PIC_LABELS = {"image", "figure"}
CAPTION_LABELS = {"caption", "figure_caption", "table_caption", "vision_footnote"}
CODE_LABELS = {"code"}


def block_to_html(label: str, content: str) -> str:
    label = label.lower()
    if label in TABLE_LABELS:
        return f'<div class="block table-block">\n{content}\n</div>'
    if label in FORMULA_LABELS:
        return f'<div class="block formula-block">\\[{html_lib.escape(content)}\\]</div>'
    if label in PIC_LABELS:
        return f'<div class="block image-block"><p>[图片: {html_lib.escape(content)}]</p></div>'
    if label in TITLE_LABELS:
        return f'<h1 class="block doc-title">{html_lib.escape(content)}</h1>'
    if label in HEADING_LABELS:
        return f'<h2 class="block paragraph-title">{html_lib.escape(content)}</h2>'
    if label in CAPTION_LABELS:
        return f'<p class="block caption">{html_lib.escape(content)}</p>'
    if label in CODE_LABELS:
        return f'<pre class="block code-block"><code>{html_lib.escape(content)}</code></pre>'
    return f'<p class="block text-block">{html_lib.escape(content)}</p>'


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['\\\\(','\\\\)'], ['$','$']], displayMath: [['\\\\[','\\\\]'], ['$$','$$']] }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  body {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
          max-width: 960px; margin: 40px auto; padding: 0 20px; line-height: 1.8; color: #222; }}
  h1.doc-title {{ text-align: center; font-size: 1.6em; margin-bottom: 1em; }}
  h2.paragraph-title {{ font-size: 1.2em; margin-top: 1.5em; border-left: 4px solid #4a90d9; padding-left: 8px; }}
  .text-block {{ margin: 0.6em 0; }}
  .table-block {{ overflow-x: auto; margin: 1em 0; }}
  .table-block table {{ border-collapse: collapse; width: 100%; word-wrap: break-word; }}
  .table-block td, .table-block th {{ border: 1px solid #999; padding: 6px 10px; }}
  .formula-block {{ margin: 1em 0; text-align: center; font-size: 1.05em; }}
  .caption {{ color: #666; font-size: 0.9em; text-align: center; }}
  .image-block {{ text-align: center; color: #888; }}
  .code-block {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }}
  hr.page-sep {{ border: none; border-top: 2px dashed #ccc; margin: 2em 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


# --------------------------------------------------------------------------- #
#  Main pipeline                                                              #
# --------------------------------------------------------------------------- #


def get_image_files(path: str, recursive: bool = False) -> List[str]:
    """Get all image files from a directory or return a single file.

    Supported formats: jpg, jpeg, png (case-insensitive).
    Files are sorted alphabetically. If `recursive` is True and path is a
    directory, the directory tree will be scanned recursively. If `path` is a
    file, the `recursive` flag is ignored.
    """
    p = Path(path)
    if p.is_file():
        return [path]
    if p.is_dir():
        image_exts = {'.jpg', '.jpeg', '.png'}
        if recursive:
            files = [f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in image_exts]
            files = sorted(files, key=lambda fp: str(fp))
            return [str(f) for f in files]
        images = []
        for file in sorted(p.iterdir()):
            if file.is_file() and file.suffix.lower() in image_exts:
                images.append(str(file))
        return images
    raise ValueError(f"Path does not exist: {path}")


def format_output_path(base_out: str, index: int, total: int) -> str:
    """Format output path with index suffix for multiple files.

    For multiple files, adds _{idx:0xd}.html suffix where x is determined by total count.
    For single file, returns base_out unchanged (auto-appends .html if missing).
    """
    base_path = Path(base_out)

    # Ensure .html extension
    if total == 1:
        if base_path.suffix.lower() != '.html':
            return base_out + '.html'
        return base_out

    # Calculate number of digits needed
    digits = len(str(total - 1))

    # Get stem and ensure .html suffix for multiple files
    if base_path.suffix.lower() == '.html':
        stem = base_path.stem
        suffix = '.html'
    else:
        # No extension or other extension: use full name as stem and add .html
        stem = base_path.name
        suffix = '.html'

    parent = base_path.parent
    indexed_name = f"{stem}_{index:0{digits}d}{suffix}"
    return str(parent / indexed_name)


def process_image(
    image_path: str,
    layout: LayoutDetector,
    vl: VLRecognizer,
) -> List[dict]:
    """Run layout detection + VL recognition on one image. Returns list of blocks."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    boxes = layout(img_bgr)

    blocks: List[dict] = []
    for det in boxes:
        label = det["label"]
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        crop = img_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        if label in IMAGE_LABELS:
            # Image-like blocks: no VL recognition, store label name as content.
            blocks.append({"block_label": label, "block_content": label})
            continue

        prompt = _prompt_for_block(label)

        if "formula" in label and label != "formula_number":
            trimmed = crop_margin(crop)
            if trimmed.shape[0] > 2 and trimmed.shape[1] > 2:
                crop = trimmed

        text = vl.recognise(crop, prompt)

        if label == "table":
            html_str = convert_otsl_to_html(text)
            if html_str:
                text = html_str
        else:
            text = _normalize_formula(text, label)

        blocks.append({"block_label": label, "block_content": text})
    return blocks


def render_html(image_path: str, blocks: List[dict]) -> str:
    parts = []
    for b in blocks:
        label = b["block_label"]
        content = b["block_content"]
        if not content or label in DROP_LABELS:
            continue
        parts.append(block_to_html(label, content))
    body = '<div class="page" id="page-1">\n' + "\n".join(parts) + "\n</div>"
    return HTML_TEMPLATE.format(
        title=html_lib.escape(Path(image_path).name),
        body=body,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="输入图片路径（文件或目录）")
    parser.add_argument("--out", required=True, metavar="PATH", help="输出 HTML 文件路径（或路径前缀，用于多个文件）")
    parser.add_argument("--model-path", required=True,
                        help="HuggingFace repo id or local directory of the PaddleOCR-VL model")
    parser.add_argument("--layout-onnx", required=True,
                        help="Path to PP-DocLayoutV3 ONNX model")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively scan input directory for images (ignored if `images` is a file)")
    args = parser.parse_args()

    print(f"[*] Loading layout model: {args.layout_onnx}")
    layout = LayoutDetector(args.layout_onnx)
    print(f"[*] Loading VL model:     {args.model_path}")
    vl = VLRecognizer(args.model_path)

    # Get list of image files to process
    image_files = get_image_files(args.images, args.recursive)
    if not image_files:
        print("[!] No image files found.")
        return

    print(f"[*] Found {len(image_files)} image(s)")

    for idx, image_path in enumerate(image_files):
        print(f"[*] Processing [{idx+1}/{len(image_files)}]: {image_path}")
        blocks = process_image(image_path, layout, vl)
        print(f"[*] Got {len(blocks)} blocks")

        html_text = render_html(image_path, blocks)
        output_path = format_output_path(args.out, idx, len(image_files))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html_text, encoding="utf-8")
        print(f"[+] Saved: {output_path}")


if __name__ == "__main__":
    main()
