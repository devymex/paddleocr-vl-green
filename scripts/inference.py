from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import List

import cv2

from src import (
    LayoutDetector,
    VLRecognizer,
    render_html,
    run_pipeline,
)


# --------------------------------------------------------------------------- #
#  File I/O helpers                                                           #
# --------------------------------------------------------------------------- #


def get_image_files(path: str, recursive: bool = False) -> List[str]:
    """Get all image files from a directory or return a single file.

    Supported formats: jpg, jpeg, png (case-insensitive).
    Files are sorted alphabetically. If ``recursive`` is True and path is a
    directory, the directory tree will be scanned recursively.
    """
    p = Path(path)
    if p.is_file():
        return [path]
    if p.is_dir():
        image_exts = {'.jpg', '.jpeg', '.png'}
        if recursive:
            return sorted(
                str(f) for f in p.rglob('*')
                if f.is_file() and f.suffix.lower() in image_exts
            )
        return [
            str(f) for f in sorted(p.iterdir())
            if f.is_file() and f.suffix.lower() in image_exts
        ]
    raise ValueError(f"Path does not exist: {path}")


def format_output_path(base_out: str, index: int, total: int) -> str:
    """Format output path with index suffix for multiple files."""
    base_path = Path(base_out)
    if total == 1:
        return base_out if base_path.suffix.lower() == '.html' else base_out + '.html'
    digits = len(str(total - 1))
    stem = base_path.stem if base_path.suffix.lower() == '.html' else base_path.name
    return str(base_path.parent / f"{stem}_{index:0{digits}d}.html")


def process_image(
    image_path: str,
    layout: LayoutDetector,
    vl: VLRecognizer,
) -> List[dict]:
    """Read an image from disk, run layout + VL recognition, return blocks."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return run_pipeline(img_bgr, layout, vl)


# --------------------------------------------------------------------------- #
#  CLI entry point                                                            #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="输入图片路径（文件或目录）")
    parser.add_argument("--out", required=True, metavar="PATH",
                        help="输出 HTML 文件路径（或路径前缀，用于多个文件）")
    parser.add_argument("--model-path", required=True,
                        help="HuggingFace repo id or local directory of the PaddleOCR-VL model")
    parser.add_argument("--layout-onnx", required=True,
                        help="Path to PP-DocLayoutV3 ONNX model")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively scan input directory for images")
    args = parser.parse_args()

    print(f"[*] Loading layout model: {args.layout_onnx}")
    layout = LayoutDetector(args.layout_onnx)
    print(f"[*] Loading VL model:     {args.model_path}")
    vl = VLRecognizer(args.model_path)

    image_files = get_image_files(args.images, args.recursive)
    if not image_files:
        print("[!] No image files found.")
        return

    print(f"[*] Found {len(image_files)} image(s)")

    for idx, image_path in enumerate(image_files):
        print(f"[*] Processing [{idx+1}/{len(image_files)}]: {image_path}")
        blocks = process_image(image_path, layout, vl)
        print(f"[*] Got {len(blocks)} blocks")

        html_text = render_html(blocks, title=Path(image_path).name)
        output_path = format_output_path(args.out, idx, len(image_files))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html_text, encoding="utf-8")
        print(f"[+] Saved: {output_path}")


if __name__ == "__main__":
    main()
