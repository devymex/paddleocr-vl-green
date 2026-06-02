from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import base64
import logging

import cv2
import numpy as np
from flask import Flask, jsonify, make_response, request

from src import (
    LayoutDetector,
    VLRecognizer,
    render_html,
    run_pipeline,
)

# --------------------------------------------------------------------------- #
#  Logging                                                                    #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Flask app                                                                  #
# --------------------------------------------------------------------------- #

app = Flask(__name__)
# Allow large base64 payloads (up to ~100 MB raw JSON)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Models are loaded once at startup and shared across requests
_layout: LayoutDetector | None = None
_vl: VLRecognizer | None = None


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #


def _decode_image(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded image (with optional data-URI prefix) to BGR ndarray."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from the provided base64 data.")
    return img



# --------------------------------------------------------------------------- #
#  Routes                                                                     #
# --------------------------------------------------------------------------- #


@app.route("/health", methods=["GET"])
def health():
    """Quick liveness / readiness probe."""
    ready = _layout is not None and _vl is not None
    return jsonify({"status": "ok" if ready else "models_not_loaded"}), 200 if ready else 503


@app.route("/process", methods=["POST"])
def process():
    """Process a single image and return structured data or an HTML page.

    Request body (JSON)
    -------------------
    ``image``    — **required** base64-encoded image (JPEG / PNG).
                   A ``data:image/...;base64,`` prefix is accepted and stripped.
    ``format``   — ``"json"`` (default) or ``"html"``.

    JSON response (``format=json``)
    --------------------------------
    .. code-block:: json

        {
            "blocks": [
                {"label": "paragraph_title", "content": "Introduction"},
                {"label": "text",            "content": "Lorem ipsum ..."},
                {"label": "table",           "content": "<table>...</table>"}
            ]
        }

    HTML response (``format=html``)
    ---------------------------------
    A fully rendered ``text/html`` page with MathJax for formulae and
    embedded CSS, ready for display in a browser.
    """
    if _layout is None or _vl is None:
        return jsonify({"error": "Models are not loaded yet."}), 503

    # ---- parse request ----
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    b64_image = body.get("image")
    if not b64_image:
        return jsonify({"error": "Missing required field: 'image'."}), 400

    fmt = str(body.get("format", "json")).lower()
    if fmt not in ("json", "html"):
        return jsonify({"error": "'format' must be 'json' or 'html'."}), 400

    # ---- decode image ----
    try:
        img_bgr = _decode_image(b64_image)
    except Exception as exc:
        logger.warning("Image decode failed: %s", exc)
        return jsonify({"error": f"Image decode error: {exc}"}), 400

    # ---- run pipeline ----
    try:
        logger.info("Processing image (h=%d w=%d) format=%s", *img_bgr.shape[:2], fmt)
        blocks = run_pipeline(img_bgr, _layout, _vl)
        logger.info("Done. %d blocks detected.", len(blocks))
    except Exception as exc:
        logger.exception("Pipeline error")
        return jsonify({"error": f"Processing error: {exc}"}), 500

    # ---- build response ----
    if fmt == "html":
        html_text = render_html(blocks)
        resp = make_response(html_text)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp

    return jsonify({"blocks": blocks})


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #


def load_models(layout_onnx: str, model_path: str) -> None:
    global _layout, _vl
    logger.info("Loading layout model: %s", layout_onnx)
    _layout = LayoutDetector(layout_onnx)
    logger.info("Loading VL model:     %s", model_path)
    _vl = VLRecognizer(model_path)
    logger.info("Both models ready.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PaddleOCR-VL Flask inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument(
        "--layout-onnx",
        required=True,
        help="Path to PP-DocLayoutV3 ONNX model",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace repo id or local directory of PaddleOCR-VL",
    )
    args = parser.parse_args()

    load_models(layout_onnx=args.layout_onnx, model_path=args.model_path)
    # Use threaded=False to avoid concurrent GPU access; for multi-worker
    # production deployment use gunicorn with --workers 1.
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
