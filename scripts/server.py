#!/usr/bin/env python3
"""PaddleOCR-VL Flask server — multi-worker, load-balanced edition.

Supports a flexible ``--device`` spec identical to ``server.py`` so you can
run multiple workers across one or more GPUs (or CPU):

    python scripts/server_simple.py --device cpu
    python scripts/server_simple.py --device cuda:0
    python scripts/server_simple.py --device cuda:0,0,1,1   # 4 workers

Each worker runs in its own process (spawned, not forked) and owns an
independent copy of both models.  Flask threads dispatch tasks to the shared
queue; the first idle worker picks up the task, providing natural FIFO +
load-balanced concurrency.
"""
from __future__ import annotations

import argparse
import base64
import logging
import multiprocessing as mp
from typing import List, Optional

import cv2
import numpy as np
import torch

from flask import Flask, jsonify, make_response, request
from src import render_html, LayoutDetector, VLRecognizer, run_pipeline as _run_pipeline

# --------------------------------------------------------------------------- #
#  Logging                                                                    #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Flask app & shared state (main process only)                              #
# --------------------------------------------------------------------------- #

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max request

# Set by _start_workers(); accessed by Flask route handlers in the main process.
_task_queue: Optional[mp.queues.Queue] = None  # type: ignore[type-arg]
_workers_ready: bool = False


# --------------------------------------------------------------------------- #
#  Device spec parsing                                                        #
# --------------------------------------------------------------------------- #

def parse_device(device_str: Optional[str]) -> List[Optional[int]]:
    """Parse a device spec string into a list of GPU IDs (``None`` = CPU).

    If device_str is "auto", auto-detects: returns one GPU worker per available GPU,
    or [None] if no GPUs are available.

    Examples::

        "auto"          -> [0, 1, ...]     # auto-detect: one worker per GPU (or [None] if no GPU)
        "cpu"           -> [None]           # single CPU worker
        "cuda"          -> [0]              # one worker on GPU 0
        "cuda:0"        -> [0]              # one worker on GPU 0
        "cuda:0,0,1,1"  -> [0, 0, 1, 1]    # 4 workers: 2 on GPU 0, 2 on GPU 1

    GPU indices are logical indices and respect ``CUDA_VISIBLE_DEVICES``.
    """
    if device_str is None or (isinstance(device_str, str) and device_str.strip().lower() == "auto"):
        # Auto-detect: try to import torch and check CUDA availability
        try:
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                return list(range(n_gpus))
        except ImportError:
            pass
        return [None]  # Fall back to CPU if no CUDA or torch not available

    s = device_str.strip().lower()
    if s == "cpu":
        return [None]
    if s == "cuda":
        return [0]
    if s.startswith("cuda:"):
        parts_str = s.split(":", 1)[1]
        parts = [x.strip() for x in parts_str.split(",") if x.strip()]
        if not parts:
            raise ValueError(f"No GPU indices found in device spec {device_str!r}.")
        try:
            return [int(x) for x in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid GPU index in device spec {device_str!r}.") from exc
    raise ValueError(
        f"Invalid device spec {device_str!r}. "
        "Use 'cpu', 'cuda' or 'cuda:<id>[,<id>,...]', e.g. 'cuda:0,0,1,1'."
    )


# --------------------------------------------------------------------------- #
#  Worker process                                                              #
# --------------------------------------------------------------------------- #

def _worker_fn(
    worker_id: int,
    gpu_id: Optional[int],
    layout_onnx: str,
    model_path: str,
    task_queue: mp.queues.Queue,  # type: ignore[type-arg]
    ready_queue: mp.queues.Queue,  # type: ignore[type-arg]
) -> None:
    """Runs in a child process: loads models once, then handles tasks from *task_queue*.

    Each task is a tuple ``(img_bgr, fmt, child_conn)`` where *child_conn* is the
    write end of a :func:`multiprocessing.Pipe`.  Results (or errors) are sent back
    through *child_conn* as a dict ``{"ok": bool, ...}``.

    A ``None`` item in *task_queue* is a poison pill that causes the worker to exit.
    """
    tag = f"worker-{worker_id}/{'cpu' if gpu_id is None else f'cuda:{gpu_id}'}"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [{tag}] %(message)s",
        force=True,
    )
    log = logging.getLogger(__name__)

    # Build device strings before any GPU initialisation
    if gpu_id is not None:
        torch_device = f"cuda:{gpu_id}"
        ort_providers: list = [
            ("CUDAExecutionProvider", {"device_id": gpu_id}),
            "CPUExecutionProvider",
        ]
    else:
        torch_device = "cpu"
        ort_providers = ["CPUExecutionProvider"]

    # Load models
    try:
        log.info("Loading LayoutDetector …")
        layout = LayoutDetector(layout_onnx, providers=ort_providers)

        log.info("Loading VLRecognizer on %s …", torch_device)
        vl = VLRecognizer(model_path, device=torch_device)

        log.info("Models loaded. Worker ready.")
        ready_queue.put(("ok", worker_id))
    except Exception as exc:
        log.exception("Failed to load models")
        ready_queue.put(("error", worker_id, str(exc)))
        return

    # Inference loop — blocks on the shared queue; each worker picks up a task
    # only when it is idle, providing natural FIFO + load-balanced dispatch.
    while True:
        item = task_queue.get()
        if item is None:          # poison pill → clean shutdown
            log.info("Received shutdown signal. Exiting.")
            break
        img_bgr, fmt, child_conn = item
        try:
            log.info("Processing image h=%d w=%d fmt=%s", *img_bgr.shape[:2], fmt)
            blocks = _run_pipeline(img_bgr, layout, vl)
            log.info("Done — %d blocks.", len(blocks))
            child_conn.send({"ok": True, "blocks": blocks})
        except Exception as exc:
            log.exception("Pipeline error")
            child_conn.send({"ok": False, "error": str(exc)})
        finally:
            child_conn.close()


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


def _dispatch(img_bgr: np.ndarray, fmt: str) -> dict:
    """Submit an inference task to the worker pool and block until the result arrives.

    Thread-safe: each call creates its own :func:`multiprocessing.Pipe` so
    concurrent Flask threads do not mix up results.
    """
    parent_conn, child_conn = mp.Pipe(duplex=False)
    _task_queue.put((img_bgr, fmt, child_conn))  # type: ignore[union-attr]
    result: dict = parent_conn.recv()
    parent_conn.close()
    return result


# --------------------------------------------------------------------------- #
#  Routes                                                                     #
# --------------------------------------------------------------------------- #

@app.route("/health", methods=["GET"])
def health():
    """Liveness / readiness probe.

    Returns ``200 ok`` once all workers have finished loading their models,
    ``503 models_not_loaded`` otherwise.
    """
    ready = _workers_ready
    return jsonify({"status": "ok" if ready else "models_not_loaded"}), 200 if ready else 503


@app.route("/process", methods=["POST"])
def process():
    """Process a single image and return structured data or an HTML page.

    Request body (JSON)
    -------------------
    ``image``  — **required** base64-encoded image (JPEG / PNG).
                 A ``data:image/...;base64,`` prefix is accepted and stripped.
    ``format`` — ``"json"`` (default) or ``"html"``.

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
    --------------------------------
    A fully rendered ``text/html`` page with MathJax for formulae and
    embedded CSS, ready for display in a browser.
    """
    if not _workers_ready:
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
        logger.info("Image decoded: %dx%d", img_bgr.shape[1], img_bgr.shape[0])
    except Exception as exc:
        logger.warning("Image decode failed: %s", exc)
        return jsonify({"error": f"Image decode error: {exc}"}), 400

    # ---- dispatch to worker pool ----
    try:
        result = _dispatch(img_bgr, fmt)
    except Exception as exc:
        logger.exception("Dispatch error")
        return jsonify({"error": f"Processing error: {exc}"}), 500

    if not result.get("ok"):
        return jsonify({"error": result.get("error", "Unknown pipeline error")}), 500

    blocks = result["blocks"]

    # ---- build response ----
    if fmt == "html":
        html_text = render_html(blocks)
        resp = make_response(html_text)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp

    return jsonify({"blocks": blocks})


# --------------------------------------------------------------------------- #
#  Worker pool management                                                     #
# --------------------------------------------------------------------------- #

def _start_workers(
    gpu_ids: List[Optional[int]],
    layout_onnx: str,
    model_path: str,
    ctx,  # multiprocessing context (spawn / fork / forkserver)
) -> tuple:
    """Spawn worker processes and block until every worker reports ready.

    Returns ``(task_queue, processes)``.
    """
    global _workers_ready

    task_queue = ctx.Queue()
    ready_queue = ctx.Queue()
    n = len(gpu_ids)
    processes: List[mp.Process] = []

    for i, gid in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker_fn,
            args=(i, gid, layout_onnx, model_path, task_queue, ready_queue),
            daemon=True,
            name=f"worker-{i}",
        )
        p.start()
        label = "cpu" if gid is None else f"cuda:{gid}"
        logger.info("Launched %s (worker-%d, pid=%d)", label, i, p.pid)
        processes.append(p)

    ok_count = 0
    for _ in range(n):
        msg = ready_queue.get()
        if msg[0] == "ok":
            ok_count += 1
            logger.info("Worker %d ready (%d/%d).", msg[1], ok_count, n)
        else:
            logger.error("Worker %d failed to start: %s", msg[1], msg[2])

    if ok_count < n:
        raise RuntimeError(
            f"{n - ok_count}/{n} worker(s) failed to load models. Aborting."
        )

    _workers_ready = True
    logger.info("All %d worker(s) ready.", n)
    return task_queue, processes


# --------------------------------------------------------------------------- #
#  Entry point                                                                #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL multi-worker HTTP inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5000, help="Bind port")
    parser.add_argument(
        "--layout-onnx",
        default="/gemini/data-1/model/paddle/paddleocr-layout/pp_doclayoutv3.onnx",
        metavar="PATH",
        help="Path to PP-DocLayoutV3 ONNX model",
    )
    parser.add_argument(
        "--model-path",
        default="/gemini/data-1/model/paddle/paddleocr-vl-1.6",
        metavar="PATH",
        help="Local directory or HuggingFace repo id of PaddleOCR-VL",
    )
    parser.add_argument(
        "--device",
        default="auto",
        metavar="SPEC",
        help=(
            "Device spec controlling the worker pool (default: 'auto'). "
            "'auto' auto-detects GPUs and starts one worker per GPU, "
            "or falls back to CPU if no GPUs are available. "
            "'cpu' starts a single CPU worker. "
            "'cuda' or 'cuda:0' starts one worker on GPU 0. "
            "'cuda:<id>[,<id>,...]' starts one worker per listed GPU id "
            "(the same id may appear multiple times to run several workers on one GPU), "
            "e.g. 'cuda:0,0,1,1' starts 4 workers — 2 on GPU 0, 2 on GPU 1. "
            "IDs are logical indices and respect the CUDA_VISIBLE_DEVICES env var."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    args = parser.parse_args()

    try:
        gpu_ids = parse_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))

    labels = ["cpu" if g is None else f"cuda:{g}" for g in gpu_ids]
    logger.info(
        "Device spec '%s' → %d worker(s): %s",
        args.device,
        len(gpu_ids),
        labels,
    )

    # 'spawn' gives each worker a clean Python interpreter — no CUDA-context
    # inheritance issues that can arise with the default 'fork' on Linux.
    ctx = mp.get_context("spawn")

    global _task_queue
    _task_queue, _processes = _start_workers(
        gpu_ids=gpu_ids,
        layout_onnx=args.layout_onnx,
        model_path=args.model_path,
        ctx=ctx,
    )

    logger.info("Starting Flask server on %s:%d …", args.host, args.port)
    # threaded=True: Flask handles concurrent HTTP requests in separate threads;
    # each thread blocks on _dispatch() waiting for an idle worker, so the
    # effective parallelism is bounded by the number of inference workers.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
