"""
Test suite for inference.py and server.py.

Run from the project root:
    python scripts/test.py
    python scripts/test.py --model-path /path/to/model --layout-onnx /path/to/layout.onnx
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src import LayoutDetector, VLRecognizer

# Ensure project root is on sys.path so 'src' is importable.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_SAMPLE_IMAGE = str(ROOT / "sample" / "contract.jpg")

# Set by CLI args at startup; required for integration tests.
_model_path: str | None = None
_layout_onnx: str | None = None


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _image_to_b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


# --------------------------------------------------------------------------- #
# Tests: inference helpers (no model needed)                                  #
# --------------------------------------------------------------------------- #

class TestInferenceHelpers(unittest.TestCase):
    """Unit tests for the pure-Python helpers in inference.py."""

    def setUp(self):
        # Import lazily so sys.path is already patched.
        from scripts.inference import get_image_files, format_output_path
        self.get_image_files = get_image_files
        self.format_output_path = format_output_path

    # ---- get_image_files -------------------------------------------------- #

    def test_single_file_returned_as_list(self):
        result = self.get_image_files(_SAMPLE_IMAGE)
        self.assertEqual(result, [_SAMPLE_IMAGE])

    def test_nonexistent_path_raises(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.get_image_files("/no/such/path/image.jpg")

    def test_directory_returns_images(self):
        sample_dir = str(ROOT / "sample")
        result = self.get_image_files(sample_dir)
        self.assertIsInstance(result, list)
        for f in result:
            self.assertTrue(f.lower().endswith((".jpg", ".jpeg", ".png")))

    # ---- format_output_path ----------------------------------------------- #

    def test_single_file_keeps_html_extension(self):
        path = self.format_output_path("output/result.html", 0, 1)
        self.assertTrue(path.endswith(".html"))

    def test_single_file_adds_html_extension_if_missing(self):
        path = self.format_output_path("output/result", 0, 1)
        self.assertTrue(path.endswith(".html"))

    def test_multi_file_adds_index_suffix(self):
        path0 = self.format_output_path("output/result.html", 0, 3)
        path1 = self.format_output_path("output/result.html", 1, 3)
        self.assertNotEqual(path0, path1)
        self.assertIn("result_0", path0)
        self.assertIn("result_1", path1)


# --------------------------------------------------------------------------- #
# Tests: server helpers (no model needed)                                     #
# --------------------------------------------------------------------------- #

class TestServerHelpers(unittest.TestCase):
    """Unit tests for server.py helpers that don't require loaded models."""

    def setUp(self):
        import scripts.server as srv
        self.srv = srv
        self.client = srv.app.test_client()

    def test_decode_image_plain_b64(self):
        b64 = _image_to_b64(_SAMPLE_IMAGE)
        img = self.srv._decode_image(b64)
        self.assertEqual(img.ndim, 3)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)

    def test_decode_image_data_uri(self):
        b64 = "data:image/jpeg;base64," + _image_to_b64(_SAMPLE_IMAGE)
        img = self.srv._decode_image(b64)
        self.assertEqual(img.ndim, 3)

    def test_decode_image_invalid_raises(self):
        with self.assertRaises(Exception):
            self.srv._decode_image("not-valid-base64!!!")

    def test_health_without_models_returns_503(self):
        # Reset module-level model references so they are None.
        self.srv._layout = None
        self.srv._vl = None
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 503)
        body = json.loads(resp.data)
        self.assertEqual(body["status"], "models_not_loaded")

    def test_process_without_models_returns_503(self):
        self.srv._layout = None
        self.srv._vl = None
        payload = {"image": _image_to_b64(_SAMPLE_IMAGE)}
        resp = self.client.post(
            "/process",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 503)

    def test_process_missing_image_field(self):
        # Temporarily inject dummy models so the endpoint goes past the model check.
        self.srv._layout = object()
        self.srv._vl = object()
        resp = self.client.post(
            "/process",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = json.loads(resp.data)
        self.assertIn("image", body["error"])

    def test_process_invalid_format(self):
        self.srv._layout = object()
        self.srv._vl = object()
        payload = {"image": _image_to_b64(_SAMPLE_IMAGE), "format": "xml"}
        resp = self.client.post(
            "/process",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_non_json_body_returns_400(self):
        self.srv._layout = object()
        self.srv._vl = object()
        resp = self.client.post(
            "/process",
            data="not json at all",
            content_type="text/plain",
        )
        self.assertEqual(resp.status_code, 400)


# --------------------------------------------------------------------------- #
# Integration tests: full pipeline (requires models)                         #
# --------------------------------------------------------------------------- #

class TestInferencePipeline(unittest.TestCase):
    """End-to-end test: load models, run inference on sample/contract.jpg."""

    layout: LayoutDetector
    vl: VLRecognizer

    @classmethod
    def setUpClass(cls):
        from scripts.inference import process_image
        from src import LayoutDetector, VLRecognizer
        cls.process_image = staticmethod(process_image)
        assert _layout_onnx is not None, "--layout-onnx is required"
        assert _model_path is not None, "--model-path is required"
        print(f"\n[setup] Loading layout model: {_layout_onnx}")
        cls.layout = LayoutDetector(_layout_onnx)
        print(f"[setup] Loading VL model:     {_model_path}")
        cls.vl = VLRecognizer(_model_path)

    def test_process_image_returns_blocks(self):
        blocks = self.process_image(_SAMPLE_IMAGE, self.layout, self.vl)
        self.assertIsInstance(blocks, list)
        self.assertGreater(len(blocks), 0, "Expected at least one block from contract.jpg")

    def test_each_block_has_label_and_content(self):
        blocks = self.process_image(_SAMPLE_IMAGE, self.layout, self.vl)
        for block in blocks:
            self.assertIn("label", block, f"Block missing 'label': {block}")
            self.assertIn("content", block, f"Block missing 'content': {block}")

    def test_html_output_written_to_disk(self):
        from src import render_html
        blocks = self.process_image(_SAMPLE_IMAGE, self.layout, self.vl)
        html = render_html(blocks, title="contract.jpg")
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w", encoding="utf-8", delete=False
        ) as fh:
            fh.write(html)
            tmp_path = fh.name
        try:
            self.assertGreater(os.path.getsize(tmp_path), 0)
            content = Path(tmp_path).read_text(encoding="utf-8")
            self.assertIn("<html", content.lower())
        finally:
            os.unlink(tmp_path)

    def test_output_matches_reference(self):
        """Compare rendered HTML against output/contract.html (regression check)."""
        import difflib
        from src import render_html

        ref_path = ROOT / "output" / "contract.html"
        self.assertTrue(ref_path.exists(), f"Reference file not found: {ref_path}")
        reference = ref_path.read_text(encoding="utf-8")

        blocks = self.process_image(_SAMPLE_IMAGE, self.layout, self.vl)
        generated = render_html(blocks, title="contract.jpg")

        if generated != reference:
            diff = "\n".join(difflib.unified_diff(
                reference.splitlines(),
                generated.splitlines(),
                fromfile="output/contract.html",
                tofile="generated",
                lineterm="",
            ))
            self.fail(f"Output does not match reference:\n\n{diff}")


class TestServerPipeline(unittest.TestCase):
    """End-to-end test: server /health and /process with real models."""

    @classmethod
    def setUpClass(cls):
        import scripts.server as srv
        assert _layout_onnx is not None, "--layout-onnx is required"
        assert _model_path is not None, "--model-path is required"
        srv.load_models(layout_onnx=_layout_onnx, model_path=_model_path)
        cls.srv = srv
        cls.client = srv.app.test_client()

    def test_health_with_models_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertEqual(body["status"], "ok")

    def test_process_json_format(self):
        payload = {
            "image": _image_to_b64(_SAMPLE_IMAGE),
            "format": "json",
        }
        resp = self.client.post(
            "/process",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertIn("blocks", body)
        self.assertGreater(len(body["blocks"]), 0)

    def test_process_html_format(self):
        payload = {
            "image": _image_to_b64(_SAMPLE_IMAGE),
            "format": "html",
        }
        resp = self.client.post(
            "/process",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.content_type)
        self.assertIn(b"<html", resp.data.lower())


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", required=True,
                        help="Path or HuggingFace repo ID for the VL model")
    parser.add_argument("--layout-onnx", required=True,
                        help="Path to pp_doclayoutv3.onnx")
    parser.add_argument("--no-integration", action="store_true",
                        help="Skip tests that load the full VL model")
    known, remaining = parser.parse_known_args()

    _model_path = known.model_path
    _layout_onnx = known.layout_onnx

    if known.no_integration:
        # Remove integration test classes so unittest won't run them.
        del TestInferencePipeline
        del TestServerPipeline
    else:
        # Make the paths available to the integration test classes.
        pass

    # Pass remaining argv (e.g. -v) to unittest.
    sys.argv = [sys.argv[0]] + remaining
    unittest.main(verbosity=2)
