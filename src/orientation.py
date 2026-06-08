"""Document orientation detection using PP-LCNet_x1_0_doc_ori ONNX model."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore


class OrientationDetector:
    """
    PP-LCNet_x1_0_doc_ori classifier using ONNX Runtime.
    Classifies document images into 4 orientations: 0°, 90°, 180°, 270°
    """

    # Model constants
    INPUT_SIZE = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    ORIENTATIONS = [0, 90, 180, 270]  # Degrees, matching class indices

    def __init__(self, model_path: str, providers: Optional[list] = None) -> None:
        """
        Initialize the orientation detector.

        Args:
            model_path: Path to the ONNX model file
            providers: ONNX Runtime execution providers (defaults to CPU/GPU auto-detection)
        """
        if ort is None:
            raise ImportError("onnxruntime is required for OrientationDetector")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model inference.

        Steps:
        1. Resize: short side to 256, keep aspect ratio
        2. Center crop: 224x224
        3. Normalize: using ImageNet mean/std
        4. Convert: BGR to RGB, and to CHW format

        Args:
            image: Input image (BGR format from cv2)

        Returns:
            Preprocessed image array (CHW, float32)
        """
        h, w = image.shape[:2]

        # Step 1: Resize short side to 256
        scale = 256.0 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Step 2: Center crop to 224x224
        h, w = image.shape[:2]
        top = (h - self.INPUT_SIZE) // 2
        left = (w - self.INPUT_SIZE) // 2
        image = image[top : top + self.INPUT_SIZE, left : left + self.INPUT_SIZE]

        # Step 3: Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 4: Normalize (scale to 0-1, then normalize with ImageNet stats)
        image_float = image_rgb.astype(np.float32) / 255.0
        image_float = (image_float - self.MEAN) / self.STD

        # Step 5: Convert HWC to CHW
        image_chw = np.transpose(image_float, (2, 0, 1))

        # Step 6: Add batch dimension
        image_batch = np.expand_dims(image_chw, axis=0)

        return image_batch.astype(np.float32)

    def predict(self, image: np.ndarray) -> dict:
        """
        Predict document orientation for an image.

        Args:
            image: Input image in BGR format (from cv2.imread)

        Returns:
            Dictionary with keys:
            - 'orientation': Predicted orientation in degrees (0, 90, 180, or 270)
            - 'score': Confidence score (0-1)
            - 'all_scores': Scores for all orientation classes
        """
        # Preprocess
        input_data = self._preprocess(image)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_data})
        scores_output = outputs[0]
        if hasattr(scores_output, "__getitem__"):
            scores = np.asarray(scores_output)[0]
        else:
            scores = np.asarray(scores_output)

        # Get prediction
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])

        result = {
            "orientation": self.ORIENTATIONS[class_id],
            "score": score,
            "all_scores": {
                o: float(s) for o, s in zip(self.ORIENTATIONS, scores)
            },
        }

        return result

    def correct_image(self, image: np.ndarray, orientation: int) -> np.ndarray:
        """
        Rotate image to correct orientation (0 degrees).

        Args:
            image: Input image in BGR format
            orientation: Predicted orientation in degrees (0, 90, 180, 270)

        Returns:
            Corrected image (rotated so it becomes 0 degrees)
        """
        if orientation == 0:
            return image
        elif orientation == 90:
            # Rotate counter-clockwise 90 degrees
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == 180:
            # Rotate 180 degrees
            return cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 270:
            # Rotate counter-clockwise 270 degrees (or clockwise 90)
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            raise ValueError(f"Invalid orientation: {orientation}")
