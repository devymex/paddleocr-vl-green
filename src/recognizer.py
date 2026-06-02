from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from .model import (
    PaddleOCRVLConfig,
    PaddleOCRVLForConditionalGeneration,
    PaddleOCRVLImageProcessor,
    PaddleOCRVLProcessor,
)

from .config import VL_MAX_NEW_TOKENS, VL_MAX_PIXELS, VL_MIN_PIXELS


class VLRecognizer:
    def __init__(self, model_path: str, device: Optional[str] = None, dtype=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        self.dtype = dtype
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
