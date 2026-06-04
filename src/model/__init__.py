"""Local copy of the PaddleOCR-VL-1.6 model code from the HuggingFace model dir.

The four `_paddleocr_vl` modules are vendored verbatim from the model directory
(`modeling_*.py`, `configuration_*.py`, `image_processing_*.py`,
`processing_*.py`). They depend on `transformers` (tested with 4.55.x). Custom
compat shims for newer / older transformers versions are applied here.
"""

import inspect
import warnings
import logging

import torch

# Suppress known benign transformers logger messages (>= 5.x compatibility)
logging.getLogger("transformers.modeling_rope_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)  # Keep errors, suppress lower levels

# Suppress transformers FutureWarning
warnings.filterwarnings("ignore", message=".*rope_config_validation.*", category=FutureWarning)

# ----------------------------------------------------------------------------
# Compat: newer transformers (>=4.56?) removed the "default" rope_type from
# ROPE_INIT_FUNCTIONS. Re-register it as standard RoPE (inv_freq from theta).
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
            if config is not None:
                base = getattr(config, "rope_theta", 10000.0)
                head_dim = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads
                )
                partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            else:
                base = rope_kwargs.get("base", 10000.0)
                head_dim = rope_kwargs["dim"]
                partial_rotary_factor = 1.0
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
            )
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
except Exception as e:  # noqa: BLE001
    warnings.warn(f"Could not register 'default' rope: {e}")

# ----------------------------------------------------------------------------
# Patch transformers.masking_utils.create_causal_mask: the model code calls it
# with kwarg `inputs_embeds`, but some transformers versions renamed it to
# `input_embeds`. Provide a wrapper that accepts both names.
try:
    from transformers import masking_utils as _mu
    _orig = _mu.create_causal_mask
    _sig = inspect.signature(_orig)
    _accepts_cache_position = "cache_position" in _sig.parameters
    _accepts_inputs_embeds = "inputs_embeds" in _sig.parameters
    _accepts_input_embeds = "input_embeds" in _sig.parameters
    def _create_causal_mask_compat(*args, **kwargs):
        if not _accepts_cache_position:
            kwargs.pop("cache_position", None)
        if "inputs_embeds" in kwargs and not _accepts_inputs_embeds and _accepts_input_embeds:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        elif "input_embeds" in kwargs and not _accepts_input_embeds and _accepts_inputs_embeds:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return _orig(*args, **kwargs)
    _mu.create_causal_mask = _create_causal_mask_compat
except Exception as e:  # noqa: BLE001
    warnings.warn(f"Could not patch create_causal_mask: {e}")

from . import configuration  # noqa: F401,E402
from . import image_processing  # noqa: F401,E402
from . import processing  # noqa: F401,E402
from . import modeling  # noqa: F401,E402

# ----------------------------------------------------------------------------
# Compat: newer transformers (>=4.56?) `_init_weights` invokes
# `module.compute_default_rope_parameters` on RotaryEmbedding sub-modules to
# recompute inv_freq after init. The vendored model uses an older-style
# RotaryEmbedding that stores `rope_init_fn` instead. Add the method.
def _compute_default_rope_method(self, config=None, device=None, **kwargs):
    cfg = config if config is not None else self.config
    inv_freq, attention_scaling = self.rope_init_fn(cfg, device)
    return inv_freq, attention_scaling

for _cls_name in ("RotaryEmbedding", "Ernie4_5VisionRotaryEmbedding"):
    _cls = getattr(modeling, _cls_name, None)
    if _cls is not None and not hasattr(_cls, "compute_default_rope_parameters"):
        _cls.compute_default_rope_parameters = _compute_default_rope_method

from .configuration import PaddleOCRVLConfig
from .image_processing import PaddleOCRVLImageProcessor
from .modeling import PaddleOCRVLForConditionalGeneration
from .processing import PaddleOCRVLProcessor
from ..otsl_to_html import convert_otsl_to_html

__all__ = [
    "PaddleOCRVLConfig",
    "PaddleOCRVLImageProcessor",
    "PaddleOCRVLForConditionalGeneration",
    "PaddleOCRVLProcessor",
    "convert_otsl_to_html",
]
