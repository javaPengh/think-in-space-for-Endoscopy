from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_5_27b import Qwen3_5_27B


@register_model("qwen3_5_35b_a3b")
class Qwen3_5_35B_A3B(Qwen3_5_27B):
    MODEL_CLASS_NAME = "Qwen3_5MoeForConditionalGeneration"
    EXPECTED_MODEL_TYPE = "qwen3_5_moe"
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B"
    MODEL_DISPLAY_NAME = "Qwen3.5-35B-A3B"
    TOKEN_USAGE_MODEL_NAME = "qwen3_5_35b_a3b"
    TOKEN_USAGE_FILENAME = "qwen3_5_35b_a3b_token_usage.jsonl"
