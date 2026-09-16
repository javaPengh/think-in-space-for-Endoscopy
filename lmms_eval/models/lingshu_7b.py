from lmms_eval.api.registry import register_model

from .qwen3vl import Qwen3VL


@register_model("lingshu_7b")
class Lingshu7B(Qwen3VL):
    MODEL_CLASS_NAME = "Qwen2_5_VLForConditionalGeneration"
    DEFAULT_PRETRAINED = "lingshu-medical-mllm/Lingshu-7B"
    MODEL_DISPLAY_NAME = "Lingshu-7B"
    TOKEN_USAGE_MODEL_NAME = "lingshu_7b"
    TOKEN_USAGE_FILENAME = "lingshu_7b_token_usage.jsonl"
