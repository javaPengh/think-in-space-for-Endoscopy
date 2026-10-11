from lmms_eval.api.registry import register_model
from lmms_eval.models.gemma_4_31b import Gemma4_31B


@register_model("gemma_4_26b_a4b")
class Gemma4_26B_A4B(Gemma4_31B):
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/google/gemma-4-26B-A4B-it"
    MODEL_DISPLAY_NAME = "Gemma-4-26B-A4B-it"
    TOKEN_USAGE_MODEL_NAME = "gemma_4_26b_a4b"
    TOKEN_USAGE_FILENAME = "gemma_4_26b_a4b_token_usage.jsonl"
