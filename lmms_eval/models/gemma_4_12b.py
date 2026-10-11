from lmms_eval.api.registry import register_model
from lmms_eval.models.gemma_4_31b import Gemma4_31B


@register_model("gemma_4_12b")
class Gemma4_12B(Gemma4_31B):
    MODEL_CLASS_NAME = "Gemma4UnifiedForConditionalGeneration"
    EXPECTED_MODEL_TYPE = "gemma4_unified"
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/google/gemma-4-12B-it"
    MODEL_DISPLAY_NAME = "Gemma-4-12B-it"
    TOKEN_USAGE_MODEL_NAME = "gemma_4_12b"
    TOKEN_USAGE_FILENAME = "gemma_4_12b_token_usage.jsonl"
