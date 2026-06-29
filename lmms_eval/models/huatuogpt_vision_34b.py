import transformers

from lmms_eval.api.registry import register_model

from .medgemma_27b import MedGemma27B


@register_model("huatuogpt_vision_34b")
class HuatuoGPTVision34B(MedGemma27B):
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/FreedomIntelligence/HuatuoGPT-Vision-34B-hf"
    MODEL_DISPLAY_NAME = "HuatuoGPT-Vision-34B-hf"
    TOKEN_USAGE_MODEL_NAME = "huatuogpt_vision_34b"
    TOKEN_USAGE_FILENAME = "huatuogpt_vision_34b_token_usage.jsonl"
    DOWNLOAD_HINT = "Download FreedomIntelligence/HuatuoGPT-Vision-34B-hf from ModelScope to this local path, or pass a local pretrained path."

    def _resolve_model_class(self):
        for class_name in ("AutoModelForVision2Seq", "AutoModelForCausalLM"):
            model_class = getattr(transformers, class_name, None)
            if model_class is not None:
                return model_class
        raise ImportError(f"{self.MODEL_DISPLAY_NAME} requires Transformers with AutoModelForVision2Seq " "or AutoModelForCausalLM. Use the HuatuoGPT-Vision-compatible Transformers environment.")
