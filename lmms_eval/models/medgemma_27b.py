import transformers

from lmms_eval.api.registry import register_model

from .frame_sequence_vision import FrameSequenceVisionModel


@register_model("medgemma_27b")
class MedGemma27B(FrameSequenceVisionModel):
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/google/medgemma-27b-it"
    MODEL_DISPLAY_NAME = "MedGemma-27B-IT"
    TOKEN_USAGE_MODEL_NAME = "medgemma_27b"
    TOKEN_USAGE_FILENAME = "medgemma_27b_token_usage.jsonl"
    DOWNLOAD_HINT = "Download google/medgemma-27b-it from ModelScope to this local path, or pass a local pretrained path."

    def _resolve_model_class(self):
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            raise ImportError(f"{self.MODEL_DISPLAY_NAME} requires Transformers with AutoModelForImageTextToText. " "Use the MedGemma-compatible Transformers environment.")
        return model_class

    def _build_inputs(self, content, images):
        messages = [{"role": "user", "content": content}]
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            processor_kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
            if images:
                processor_kwargs["images"] = images
            inputs = self._processor(**processor_kwargs)
        return self._move_inputs_to_device(inputs)
