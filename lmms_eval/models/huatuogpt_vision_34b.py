import transformers

from lmms_eval.api.registry import register_model

from .frame_sequence_vision import FrameSequenceVisionModel


@register_model("huatuogpt_vision_34b")
class HuatuoGPTVision34B(FrameSequenceVisionModel):
    DEFAULT_PRETRAINED = "~/.cache/modelscope/hub/models/FreedomIntelligence/HuatuoGPT-Vision-34B-hf"
    MODEL_DISPLAY_NAME = "HuatuoGPT-Vision-34B-hf"
    TOKEN_USAGE_MODEL_NAME = "huatuogpt_vision_34b"
    TOKEN_USAGE_FILENAME = "huatuogpt_vision_34b_token_usage.jsonl"
    DOWNLOAD_HINT = "Download FreedomIntelligence/HuatuoGPT-Vision-34B-hf from ModelScope to this local path, or pass a local pretrained path."
    MANUAL_CHAT_TEMPLATE = "<|user|>\n{prompt}\n<|assistant|>\n"

    def _resolve_model_class(self):
        for class_name in ("AutoModelForVision2Seq", "AutoModelForCausalLM"):
            model_class = getattr(transformers, class_name, None)
            if model_class is not None:
                return model_class
        raise ImportError(f"{self.MODEL_DISPLAY_NAME} requires Transformers with AutoModelForVision2Seq " "or AutoModelForCausalLM. Use the HuatuoGPT-Vision-compatible Transformers environment.")

    def _extract_text(self, content):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts).strip()

    def _build_prompt_text(self, content, images):
        text = self._extract_text(content)
        image_tokens = "\n".join(["<image>"] * len(images))
        prompt_body = "\n".join(part for part in [image_tokens, text] if part).strip()

        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_body}], tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.MANUAL_CHAT_TEMPLATE.format(prompt=prompt_body)

        image_token_count = prompt.count("<image>")
        if image_token_count != len(images):
            raise ValueError(f"{self.MODEL_DISPLAY_NAME} prompt contains {image_token_count} <image> tokens but received {len(images)} images.")
        return prompt

    def _build_inputs(self, content, images):
        prompt = self._build_prompt_text(content, images)
        processor_kwargs = {"text": [prompt], "padding": True, "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        inputs = self._processor(**processor_kwargs)
        return self._move_inputs_to_device(inputs)
