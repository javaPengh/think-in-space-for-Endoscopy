import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

DEFAULT_IMAGE_TOKEN = "<image>"
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_onevision_1_5")
class Llava_OneVision_1_5(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "vicuna_v1",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        max_pixels: Optional[int] = 602112,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        video_sampling_strategy: str = "uniform",
        keyframe_mapping_path: str = "data/keyframe_mapping.json",
        **kwargs,
    ) -> None:
        super().__init__()
        if "video_sampling_strategy" in kwargs:
            kwargs.pop("video_sampling_strategy")
        if "keyframe_mapping_path" in kwargs:
            kwargs.pop("keyframe_mapping_path")
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.video_sampling_strategy = video_sampling_strategy
        self.keyframe_mapping = {}
        self.sample_frames_version = None
        if self.video_sampling_strategy == "specific":
            import os
            if not os.path.exists(keyframe_mapping_path):
                raise ValueError(f"Keyframe mapping file not found at {keyframe_mapping_path}. Required when video_sampling_strategy is 'specific'.")
            with open(keyframe_mapping_path, "r", encoding="utf-8") as f:
                self.keyframe_mapping = json.load(f)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {}
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.max_pixels = int(max_pixels) if max_pixels is not None else None
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend


        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
            **llava_model_args,
        )
        processor_kwargs = {"trust_remote_code": True}
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "left"
            self._tokenizer = self._processor.tokenizer
        else:
            raise ValueError("AutoProcessor did not provide a tokenizer for the LLaVA-OneVision model.")

        self._config = self._model.config
        self._max_length = getattr(self._tokenizer, "model_max_length", None)
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.chat_template = conv_template if conv_template and "{%" in conv_template else None
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def _apply_chat_template(self, messages, add_generation_prompt: bool):
        if self.chat_template is not None:
            self.tokenizer.chat_template = self.chat_template
        elif self.tokenizer.chat_template is None and getattr(self._processor, "chat_template", None) is None:
            self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
        if hasattr(self._processor, "apply_chat_template"):
            return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def _strip_legacy_visual_tokens(self, text: str) -> str:
        return re.sub(r"(<image>|<video>)\s*", "", text).lstrip()

    def _build_user_content(self, text: str, visuals, task_type: str):
        if visuals is None:
            return text

        text = self._strip_legacy_visual_tokens(text)
        if task_type == "video":
            video_item = {"type": "video", "video": visuals}
            if self.max_pixels is not None:
                video_item["max_pixels"] = self.max_pixels
            return [video_item, {"type": "text", "text": text}]
        if task_type == "image":
            image_items = []
            for image in visuals:
                image_item = {"type": "image", "image": image}
                if self.max_pixels is not None:
                    image_item["max_pixels"] = self.max_pixels
                image_items.append(image_item)
            return image_items + [{"type": "text", "text": text}]

        return text

    def _drop_unsupported_model_inputs(self, model_inputs):
        # LLaVA-OneVision-1.5 uses Qwen2.5-VL's processor, which may emit this
        # timing field, but the remote LLaVA model forward does not consume it.
        model_inputs.pop("second_per_grid_ts", None)
        return model_inputs

    def _ensure_pil_images(self, frames):
        pil_frames = []
        for frame in frames:
            if isinstance(frame, PIL.Image.Image):
                pil_frames.append(frame)
            else:
                pil_frames.append(PIL.Image.fromarray(frame))
        return pil_frames

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            continuation = doc_to_target if isinstance(doc_to_target, str) else doc_to_target(self.task_dict[task][split][doc_id])

            task_type = "text"
            visuals = None
            placeholder_count = 0

            if visual is not None and visual != []:
                visuals = self.flatten([visual])
                if isinstance(visuals[0], PIL.Image.Image):
                    task_type = "image"
                    placeholder_count = len(visuals)
                elif isinstance(visuals[0], str):
                    try:
                        doc_data = self.task_dict[task][split][doc_id]
                        question_key = None
                        for possible_key in ["question_id", "id", "ID", "Question_ID", "questionId"]:
                            if possible_key in doc_data:
                                question_key = str(doc_data[possible_key])
                                break
                        if question_key is None:
                            question_key = str(doc_id)

                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visuals, self.max_frames_num, question_key=question_key, task=task)
                        else:
                            frames = read_video_pyav(visuals[0], num_frm=self.max_frames_num)
                        visuals = self._ensure_pil_images(frames)
                        task_type = "video"
                        placeholder_count = len(visuals) if self.token_strategy == "multiple" else 1
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        visuals = None
                        task_type = "text"

            messages = [{"role": "user", "content": self._build_user_content(contexts, visuals, task_type)}, {"role": "assistant", "content": continuation}]
            prompt = self._apply_chat_template(messages[:-1], add_generation_prompt=True)
            prompt_and_continuation = self._apply_chat_template(messages, add_generation_prompt=False)

            if visuals is None:
                model_inputs = self._processor(text=[prompt_and_continuation], return_tensors="pt")
            elif task_type == "video":
                model_inputs = self._processor(text=[prompt_and_continuation], videos=[visuals], return_tensors="pt")
            else:
                model_inputs = self._processor(text=[prompt_and_continuation], images=visuals, return_tensors="pt")
            model_inputs = self._drop_unsupported_model_inputs(model_inputs)

            model_inputs = model_inputs.to(self.device, self.model.dtype)

            labels = model_inputs["input_ids"].clone()
            context_ids = self._processor(text=[prompt], return_tensors="pt")["input_ids"]
            labels[0, : context_ids.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = model_inputs["input_ids"][:, context_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, context_ids.shape[1] : model_inputs["input_ids"].shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num, question_key=None, task=None):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
            video_path = video_path[0]
            
        total_frame_num = len(vr)
        
        if self.video_sampling_strategy == "specific":
            found_indices = None
            if self.keyframe_mapping and question_key is not None:
                for dataset_name, questions in self.keyframe_mapping.items():
                    if question_key in questions:
                        found_indices = questions[question_key]
                        break
            if found_indices is None or len(found_indices) == 0:
                raise ValueError(f"Specific frame indices not found or empty for question ID: {question_key} in {task}")
            frame_idx = found_indices
        else:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            
        # ====== 新增：持久化保存采样帧用于分析 ======
        import os
        import cv2
        import re
        from pathlib import Path
        
        extracted_model_name = self.pretrained.split("/")[-1] if "/" in self.pretrained else self.pretrained
        base_model_dir = os.path.join("sample_frames", f"{extracted_model_name}-{max_frames_num}f")
        
        if self.sample_frames_version is None:
            env_key = f"SAMPLE_FRAMES_VERSION_{extracted_model_name}_{max_frames_num}"
            if env_key not in os.environ:
                os.makedirs(base_model_dir, exist_ok=True)
                existing_versions = []
                for d in os.listdir(base_model_dir):
                    if os.path.isdir(os.path.join(base_model_dir, d)) and re.match(r"^v_\d+$", d):
                        existing_versions.append(int(d.split("_")[1]))
                next_version = max(existing_versions) + 1 if existing_versions else 1
                os.environ[env_key] = f"v_{next_version:02d}"
            self.sample_frames_version = os.environ[env_key]
            
        base_save_dir = os.path.join(base_model_dir, self.sample_frames_version)
        os.makedirs(base_save_dir, exist_ok=True)
        video_stem = Path(video_path).stem
        
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        
        for i, idx in enumerate(frame_idx):
            try:
                frame_file = f"{video_stem}_frame{idx:04d}.jpg"
                save_path = os.path.join(base_save_dir, frame_file)
                cv2.imwrite(save_path, cv2.cvtColor(spare_frames[i], cv2.COLOR_RGB2BGR))
            except Exception as e:
                eval_logger.warning(f"Failed to save sampled frame {idx}: {e}")
        # ============================================

        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            visuals_for_batch = None

            for visual, context in zip(batched_visuals, batched_contexts):
                task_type = "text"
                placeholder_count = 0

                if visual is None or visual == []:
                    visuals_for_batch = None
                else:
                    if isinstance(visual[0], PIL.Image.Image):
                        visuals_for_batch = visual
                        task_type = "image"
                        placeholder_count = len(visual)
                    elif isinstance(visual[0], str):
                        try:
                            doc_data = self.task_dict[task][split][batched_doc_id[0]]
                            question_key = None
                            for possible_key in ["question_id", "id", "ID", "Question_ID", "questionId"]:
                                if possible_key in doc_data:
                                    question_key = str(doc_data[possible_key])
                                    break
                            if question_key is None:
                                question_key = str(batched_doc_id[0])

                            if self.video_decode_backend == "decord":
                                frames = self.load_video(visual, self.max_frames_num, question_key=question_key, task=task)
                            else:
                                frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                            visuals_for_batch = self._ensure_pil_images(frames)
                            task_type = "video"
                            placeholder_count = len(visuals_for_batch) if self.token_strategy == "multiple" else 1
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            visuals_for_batch = None
                            task_type = "text"

            question = context
            if utils.is_json(question):
                question_items = json.loads(question)
                messages = []
                for idx, item in enumerate(question_items):
                    role = "user" if idx % 2 == 0 else "assistant"
                    content = self._build_user_content(item["value"], visuals_for_batch, task_type) if idx == 0 and role == "user" else item["value"]
                    messages.append({"role": role, "content": content})
                prompt_question = self._apply_chat_template(messages, add_generation_prompt=True)
            else:
                messages = [{"role": "user", "content": self._build_user_content(question, visuals_for_batch, task_type)}]
                prompt_question = self._apply_chat_template(messages, add_generation_prompt=True)

            question_input.append(prompt_question)

            gen_kwargs.setdefault("max_new_tokens", 1024)
            gen_kwargs.setdefault("do_sample", False)
            gen_kwargs.setdefault("num_beams", 1)

            if not gen_kwargs.get("do_sample", True):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)

            if visuals_for_batch is None:
                model_inputs = self._processor(text=question_input, return_tensors="pt")
            elif task_type == "video":
                # 传入 videos 参数
                model_inputs = self._processor(text=question_input, videos=[visuals_for_batch], return_tensors="pt")
            else:
                model_inputs = self._processor(text=question_input, images=visuals_for_batch, return_tensors="pt")
            model_inputs = self._drop_unsupported_model_inputs(model_inputs)

            model_inputs = model_inputs.to(self.device, self.model.dtype)

            with torch.inference_mode():
                cont = self.model.generate(**model_inputs, use_cache=self.use_cache, **gen_kwargs)

            cont = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs["input_ids"], cont)]
            decoder = self._processor if hasattr(self._processor, "batch_decode") else self.tokenizer
            text_outputs = decoder.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)

        pbar.close()
        return res
