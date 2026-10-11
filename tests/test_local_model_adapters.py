"""CPU-only checks of real adapter methods, with no weights or ML runtime required."""

import ast
import copy
import gc
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "lmms_eval/models"
spec = importlib.util.spec_from_file_location("answer_explanation", ROOT / "lmms_eval/api/answer_explanation.py")
explanation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explanation)


def execute_adapter(filename, namespace):
    tree = ast.parse((MODELS / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)


class FakeDevice:
    def __init__(self, value):
        self.value = str(value)
        self.type = self.value.split(":")[0]

    def __str__(self):
        return self.value


class FakeTensor:
    def __init__(self, count=3, floating=False):
        self.count = count
        self.floating = floating
        self.casts = []
        self.shape = (1, count)

    def is_floating_point(self):
        return self.floating

    def to(self, **kwargs):
        self.casts.append(kwargs)
        return self

    def numel(self):
        return self.count

    def __getitem__(self, key):
        return FakeTensor(self.count - key[1].start)


class FakeInputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        self.device = device
        return self


class FakeQwen:
    @property
    def model(self):
        return self._model

    def _reset_token_usage(self):
        pass


class LocalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.backend = MagicMock()
        self.backend.get_input_embeddings.return_value = types.SimpleNamespace(weight=types.SimpleNamespace(device="cuda:2"))
        self.model_class = MagicMock()
        self.model_class.from_pretrained.return_value.eval.return_value = self.backend
        self.processor = MagicMock()
        self.processor.apply_chat_template.return_value = "rendered prompt"
        self.inputs = FakeInputs(input_ids=FakeTensor(), attention_mask=FakeTensor(), pixel_values=FakeTensor(floating=True))
        self.processor.return_value = self.inputs
        accelerator = types.SimpleNamespace(num_processes=1, process_index=0, local_process_index=0)
        self.namespace = {
            "__file__": str(MODELS / "local_native_video.py"),
            "gc": gc,
            "json": json,
            "math": math,
            "os": os,
            "Path": Path,
            "timedelta": timedelta,
            "np": types.SimpleNamespace(asarray=lambda image: image, stack=lambda frames: list(frames)),
            "torch": types.SimpleNamespace(device=FakeDevice, bfloat16="bf16", is_tensor=lambda value: isinstance(value, FakeTensor), inference_mode=MagicMock(), cuda=MagicMock()),
            "transformers": types.SimpleNamespace(__version__="compatible", Qwen3_5ForConditionalGeneration=self.model_class),
            "Accelerator": MagicMock(return_value=accelerator),
            "InitProcessGroupKwargs": MagicMock(),
            "AutoProcessor": MagicMock(),
            "eval_logger": MagicMock(),
            "lmms": type("LMMS", (), {}),
            "Qwen3VL": FakeQwen,
            "AnswerExplanation": explanation.AnswerExplanation,
            "normalize_visual_input_mode": lambda mode: mode,
            "is_blind_mode": lambda mode: mode == "none",
            "inference_max_memory": MagicMock(return_value={0: 1000, 1: 1000}),
            "register_model": lambda name: lambda cls: cls,
        }
        self.namespace["AutoProcessor"].from_pretrained.return_value = self.processor
        execute_adapter("local_native_video.py", self.namespace)
        execute_adapter("qwen3_5_27b.py", self.namespace)
        self.cls = self.namespace["Qwen3_5_27B"]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")

    def adapter(self, **kwargs):
        return self.cls(pretrained=str(self.path), **kwargs)

    def test_missing_weights_and_wrong_architecture_fail_before_loading(self):
        with self.assertRaises(FileNotFoundError):
            self.cls(pretrained=str(self.path / "missing"))
        (self.path / "config.json").write_text('{"model_type":"gemma3"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "model_type"):
            self.adapter()
        self.model_class.from_pretrained.assert_not_called()

    def test_old_transformers_has_actionable_error(self):
        self.namespace["transformers"] = types.SimpleNamespace(__version__="4.57.6")
        with self.assertRaisesRegex(ImportError, "Qwen3_5ForConditionalGeneration"):
            self.adapter()
        self.model_class.from_pretrained.assert_not_called()

    def test_loader_is_offline_and_reserves_memory(self):
        model = self.adapter()
        options = self.model_class.from_pretrained.call_args.kwargs
        self.assertTrue(options["local_files_only"])
        self.assertEqual(options["device_map"], "auto")
        self.assertEqual(options["attn_implementation"], "sdpa")
        self.assertEqual(options["max_memory"], {0: 1000, 1: 1000})
        self.namespace["AutoProcessor"].from_pretrained.assert_called_once_with(str(self.path), local_files_only=True)
        self.assertEqual(str(model._device), "cuda:2")

    def test_embedding_dispatch_hook_determines_input_device(self):
        self.backend.get_input_embeddings.return_value = types.SimpleNamespace(weight=types.SimpleNamespace(device="meta"), _hf_hook=types.SimpleNamespace(execution_device=3))
        self.assertEqual(str(self.adapter()._device), "cuda:3")

    def test_parallel_processes_rejected_before_weight_load(self):
        self.namespace["Accelerator"].return_value.num_processes = 2
        with self.assertRaisesRegex(ValueError, "num_processes 1"):
            self.adapter(device_map="balanced")
        self.model_class.from_pretrained.assert_not_called()

    def test_option_validation_precedes_loading(self):
        for options in ({"max_frames_num": 0}, {"video_sample_fps": "nan"}, {"video_sample_fps": 0}, {"enable_thinking": "typo"}, {"batch_size": 2}, {"unknown_option": 1}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                self.adapter(**options)
        self.model_class.from_pretrained.assert_not_called()

    def test_native_video_preserves_irregular_timestamps_and_explanation_visuals(self):
        model = self.adapter()
        model._rgb_image = lambda frame: frame
        content = model._build_video_content(["frame A", "frame B"], 1, 25, 500, [75, 240])
        messages = [{"role": "user", "content": [content, {"type": "text", "text": "question"}]}]
        original = copy.deepcopy(messages)
        for conversation in (messages, explanation.AnswerExplanation().messages(messages, "A")):
            model._prepare_inputs(conversation)
            options = self.processor.call_args.kwargs
            self.assertEqual(options["videos"], [["frame A", "frame B"]])
            self.assertEqual(options["video_metadata"][0], {"fps": 25, "total_num_frames": 500, "frames_indices": [75, 240]})
            self.assertFalse(options["do_sample_frames"])
            self.assertFalse(self.processor.apply_chat_template.call_args.kwargs["enable_thinking"])
            # Some processors pad temporal patches by modifying their metadata; history must stay unchanged.
            options["video_metadata"][0]["frames_indices"].append(240)
        self.assertEqual(messages, original)
        self.assertFalse(self.inputs.input_ids.casts)
        self.assertFalse(self.inputs["attention_mask"].casts)
        self.assertEqual(self.inputs["pixel_values"].casts[-1], {"dtype": "bf16"})

    def test_blind_text_inputs_do_not_include_visuals(self):
        model = self.adapter(visual_input_mode="none")
        model._prepare_inputs([{"role": "user", "content": [{"type": "text", "text": "question"}]}])
        options = self.processor.call_args.kwargs
        self.assertNotIn("images", options)
        self.assertNotIn("videos", options)

    def test_image_processor_receives_local_rgb_image(self):
        model = self.adapter()
        model._rgb_image = lambda value: f"rgb:{value}"
        model._prepare_inputs([{"role": "user", "content": [{"type": "image", "image": "local.png"}]}])
        self.assertEqual(self.processor.call_args.kwargs["images"], ["rgb:local.png"])

    def test_generation_trims_prompt_and_counts_new_tokens(self):
        model = self.adapter()
        model._debug_dump_prompt = MagicMock()
        model._record_token_usage = MagicMock()
        self.backend.generate.return_value = types.SimpleNamespace(sequences=FakeTensor(8))
        self.processor.batch_decode.return_value = ["B"]
        answer = model._generate_from_messages([{"role": "user", "content": [{"type": "text", "text": "question"}]}], {"max_new_tokens": 5}, "text")
        self.assertEqual(answer, "B")
        model._record_token_usage.assert_called_once_with("text", 3, 5)

    def test_gemma_unified_and_moe_use_their_correct_classes(self):
        for filename in ("gemma_4_31b.py", "gemma_4_26b_a4b.py", "gemma_4_12b.py", "qwen3_5_35b_a3b.py"):
            execute_adapter(filename, self.namespace)
        self.assertEqual(self.namespace["Gemma4_12B"].MODEL_CLASS_NAME, "Gemma4UnifiedForConditionalGeneration")
        self.assertEqual(self.namespace["Gemma4_26B_A4B"].MODEL_CLASS_NAME, "Gemma4ForConditionalGeneration")
        self.assertEqual(self.namespace["Qwen3_5_35B_A3B"].MODEL_CLASS_NAME, "Qwen3_5MoeForConditionalGeneration")
        gemma = object.__new__(self.namespace["Gemma4_12B"])
        gemma.max_soft_tokens = 140
        self.assertEqual(gemma._processor_options([1], [2]), {"images_kwargs": {"max_soft_tokens": 140}, "videos_kwargs": {"max_soft_tokens": 140}})
        with self.assertRaises(ValueError):
            self.namespace["Gemma4_31B"](max_soft_tokens=100)

    def test_internvl_wrapper_is_offline_and_rejects_unsupported_fps(self):
        class InternBackend:
            def __init__(self, **kwargs):
                self.options = kwargs

        self.namespace["InternVL3_5"] = InternBackend
        execute_adapter("internvl3_5_30b_a3b.py", self.namespace)
        cls = self.namespace["InternVL3_5_30B_A3B"]
        model = cls(pretrained=str(self.path))
        self.assertTrue(model.options["local_files_only"])
        self.assertEqual(model.options["device_map"], "balanced")
        with self.assertRaisesRegex(ValueError, "fps"):
            cls(pretrained=str(self.path), video_sample_fps=1)

    def test_six_adapters_are_registered_and_exposed_in_launcher(self):
        registry = ast.parse((MODELS / "__init__.py").read_text(encoding="utf-8"))
        available = next(ast.literal_eval(node.value) for node in registry.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "AVAILABLE_MODELS" for target in node.targets))
        script = (ROOT / "evaluate_all_in_one.sh").read_text(encoding="utf-8")
        names = ["qwen3_5_27b", "qwen3_5_35b_a3b", "internvl3_5_30b_a3b", "gemma_4_12b", "gemma_4_26b_a4b", "gemma_4_31b"]
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, available)
                tree = ast.parse((MODELS / f"{name}.py").read_text(encoding="utf-8"))
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
                self.assertEqual(cls.name, available[name])
                self.assertEqual(ast.literal_eval(cls.decorator_list[0].args[0]), name)
                case = script.split(f'    "{name}")', 1)[1].split(";;", 1)[0]
                self.assertIn(f'model_family="{name}"', case)
                self.assertIn("num_processes=1", case)

    def test_launcher_builds_six_commands_without_running_models(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash is not available for the launcher protocol check.")
        names = ["qwen3_5_27b", "qwen3_5_35b_a3b", "internvl3_5_30b_a3b", "gemma_4_12b", "gemma_4_26b_a4b", "gemma_4_31b"]
        env = dict(os.environ, VSI_LLM_EXTRACTOR_API_KEY="local-protocol-test")
        # Replace the launcher with a shell function; no Python/GPU/remote evaluation runs.
        command = 'accelerate() { printf "MOCK_LAUNCH %s\\n" "$*"; }; export -f accelerate; bash "$1" "${@:2}"'
        result = subprocess.run(
            [
                bash,
                "-c",
                command,
                "launcher-test",
                (ROOT / "evaluate_all_in_one.sh").as_posix(),
                "--model",
                ",".join(names),
                "--cuda_visible_devices",
                "0,1",
                "--num_processes",
                "4",
                "--num_frames",
                "3",
                "--video_sampling_strategy",
                "specific",
                "--save_sample_frames",
                "true",
                "--explain_answer",
                "true",
                "--limit",
                "2",
            ],
            cwd=self.path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        launches = [line for line in result.stdout.splitlines() if line.startswith("MOCK_LAUNCH ")]
        self.assertEqual(len(launches), 6)
        for name, launch in zip(names, launches):
            with self.subTest(name=name):
                self.assertIn(f"--model {name} ", launch)
                self.assertIn("--num_processes=1", launch)
                self.assertIn("video_sampling_strategy=specific", launch)
                self.assertIn("save_sample_frames=true", launch)
                self.assertIn("explain_answer=true", launch)
                self.assertIn("max_frames_num=3", launch)


if __name__ == "__main__":
    unittest.main()
