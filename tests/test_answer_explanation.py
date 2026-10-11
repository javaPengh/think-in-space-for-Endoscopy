"""CPU-only protocol tests using real adapter methods and fake model backends."""

import ast
import copy
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("answer_explanation", ROOT / "lmms_eval/api/answer_explanation.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def adapter_method(filename, class_name, namespace):
    # Avoid importing CUDA/transformers and unrelated optional model dependencies.
    tree = ast.parse((ROOT / "lmms_eval/models" / filename).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "generate_until")
    exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
    return namespace["generate_until"]


def request(doc_id=0):
    return types.SimpleNamespace(args=(f"question {doc_id}", {"max_new_tokens": 8, "until": ["\n"]}, lambda doc: ["clip.mp4"], doc_id, "task", "test"))


class ExplanationTests(unittest.TestCase):
    def namespace(self):
        return dict(
            List=List,
            os=os,
            tqdm=MagicMock(),
            torch=MagicMock(),
            DEFAULT_GEN_KWARGS={"max_new_tokens": 1024, "do_sample": False},
            is_blind_mode=lambda mode: mode == "none",
            strip_visual_context=lambda text: text,
            EXPLANATION_PROMPT=helper.EXPLANATION_PROMPT,
        )

    def model(self, enabled=True):
        model = MagicMock()
        model.answer_explanation = helper.AnswerExplanation(enabled, 99)
        model.visual_input_mode = "none"
        model.save_sample_frames = False
        model.rank = 0
        model.path = "local/model"
        model.task_dict = {"task": {"test": [{}, {}]}}
        return model

    def test_options_and_independent_generation_budget(self):
        for enabled in (False, "false", "0", "no"):
            self.assertFalse(helper.AnswerExplanation(enabled).enabled)
        for invalid in ("typo", ""):
            with self.assertRaises(ValueError):
                helper.AnswerExplanation(invalid)
        with self.assertRaises(ValueError):
            helper.AnswerExplanation(True, 0)
        original = {"max_new_tokens": 1, "max_length": 2, "until": ["\n"], "do_sample": True}
        result = helper.AnswerExplanation(True, 99).generation_kwargs(original)
        self.assertEqual(result, {"max_new_tokens": 99, "do_sample": False, "num_beams": 1})
        self.assertEqual(original["max_new_tokens"], 1)

    def test_qwen_two_turns_keep_answer_and_isolate_samples(self):
        generate = adapter_method("qwen3vl.py", "Qwen3VL", self.namespace())
        model = self.model()
        calls = []

        def backend(messages, kwargs, media_type, metadata):
            calls.append(copy.deepcopy(messages))
            return "B" if len(messages) == 1 else "explanation with another answer C"

        model._generate_from_messages.side_effect = backend
        requests = [request(0), request(1)]
        original = copy.deepcopy(requests[0].args[1])
        self.assertEqual(generate(model, requests), ["B", "B"])
        self.assertEqual([len(c) for c in calls], [1, 3, 1, 3])
        self.assertEqual([m["role"] for m in calls[1]], ["user", "assistant", "user"])
        self.assertEqual(calls[1][1]["content"][0]["text"], "B")
        self.assertEqual(calls[1][2]["content"][0]["text"], helper.EXPLANATION_PROMPT)
        self.assertEqual(calls[2][0]["content"][0]["text"], "question 1")
        self.assertEqual(requests[0].args[1], original)
        self.assertEqual(requests[0].answer_explanations[0]["answer"], "B")

    def test_qwen_reuses_sampled_video_frames(self):
        namespace = self.namespace()
        frames = [object(), object()]
        namespace.update(calculate_effective_fps=lambda indices, fps: fps, Image=MagicMock())
        namespace["Image"].fromarray.return_value.convert.side_effect = frames
        generate = adapter_method("qwen3vl.py", "Qwen3VL", namespace)
        model = self.model()
        model.visual_input_mode = "visual"
        model._get_media_info.return_value = ("video", "clip.mp4")
        model._sample_frame_indices.return_value = [0, 1]
        model._build_video_content.side_effect = lambda frames, fps, *args: {"type": "video", "video": frames, "sample_fps": fps}
        model.max_pixels = None
        decord = MagicMock()
        decord.VideoReader.return_value.__len__.return_value = 2
        decord.VideoReader.return_value.get_avg_fps.return_value = 1
        decord.VideoReader.return_value.get_batch.return_value.numpy.return_value = [0, 1]
        model._generate_from_messages.side_effect = ["A", "explanation"]
        with patch.dict(sys.modules, {"decord": decord}):
            self.assertEqual(generate(model, [request()]), ["A"])
        first, second = model._generate_from_messages.call_args_list
        self.assertIs(first.args[0][0], second.args[0][0])
        self.assertEqual(first.args[0][0]["content"][0]["video"], frames)
        model._sample_frame_indices.assert_called_once()
        decord.VideoReader.assert_called_once()

    def test_disabled_mode_is_single_turn(self):
        for filename, cls in [("qwen3vl.py", "Qwen3VL"), ("internvl3_5.py", "InternVL3_5")]:
            with self.subTest(model=cls):
                model = self.model(False)
                model._generate_from_messages.return_value = "A"
                model.model.chat.return_value = ("A", [("question", "A")])
                req = request()
                generate = adapter_method(filename, cls, self.namespace())
                self.assertEqual(generate(model, [req]), ["A"])
                self.assertFalse(hasattr(req, "answer_explanations"))
                backend = model.model.chat if cls == "InternVL3_5" else model._generate_from_messages
                backend.assert_called_once()

    def test_internvl_reuses_pixels_history_and_handles_repeats(self):
        namespace = self.namespace()
        pixels = object()
        namespace.update(load_video=MagicMock(return_value=(pixels, [1, 1])), get_question_key=lambda doc, doc_id: str(doc_id))
        generate = adapter_method("internvl3_5.py", "InternVL3_5", namespace)
        model = self.model()
        model.visual_input_mode = "visual"
        model._get_media_inputs.return_value = ("video", ["clip.mp4"])
        model._cast_and_move_pixels.side_effect = lambda value: value
        seen_histories = []

        def chat(tokenizer, pixel_values, question, kwargs, **options):
            self.assertIs(pixel_values, pixels)
            self.assertEqual(options["num_patches_list"], [1, 1])
            seen_histories.append(options["history"])
            if options["history"] is None:
                return "A", [(question, "A")]
            self.assertEqual(options["history"][0][1], "A")
            self.assertEqual(question, helper.EXPLANATION_PROMPT)
            self.assertEqual(kwargs["max_new_tokens"], 99)
            return "explanation", options["history"] + [(question, "explanation")]

        model.model.chat.side_effect = chat
        repeated = request()
        self.assertEqual(generate(model, [repeated, repeated]), ["A", "A"])
        self.assertEqual(len(repeated.answer_explanations), 2)
        self.assertIsNone(seen_histories[0])
        self.assertIsNone(seen_histories[2])
        self.assertEqual(namespace["load_video"].call_count, 2)


if __name__ == "__main__":
    unittest.main()
