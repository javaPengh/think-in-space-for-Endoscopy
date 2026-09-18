"""Data provenance and compatibility checks for question-level analysis."""

import copy
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = load_module("question_matrix", "tools/update_eval_question_matrix.py")
compare = load_module("compare_answers", "tools/compare_internvl_natural_vs_restricted.py")


def sample(explanations=None):
    log = {
        "doc_id": 0,
        "doc": {"question_id": "q-1", "question": "Which side?", "options": ["A. left", "B. right"], "question_type": "relative_direction", "answer_type": "multiple_choice"},
        "target": "A",
        "resps": [["B"]],
        "filtered_resps": ["B"],
        "vsibench_score": {"accuracy": 0, "prediction": "B", "natural_prediction": "OLD FIRST-TURN COT"},
    }
    if explanations is not None:
        log["answer_explanations"] = explanations
    return {
        "time": "2026-09-18T12:00:00",
        "args": {"model": "qwen3vl", "log_samples_suffix": "qwen3vl_8b_16f_explain", "model_args": "max_frames_num=16,explain_answer=true", "data_version": "v2"},
        "logs": [log],
    }


class QuestionMatrixTests(unittest.TestCase):
    def test_second_turn_only_and_first_answer_still_scored(self):
        data = sample([[{"answer": "B", "prompt": "Please explain your answer step by step.", "explanation": "Actually I think A."}]])
        row = matrix.build_question_rows(data, ROOT / "logs/test/vsibench.json")[0]
        self.assertEqual(row["natural_prediction"], "Actually I think A.")
        self.assertEqual(row["first_turn_prediction"], "B")
        self.assertEqual(row["restricted_prediction"], "B")
        self.assertEqual(row["score"], 0)
        self.assertIs(row["is_correct"], False)
        self.assertEqual(row["question"], "Which side?")
        self.assertEqual(row["model"], "qwen3vl_8b")
        self.assertEqual(row["explanation_source"], "second_turn")

    def test_missing_empty_and_multiple_explanations(self):
        for malformed in (None, [], {"explanation": "not the log schema"}, [[{"answer": "B", "prompt": "explain"}]]):
            with self.subTest(value=malformed):
                row = matrix.build_question_rows(sample(malformed), ROOT / "x.json")[0]
                self.assertEqual(row["natural_prediction"], "")
                self.assertEqual(row["explanation_status"], "missing")
        records = [[{"answer": "B", "explanation": ""}, {"answer": "C", "explanation": "repeat 2"}], [{"answer": "D", "explanation": "request 2"}]]
        row = matrix.build_question_rows(sample(records), ROOT / "x.json")[0]
        self.assertEqual(len(row["answer_explanations"]), 3)
        self.assertEqual(row["answer_explanations"][2]["request_index"], 1)
        self.assertEqual(row["answer_explanations"][1]["repeat_index"], 1)
        self.assertEqual(row["explanation_status"], "available")
        empty = matrix.build_question_rows(sample([[{"explanation": "   "}]]), ROOT / "x.json")[0]
        self.assertEqual(empty["explanation_status"], "empty")

    def test_legacy_migration_preserves_first_turn_without_fabricating_explanation(self):
        legacy = {"natural_prediction": "historical CoT", "score": 50, "is_correct": None, "sample_path": "logs/run_natural_vsibench.json"}
        row = matrix._normalize_row(legacy)
        self.assertEqual(row["first_turn_prediction"], "historical CoT")
        self.assertEqual(row["natural_prediction"], "")
        self.assertEqual(row["answer_mode"], "natural")
        self.assertEqual(row["score"], 50)
        self.assertEqual(matrix._normalize_row(row), row)
        self.assertEqual(legacy["natural_prediction"], "historical CoT")

    def test_refresh_migrates_persisted_rows_and_reimports_available_sources_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "samples.json"
            source.write_text(json.dumps(sample([[{"answer": "B", "explanation": "new explanation"}]])), encoding="utf-8")
            data = path / "matrix.json"
            missing = {"sample_path": "logs/nonexistent/vsibench.json", "natural_prediction": "keep old text", "doc_id": 4}
            old = {"sample_path": str(source), "natural_prediction": "old", "doc_id": 0}
            data.write_text(json.dumps({"rows": [old, old, missing]}), encoding="utf-8")
            html = path / "nested/matrix.html"
            matrix.refresh_question_matrix_html(data, html)
            rows = json.loads(data.read_text(encoding="utf-8"))["rows"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["natural_prediction"], "new explanation")
            self.assertEqual(rows[1]["first_turn_prediction"], "keep old text")
            self.assertEqual(rows[1]["natural_prediction"], "")
            self.assertTrue(html.is_file())

    def test_reimport_is_idempotent_and_separate_runs_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data, page = root / "s.json", root / "m.json", root / "pages/m.html"
            source.write_text(json.dumps(sample()), encoding="utf-8")
            matrix.update_question_matrix_from_sample_file(source, data, page)
            matrix.update_question_matrix_from_sample_file(source, data, page)
            self.assertEqual(len(json.loads(data.read_text(encoding="utf-8"))["rows"]), 1)
            source2 = root / "s2.json"
            source2.write_text(json.dumps(sample()), encoding="utf-8")
            matrix.update_question_matrix_from_sample_file(source2, data, page)
            self.assertEqual(len(json.loads(data.read_text(encoding="utf-8"))["rows"]), 2)

    def test_inline_data_is_safe_and_page_is_standalone(self):
        hostile = "</script><script>alert('x')</script>&__MATRIX_JS__"
        data = sample([[{"explanation": hostile}]])
        rows = matrix.build_question_rows(data, ROOT / "x.json")
        page = matrix.render_question_matrix_html({"rows": rows})
        embedded = re.search(r'<script id="matrix-data" type="application/json">(.*?)</script>', page, re.S).group(1)
        self.assertNotIn("</script>", embedded)
        self.assertEqual(json.loads(embedded)["rows"][0]["natural_prediction"], hostile)
        self.assertNotIn("<script src=", page)
        self.assertNotIn("__MATRIX_CSS__", page)
        self.assertNotIn("__MATRIX_DATA__", page)
        self.assertIn("自然输出 · 第二轮解释", page)

    def test_numeric_scores_and_legacy_comparison_keep_their_meaning(self):
        data = sample([[{"answer": "2", "explanation": "explanation"}]])
        data["logs"][0]["doc"]["answer_type"] = "numeric"
        data["logs"][0]["vsibench_score"] = {matrix.MRA_METRIC: 0.5, "prediction": "2"}
        row = matrix.build_question_rows(data, ROOT / "x.json")[0]
        self.assertEqual(row["score"], 50)
        self.assertIsNone(row["is_correct"])
        self.assertTrue(row["is_scored"])
        self.assertEqual(compare.path_kind([row]), "direct")
        natural = copy.deepcopy(row)
        natural.update(answer_mode="natural", first_turn_prediction="original CoT")
        self.assertEqual(compare.path_kind([natural]), "natural")
        compared = compare.build_comparison_rows([natural], [row])[0]
        self.assertEqual(compared["natural_output"], "original CoT")


if __name__ == "__main__":
    unittest.main()
