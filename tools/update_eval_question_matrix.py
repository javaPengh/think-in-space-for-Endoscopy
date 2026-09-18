import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_DATA_PATH = DOCS_DIR / "eval_question_matrix.json"
DEFAULT_HTML_PATH = DOCS_DIR / "eval_question_matrix.html"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "question_matrix.html"
SCHEMA_VERSION = 2
MRA_METRIC = "MRA:.5:.95:.05"
SOURCE_ID_KEYS = ("question_id", "id", "ID", "Question_ID", "questionId")
NUMERIC_QUESTION_TYPES = {
    "counting(object)",
    "counting(action)",
    "polyp_size_estimation(ref)",
    "object_abs_distance",
    "object_counting",
    "action_counting",
    "object_size_estimation",
    "polyp_size_estimation",
    "room_size_estimation",
}


def update_question_matrix_from_sample_file(sample_file_path, data_path=None, html_path=None):
    sample_path = Path(sample_file_path).expanduser().resolve()
    data_path = Path(data_path or DEFAULT_DATA_PATH).expanduser().resolve()
    html_path = Path(html_path or DEFAULT_HTML_PATH).expanduser().resolve()

    with sample_path.open("r", encoding="utf-8") as f:
        sample_data = json.load(f)

    run_rows = build_question_rows(sample_data, sample_path)
    matrix_data = _read_matrix_data(data_path)
    sample_path_text = _relative_or_absolute(sample_path)
    rows = [row for row in matrix_data.get("rows", []) if str(row.get("sample_path", "")).replace("\\", "/") != sample_path_text]
    rows.extend(run_rows)
    rows.sort(key=lambda row: (str(row.get("timestamp", "")), str(row.get("data_version", "default")), str(row.get("model", "")), str(row.get("sampling_strategy", "")), _sortable_doc_id(row.get("doc_id"))))

    matrix_data = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(matrix_data, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_question_matrix_html(matrix_data), encoding="utf-8")
    return {"data_path": str(data_path), "html_path": str(html_path), "row_count": len(run_rows)}


def refresh_question_matrix_html(data_path=None, html_path=None):
    data_path = Path(data_path or DEFAULT_DATA_PATH).expanduser().resolve()
    html_path = Path(html_path or DEFAULT_HTML_PATH).expanduser().resolve()
    matrix_data = _read_matrix_data(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(matrix_data, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_question_matrix_html(matrix_data), encoding="utf-8")
    return {"data_path": str(data_path), "html_path": str(html_path), "row_count": len(matrix_data.get("rows", []))}


def build_question_rows(sample_data, sample_path):
    args = sample_data.get("args", {})
    model_args = _parse_model_args(args.get("model_args", ""))
    timestamp = sample_data.get("time") or _timestamp_from_path(sample_path) or datetime.now().isoformat(timespec="seconds")
    model = _resolve_model_name(args, model_args, sample_path)
    sampling_strategy = _sampling_label(_sampling_record(model_args))
    run_note = args.get("run_note") or args.get("note") or ""
    data_version = str(args.get("data_version") or args.get("dataset_version") or "default").strip() or "default"
    sample_path_text = _relative_or_absolute(sample_path)

    rows = []
    for log in sample_data.get("logs", []):
        score_doc = log.get("vsibench_score") or {}
        doc = log.get("doc") or {}
        source_doc = {**doc, **score_doc}
        eval_doc_id = log.get("doc_id")
        doc_id = _source_doc_id(source_doc, doc, eval_doc_id)
        answer_type = source_doc.get("answer_type") or _infer_answer_type(source_doc.get("question_type"))
        score = _score_percent(score_doc, answer_type)
        is_correct = _choice_is_correct(score_doc, answer_type)
        is_scored = _numeric_is_scored(score, answer_type)
        first_turn_prediction = _first_scalar(log.get("resps"))
        if first_turn_prediction is None:
            first_turn_prediction = score_doc.get("natural_prediction", "")
        explanations = _explanation_records(log.get("answer_explanations"))
        natural_prediction = explanations[0]["explanation"] if explanations else ""

        restricted_prediction = score_doc.get("restricted_prediction")
        extraction_was_attempted = score_doc and ("restricted_prediction" in score_doc or "prediction" in score_doc)
        if restricted_prediction is None:
            restricted_prediction = score_doc.get("prediction")
        if restricted_prediction is None and extraction_was_attempted:
            restricted_prediction = "提取失败"
        if restricted_prediction is None:
            restricted_prediction = _first_scalar(log.get("filtered_resps"))
        row_id_seed = f"{sample_path_text}|{doc_id}|{eval_doc_id}"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "row_id": hashlib.sha256(row_id_seed.encode("utf-8")).hexdigest()[:12],
                "timestamp": timestamp,
                "model": model,
                "data_version": data_version,
                "sampling_strategy": sampling_strategy,
                "note": run_note,
                "question_type": source_doc.get("question_type", ""),
                "answer_type": answer_type or "",
                "doc_id": doc_id,
                "eval_doc_id": eval_doc_id,
                "question": source_doc.get("question", ""),
                "natural_prediction": natural_prediction if natural_prediction is not None else "",
                "first_turn_prediction": first_turn_prediction,
                "answer_mode": _answer_mode(args, sample_path_text),
                "answer_explanations": explanations,
                "explanation_source": "second_turn" if explanations else "",
                "explanation_status": _explanation_status(explanations),
                "restricted_prediction": restricted_prediction if restricted_prediction is not None else "",
                "ground_truth": source_doc.get("ground_truth", log.get("target", "")),
                "options": _format_options(source_doc.get("options")),
                "score": score,
                "is_correct": is_correct,
                "is_scored": is_scored,
                "sample_path": sample_path_text,
            }
        )
    return rows


def render_question_matrix_html(data):
    payload = dict(data, rows=[_normalize_row(row) for row in data.get("rows", [])], schema_version=SCHEMA_VERSION)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = template.replace("__MATRIX_CSS__", TEMPLATE_PATH.with_suffix(".css").read_text(encoding="utf-8"))
    template = template.replace("__MATRIX_JS__", TEMPLATE_PATH.with_suffix(".js").read_text(encoding="utf-8"))
    return template.replace("__MATRIX_DATA__", _json_for_script(payload))


def _explanation_records(value):
    """Keep request/repeat identity; never mistake an answer or prompt for an explanation."""
    records = []
    if not isinstance(value, list):
        return records
    for request_index, repeats in enumerate(value):
        if not isinstance(repeats, list):
            continue
        for repeat_index, record in enumerate(repeats):
            if isinstance(record, dict) and isinstance(record.get("explanation"), str):
                records.append(dict(record, request_index=request_index, repeat_index=repeat_index))
    return records


def _explanation_status(records):
    if not records:
        return "missing"
    return "available" if any(record.get("explanation", "").strip() for record in records) else "empty"


def _answer_mode(args, path):
    mode = args.get("answer_mode")
    if mode in {"natural", "restricted"}:
        return mode
    return "natural" if "_natural" in str(args.get("log_samples_suffix", "")) or "_natural" in str(path) else "restricted"


def _normalize_row(row):
    row = dict(row)
    if row.get("schema_version", 0) < SCHEMA_VERSION:
        # Historical natural_prediction contains first-turn text, not a follow-up.
        row.setdefault("first_turn_prediction", row.get("natural_prediction", ""))
        row["natural_prediction"] = ""
        row["answer_explanations"] = []
        row["explanation_source"] = ""
        row["explanation_status"] = "missing"
    row["schema_version"] = SCHEMA_VERSION
    row.setdefault("data_version", "default")
    row.setdefault("answer_mode", _answer_mode({}, row.get("sample_path", "")))
    return row


def _json_for_script(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _read_matrix_data(data_path):
    if not data_path.exists():
        return {"rows": []}
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {"rows": data}
    rows = data.get("rows", [])
    sources = {}
    migrated = []
    for row in rows:
        path_text = row.get("sample_path", "")
        if row.get("schema_version", 0) < SCHEMA_VERSION and path_text:
            if path_text not in sources:
                source = Path(path_text.replace("\\", "/"))
                if not source.is_absolute():
                    source = REPO_ROOT / source
                sources[path_text] = None
                if source.is_file():
                    try:
                        sample_data = json.loads(source.read_text(encoding="utf-8"))
                        if isinstance(sample_data, dict) and isinstance(sample_data.get("logs"), list):
                            sources[path_text] = build_question_rows(sample_data, source)
                    except (OSError, ValueError):
                        pass  # Keep historical answers even if their source is unavailable.
                if sources[path_text] is not None:
                    migrated.extend(sources[path_text])
            if sources[path_text] is not None:
                continue
        migrated.append(_normalize_row(row))
    data["rows"] = migrated
    data["schema_version"] = SCHEMA_VERSION
    return data


def _parse_model_args(model_args):
    parsed = {}
    for part in str(model_args or "").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _resolve_model_name(args, model_args, sample_path):
    suffix_model = _model_name_from_suffix(args.get("log_samples_suffix"))
    if suffix_model:
        return suffix_model

    model_family = args.get("model") or model_args.get("model_version") or model_args.get("pretrained") or "unknown"
    pretrained = model_args.get("pretrained") or model_args.get("model_version") or model_family
    searchable = f"{pretrained} {sample_path}".lower()
    if model_family == "internvl3_5":
        if "internvl3_5-2b" in searchable or "internvl3_5_2b" in searchable:
            return "internvl3_5_2b"
        if "internvl3_5-8b" in searchable or "internvl3_5_8b" in searchable:
            return "internvl3_5_8b"
    return str(model_family)


def _model_name_from_suffix(suffix):
    text = str(suffix or "").strip()
    if not text:
        return None
    changed = True
    while changed:
        changed = False
        for token in ["_natural", "_blind", "_file", "_explain"]:
            if text.endswith(token):
                text = text[: -len(token)]
                changed = True
        new_text = re.sub(r"_\d+(?:\.\d+)?f$", "", text)
        if new_text != text:
            text = new_text
            changed = True
    return text or None


def _source_doc_id(*sources):
    for source in sources[:-1]:
        if not isinstance(source, dict):
            continue
        for key in SOURCE_ID_KEYS:
            value = source.get(key)
            if value is not None and str(value).strip():
                return _normalize_doc_id(value)
    return _normalize_doc_id(sources[-1])


def _normalize_doc_id(value):
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        return int(float(text))
    return value


def _sortable_doc_id(value):
    normalized = _normalize_doc_id(value)
    try:
        return (0, int(normalized))
    except (TypeError, ValueError):
        return (1, str(normalized))


def _sampling_record(model_args):
    visual_input_mode = model_args.get("visual_input_mode", "visual")
    video_input_mode = model_args.get("video_input_mode")
    strategy = model_args.get("video_sampling_strategy", "uniform")
    video_sample_fps = _number_or_none(model_args.get("video_sample_fps") or model_args.get("video_fps"))
    if visual_input_mode == "none":
        strategy = "blind"
        video_sample_fps = None
    elif video_input_mode == "file" and strategy == "uniform" and video_sample_fps is None:
        strategy = "fps"
        video_sample_fps = 1
    return {
        "strategy": strategy,
        "visual_input_mode": visual_input_mode,
        "video_input_mode": video_input_mode,
        "video_sample_fps": video_sample_fps,
        "max_frames_num": _number_or_none(model_args.get("max_frames_num")),
    }


def _sampling_label(sampling):
    if sampling.get("visual_input_mode") == "none" or sampling.get("strategy") == "blind":
        return "blind"
    strategy = sampling.get("strategy") or "uniform"
    suffix = "_file" if sampling.get("video_input_mode") == "file" else ""
    if strategy == "fps":
        fps = sampling.get("video_sample_fps") or 1
        return f"fps_{_format_number(fps)}{suffix}"
    if strategy == "uniform":
        frames = sampling.get("max_frames_num")
        if frames is not None:
            return f"uniform_{_format_number(frames)}f{suffix}"
        return f"uniform{suffix}"
    if strategy == "specific":
        return f"specific{suffix}"
    return f"{strategy}{suffix}"


def _number_or_none(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return number


def _format_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def _infer_answer_type(question_type):
    question_type = str(question_type or "")
    if question_type in NUMERIC_QUESTION_TYPES:
        return "numeric"
    if question_type:
        return "multiple_choice"
    return ""


def _score_percent(score_doc, answer_type):
    if not score_doc:
        return None
    if _is_choice(answer_type) and isinstance(score_doc.get("accuracy"), (int, float)):
        return float(score_doc["accuracy"]) * 100.0
    if isinstance(score_doc.get(MRA_METRIC), (int, float)):
        return float(score_doc[MRA_METRIC]) * 100.0
    return None


def _choice_is_correct(score_doc, answer_type):
    if not _is_choice(answer_type) or "accuracy" not in score_doc:
        return None
    return bool(score_doc.get("accuracy"))


def _numeric_is_scored(score, answer_type):
    if _is_choice(answer_type) or score is None:
        return None
    return float(score) > 0.0


def _is_choice(answer_type):
    return str(answer_type or "").lower() in {"mca", "mac", "mcq", "multiple_choice", "choice"}


def _format_options(options):
    if options is None:
        return ""
    if isinstance(options, list):
        return "\n".join(str(option) for option in options)
    return str(options)


def _first_scalar(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            scalar = _first_scalar(item)
            if scalar is not None:
                return scalar
        return None
    if isinstance(value, dict):
        for item in value.values():
            scalar = _first_scalar(item)
            if scalar is not None:
                return scalar
        return None
    return value


def _timestamp_from_path(path):
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", str(path))
    if match:
        return match.group(1)
    return None


def _relative_or_absolute(path):
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main():
    parser = argparse.ArgumentParser(description="Build VSI-Bench question-level answer matrix from a sample JSON file.")
    parser.add_argument("sample_file", nargs="?", help="Path to a task sample JSON file, e.g. logs/.../vsibench.json")
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH), help="Output JSON path.")
    parser.add_argument("--html_path", default=str(DEFAULT_HTML_PATH), help="Output HTML path.")
    parser.add_argument("--refresh_only", action="store_true", help="Migrate existing matrix rows and regenerate HTML; reload legacy source logs when available.")
    args = parser.parse_args()
    if args.refresh_only:
        paths = refresh_question_matrix_html(args.data_path, args.html_path)
    elif args.sample_file:
        paths = update_question_matrix_from_sample_file(args.sample_file, args.data_path, args.html_path)
    else:
        parser.error("sample_file is required unless --refresh_only is set")
    print(json.dumps(paths, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
