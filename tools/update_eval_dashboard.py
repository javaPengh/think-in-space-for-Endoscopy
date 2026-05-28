import argparse
import ast
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_DATA_PATH = DOCS_DIR / "eval_dashboard_data.json"
DEFAULT_HTML_PATH = DOCS_DIR / "eval_dashboard.html"
DEFAULT_BASELINE_EXCEL_PATH = Path(r"C:\Users\a2818\Desktop\QA\抽样测试.xlsx")

CHOICE_ANSWER_TYPES = {"mca", "mcq", "multiple_choice", "choice"}
NUMERIC_ANSWER_TYPES = {"na", "numeric", "number", "numerical"}
QUESTION_TYPE_METRIC_MAP = {
    "object_rel_direction_easy": "object_rel_direction_accuracy",
    "object_rel_direction_medium": "object_rel_direction_accuracy",
    "object_rel_direction_hard": "object_rel_direction_accuracy",
}


def update_dashboard_from_result_file(result_file_path, data_path=None, html_path=None, baseline_excel_path=None):
    result_path = Path(result_file_path).expanduser().resolve()
    data_path = Path(data_path or DEFAULT_DATA_PATH).expanduser().resolve()
    html_path = Path(html_path or DEFAULT_HTML_PATH).expanduser().resolve()

    with result_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    run = build_run_record(results, result_path)
    data = _read_data(data_path)
    runs = [item for item in data.get("runs", []) if item.get("result_path") != run["result_path"]]
    runs.append(run)
    runs.sort(key=lambda item: item.get("timestamp", ""))
    baselines = _resolve_baselines(data, baseline_excel_path)
    data = {"runs": runs}
    if baselines:
        data["baselines"] = baselines

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_dashboard_html(data), encoding="utf-8")
    return {"data_path": str(data_path), "html_path": str(html_path), "run_id": run["run_id"]}


def refresh_dashboard(data_path=None, html_path=None, baseline_excel_path=None):
    data_path = Path(data_path or DEFAULT_DATA_PATH).expanduser().resolve()
    html_path = Path(html_path or DEFAULT_HTML_PATH).expanduser().resolve()
    data = _read_data(data_path)
    baselines = _resolve_baselines(data, baseline_excel_path)
    data = {"runs": data.get("runs", [])}
    if baselines:
        data["baselines"] = baselines

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_dashboard_html(data), encoding="utf-8")
    return {"data_path": str(data_path), "html_path": str(html_path), "baseline_count": sum(len(items) for items in (baselines or {}).get("items", {}).values())}


def build_run_record(results, result_path):
    config = results.get("config", {})
    task_name, task_metrics = _first_task_metrics(results.get("results", {}))
    model_args = _parse_model_args(config.get("model_args", ""))
    timestamp = results.get("date") or _timestamp_from_path(result_path) or datetime.now().isoformat(timespec="seconds")

    model_family = config.get("model") or model_args.get("model_version") or model_args.get("pretrained") or "unknown"
    pretrained = model_args.get("pretrained") or model_args.get("model_version") or model_family
    model = _display_model_name(model_family, pretrained, result_path)
    visual_input_mode = model_args.get("visual_input_mode", "visual")
    video_input_mode = model_args.get("video_input_mode")
    sampling_strategy = model_args.get("video_sampling_strategy", "uniform")
    video_sample_fps = _number_or_none(model_args.get("video_sample_fps") or model_args.get("video_fps"))
    if visual_input_mode == "none":
        sampling_strategy = "blind"
        video_sample_fps = None
    elif video_input_mode == "file" and sampling_strategy == "uniform" and video_sample_fps is None:
        sampling_strategy = "fps"
        video_sample_fps = 1

    sampling = {
        "strategy": sampling_strategy,
        "visual_input_mode": visual_input_mode,
        "video_sample_fps": video_sample_fps,
        "max_frames_num": _number_or_none(model_args.get("max_frames_num")),
        "max_pixels": _number_or_none(model_args.get("max_pixels")),
    }
    if video_input_mode:
        sampling["video_input_mode"] = video_input_mode
    token_usage = _normalize_token_usage(config.get("token_usage")) or _fallback_token_usage(model, sampling)

    run_seed = f"{timestamp}|{model}|{task_name}|{result_path}"
    run_id = hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:12]
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "model": model,
        "model_family": model_family,
        "pretrained": pretrained,
        "task": task_name,
        "sampling": sampling,
        "metrics": task_metrics,
        "token_usage": token_usage,
        "result_path": _relative_or_absolute(result_path),
    }


def _display_model_name(model_family, pretrained, result_path):
    model_family = str(model_family or "unknown")
    pretrained_text = str(pretrained or "")
    result_text = str(result_path or "")
    searchable = f"{pretrained_text} {result_text}".lower()

    if model_family == "internvl3_5":
        if "internvl3_5-2b" in searchable or "internvl3_5_2b" in searchable:
            return "internvl3_5_2b"
        if "internvl3_5-8b" in searchable or "internvl3_5_8b" in searchable:
            return "internvl3_5_8b"

    return model_family


def _first_task_metrics(results_by_task):
    if not results_by_task:
        return "unknown", {}
    task_name = next(iter(results_by_task.keys()))
    raw_metrics = results_by_task.get(task_name, {})
    metrics = {}
    for key, value in raw_metrics.items():
        metric_name = key.split(",", 1)[0]
        if metric_name in {"alias", "samples", "tabulated_keys", "tabulated_results"} or metric_name.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)):
            metrics[metric_name] = value
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                nested_metric_name = str(nested_key).split(",", 1)[0]
                if nested_metric_name in {"alias", "samples", "tabulated_keys", "tabulated_results"} or nested_metric_name.endswith("_stderr"):
                    continue
                if isinstance(nested_value, (int, float)):
                    metrics[nested_metric_name] = nested_value
    return task_name, metrics


def _parse_model_args(model_args):
    parsed = {}
    for part in str(model_args or "").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _resolve_baselines(data, baseline_excel_path=None):
    candidate = _baseline_excel_candidate(baseline_excel_path)
    if candidate and candidate.exists():
        return build_baseline_records(candidate)
    return data.get("baselines")


def _baseline_excel_candidate(baseline_excel_path=None):
    if baseline_excel_path:
        return Path(baseline_excel_path).expanduser()
    if DEFAULT_BASELINE_EXCEL_PATH.exists():
        return DEFAULT_BASELINE_EXCEL_PATH
    return None


def build_baseline_records(excel_path):
    import pandas as pd

    excel_path = Path(excel_path).expanduser().resolve()
    sheets = pd.read_excel(excel_path, sheet_name=None)
    frames = [sheet_df for sheet_df in sheets.values() if not sheet_df.empty]
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    question_type_col = _find_column(df, ["question_type", "题型", "类别"])
    answer_type_col = _find_column(df, ["answer_type", "答案类型", "题目类型"])
    options_col = _find_column(df, ["options", "选项"])
    correct_col = _find_column(df, ["正确答案", "ground_truth", "answer", "答案"])
    numeric_truth_col = _find_column(df, ["真值", "ground_truth", "target", "value"])

    if not question_type_col:
        raise ValueError("Cannot calculate baselines because the Excel file has no question_type/题型 column.")

    items = {}
    summaries = []
    for question_type, group in df.groupby(question_type_col, dropna=True):
        question_type = str(question_type).strip()
        if not question_type:
            continue

        choice_rows = group[group.apply(lambda row: _row_answer_kind(row, answer_type_col, options_col) == "choice", axis=1)]
        if not choice_rows.empty:
            if not options_col or not correct_col:
                raise ValueError("Choice baselines require options/选项 and ground_truth/正确答案 columns.")
            metric_key = _metric_key_for_question_type(question_type, "choice")
            random_values = []
            correct_labels = []
            for _, row in choice_rows.iterrows():
                options = _parse_options(row.get(options_col))
                if options:
                    random_values.append(1.0 / len(options))
                label = _normalize_answer_label(row.get(correct_col))
                if label:
                    correct_labels.append(label)
            random_baseline = (sum(random_values) / len(random_values) * 100.0) if random_values else None
            frequency_baseline = None
            if correct_labels:
                counts = {}
                for label in correct_labels:
                    counts[label] = counts.get(label, 0) + 1
                frequency_baseline = max(counts.values()) / len(correct_labels) * 100.0

            metric_items = []
            if random_baseline is not None:
                metric_items.append({"kind": "random", "label": "随机基线", "value": random_baseline, "question_type": question_type, "n": int(len(choice_rows))})
            if frequency_baseline is not None:
                metric_items.append({"kind": "frequency", "label": "频率基线", "value": frequency_baseline, "question_type": question_type, "n": int(len(choice_rows))})
            if metric_items:
                items.setdefault(metric_key, []).extend(metric_items)
                summaries.extend(metric_items)

        numeric_rows = group[group.apply(lambda row: _row_answer_kind(row, answer_type_col, options_col) == "numeric", axis=1)]
        if not numeric_rows.empty:
            if not numeric_truth_col:
                raise ValueError("Numeric baselines require a ground_truth/真值 column.")
            values = [_to_float(value) for value in numeric_rows[numeric_truth_col].tolist()]
            values = [value for value in values if value is not None and value > 0]
            if values:
                metric_key = _metric_key_for_question_type(question_type, "numeric")
                median_value = _median(values)
                mean_value = sum(values) / len(values)
                median_mra = _constant_mra(values, median_value)
                mean_mra = _constant_mra(values, mean_value)
                item = {
                    "kind": "constant_mra",
                    "label": "常数基线MRA",
                    "value": median_mra,
                    "question_type": question_type,
                    "n": int(len(values)),
                    "constant": median_value,
                    "mean_constant": mean_value,
                    "mean_constant_mra": mean_mra,
                }
                items.setdefault(metric_key, []).append(item)
                summaries.append(item)

    return {
        "source_path": str(excel_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "items": items,
        "summary": summaries,
    }


def _find_column(df, candidates):
    normalized = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        column = normalized.get(str(candidate).strip().lower())
        if column is not None:
            return column
    return None


def _row_answer_kind(row, answer_type_col, options_col):
    answer_type = str(row.get(answer_type_col, "")).strip().lower() if answer_type_col else ""
    if answer_type in CHOICE_ANSWER_TYPES:
        return "choice"
    if answer_type in NUMERIC_ANSWER_TYPES:
        return "numeric"
    if options_col and _parse_options(row.get(options_col)):
        return "choice"
    return "numeric"


def _parse_options(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [item for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [item for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    label_matches = re.findall(r"(?:^|[\n,;])\s*[A-Z]\s*[\.\)]", text)
    if label_matches:
        return label_matches
    return [part.strip() for part in re.split(r"\n+", text) if part.strip()]


def _normalize_answer_label(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    match = re.match(r"^([A-Za-z])(?:[\.\)]|\s*$)", text)
    return match.group(1).upper() if match else text


def _metric_key_for_question_type(question_type, answer_kind):
    question_type = str(question_type).strip()
    if answer_kind == "choice":
        return QUESTION_TYPE_METRIC_MAP.get(question_type, f"{question_type}_accuracy")
    return f"{question_type}_MRA:.5:.95:.05"


def _to_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values):
    values = sorted(values)
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _constant_mra(values, constant):
    thresholds = [0.5 + index * 0.05 for index in range(10)]
    per_question_scores = []
    for truth in values:
        relative_error = abs(constant - truth) / truth
        per_question_scores.append(sum(1 for threshold in thresholds if relative_error < (1.0 - threshold)) / len(thresholds))
    return sum(per_question_scores) / len(per_question_scores) * 100.0


def _normalize_token_usage(token_usage):
    if not token_usage:
        return None
    input_tokens = _number_or_none(token_usage.get("input_tokens"))
    output_tokens = _number_or_none(token_usage.get("output_tokens"))
    total_tokens = _number_or_none(token_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "num_requests": _number_or_none(token_usage.get("num_requests")),
        "by_media": token_usage.get("by_media", {}),
    }


def _fallback_token_usage(model, sampling):
    if model != "qwen3vl":
        return None

    token_log_path = DOCS_DIR / "qwen3vl_token_usage.jsonl"
    if not token_log_path.exists():
        return None

    records = []
    with token_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("model") != model:
                continue
            if sampling.get("visual_input_mode") == "none":
                if record.get("visual_input_mode") != "none":
                    continue
                records.append(record)
                continue
            if record.get("visual_input_mode") == "none":
                continue
            if record.get("video_sampling_strategy") != sampling.get("strategy"):
                continue
            if _number_or_none(record.get("video_sample_fps")) != sampling.get("video_sample_fps"):
                continue
            records.append(record)

    if not records:
        return None
    records.sort(key=lambda item: str(item.get("timestamp", "")))
    return _normalize_token_usage(records[-1].get("token_usage"))


def _number_or_none(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _timestamp_from_path(path):
    for part in path.parts:
        if "__" in part or "-" in part:
            continue
        if part.startswith("20") and len(part) >= 8:
            return part
    return None


def _relative_or_absolute(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_data(data_path):
    if not data_path.exists():
        return {"runs": []}
    try:
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            return data
    except json.JSONDecodeError:
        pass
    return {"runs": []}


def render_dashboard_html(data):
    data_json = json.dumps(data, ensure_ascii=False)
    escaped_data_json = data_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VSI-Bench Evaluation Dashboard</title>
  <style>
    :root {{ color-scheme: light; --bg:#f7f8fa; --panel:#ffffff; --ink:#1f2937; --muted:#667085; --line:#d7dce3; --accent:#2563eb; --accent2:#0f9f6e; --warn:#b45309; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); overflow-x: hidden; }}
    header {{ padding: 22px 28px 14px; border-bottom: 1px solid var(--line); background: var(--panel); }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 700; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 18px 28px 32px; display: grid; gap: 18px; max-width: 100%; overflow: hidden; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    section {{ padding: 16px; min-width: 0; }}
    section h2 {{ margin: 0 0 12px; font-size: 16px; }}
    .filters {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    input, select {{ height: 34px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; background: #fff; min-width: 180px; }}
    select:disabled {{ color: #98a2b3; background: #f2f4f7; }}
    button {{ height: 30px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; background: #fff; color: var(--ink); cursor: pointer; }}
    button:hover {{ border-color: #b8c0cc; background: #f9fafb; }}
    .danger-btn {{ border-color: #fecaca; color: #b42318; }}
    .danger-btn:hover {{ border-color: #fca5a5; background: #fff5f5; }}
    .primary-btn {{ border-color: #b42318; background: #b42318; color: #fff; }}
    .primary-btn:hover {{ border-color: #9f1f14; background: #9f1f14; }}
    .hint {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .chart {{ height: 500px; width: 100%; display: block; }}
    table {{ width: max-content; min-width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 600; background: #fafafa; position: sticky; top: 0; }}
    .table-wrap {{ max-height: 520px; max-width: 100%; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #eef4ff; color: #1d4ed8; font-size: 12px; white-space: nowrap; }}
    .empty {{ padding: 30px; text-align: center; color: var(--muted); }}
    .modal[hidden] {{ display: none; }}
    .modal {{ position: fixed; inset: 0; z-index: 10; display: grid; place-items: center; padding: 20px; background: rgba(17, 24, 39, 0.36); }}
    .dialog {{ width: min(420px, 100%); border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: 0 18px 50px rgba(16, 24, 40, 0.18); padding: 18px; }}
    .dialog h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .dialog p {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.6; }}
    .dialog-actions {{ display: flex; justify-content: flex-end; gap: 8px; margin-top: 18px; }}
    @media (max-width: 1000px) {{ .chart {{ height: 540px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>VSI-Bench Evaluation Dashboard</h1>
    <div class="subtle">Static report generated from lmms-eval results and model token usage.</div>
  </header>
  <main>
    <section>
      <h2>筛选</h2>
      <div class="filters">
        <input id="search" placeholder="搜索模型或路径" />
        <select id="modelFilter"><option value="">全部模型</option></select>
        <select id="samplingFilter"><option value="">全部采样策略</option></select>
      </div>
      <div class="hint">筛选器不会同时处于全部状态；图表只展示同模型同采样的最新记录，评估记录保留全部历史。</div>
    </section>
    <section><h2>指标对比</h2><svg id="metricChart" class="chart"></svg></section>
    <section>
      <h2>评估记录</h2>
      <div class="table-wrap"><table id="runsTable"></table></div>
    </section>
  </main>
  <div id="deleteModal" class="modal" hidden>
    <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="deleteTitle">
      <h3 id="deleteTitle">删除评估记录</h3>
      <p id="deleteTarget">确认删除这条评估记录？</p>
      <div class="dialog-actions">
        <button id="cancelDelete" type="button">取消</button>
        <button id="confirmDelete" class="primary-btn" type="button">确认删除</button>
      </div>
    </div>
  </div>
  <script type="application/json" id="dashboard-data">{escaped_data_json}</script>
  <script>
    const state = JSON.parse(document.getElementById('dashboard-data').textContent);
    const runs = (state.runs || []).slice().sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));
    const deletedRunIdsStorageKey = 'vsi_eval_dashboard_deleted_run_ids';
    const loadDeletedRunIds = () => {{
      try {{
        if (typeof localStorage === 'undefined') return new Set();
        return new Set(JSON.parse(localStorage.getItem(deletedRunIdsStorageKey) || '[]'));
      }} catch {{
        return new Set();
      }}
    }};
    const deletedRunIds = loadDeletedRunIds();
    let pendingDeleteRunId = null;
    const fmt = v => v === null || v === undefined ? 'N/A' : Number(v).toLocaleString(undefined, {{ maximumFractionDigits: 3 }});
    const fmtScore = v => v === null || v === undefined ? 'N/A' : Number(v).toLocaleString(undefined, {{ maximumFractionDigits: 1 }});
    const html = v => String(v ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
    const metric = (r, k) => r.metrics && typeof r.metrics[k] === 'number' ? r.metrics[k] : null;
    const samplingLabel = r => {{
      const s = r.sampling || {{}};
      if (s.visual_input_mode === 'none' || s.strategy === 'blind') return 'blind';
      const mode = s.video_input_mode && s.video_input_mode !== 'frames' ? `_${{s.video_input_mode}}` : '';
      if (s.strategy === 'fps') return `fps_${{s.video_sample_fps ?? 'N/A'}}${{mode}}`;
      if (s.strategy === 'uniform') return `uniform_${{s.max_frames_num ?? 'N/A'}}f${{mode}}`;
      return `${{s.strategy || 'unknown'}}${{mode}}`;
    }};
    const token = (r, k) => r.token_usage ? r.token_usage[k] : null;
    const seriesLabel = r => `${{r.model || 'unknown'}} / ${{samplingLabel(r)}}`;
    const palette = ['#2563eb', '#c2410c', '#0f9f6e', '#7c3aed', '#be123c', '#0e7490', '#a16207', '#4f46e5', '#15803d'];
    const baselines = (state.baselines && state.baselines.items) || {{}};
    const baselineStyles = {{
      random: {{ label: '随机基线', color: '#6b7280', dash: '6 4', width: 1.5 }},
      frequency: {{ label: '频率基线', color: '#b45309', dash: '2 3', width: 1.8 }},
      constant_mra: {{ label: '常数基线MRA', color: '#0f766e', dash: '8 3', width: 1.8 }},
      default: {{ label: '基线', color: '#667085', dash: '5 4', width: 1.5 }}
    }};
    const metricLabelMap = {{
      overall: '平均分',
      image_overall: '图像总体',
      video_overall: '视频数据平均分',
      'object_abs_distance_MRA:.5:.95:.05': '绝对距离',
      'object_counting_MRA:.5:.95:.05': '物体计数',
      object_rel_direction_accuracy: '相对方向',
      object_rel_distance_accuracy: '相对距离',
      'object_size_estimation_MRA:.5:.95:.05': '尺寸估计',
      route_planning_accuracy: '路径规划',
      obj_appearance_order_accuracy: '出现顺序'
    }};
    const metricLabel = key => metricLabelMap[key] || key;
    const preferredMetrics = ['overall', 'image_overall', 'video_overall', 'object_abs_distance_MRA:.5:.95:.05', 'object_counting_MRA:.5:.95:.05', 'object_rel_direction_accuracy', 'object_rel_distance_accuracy', 'object_size_estimation_MRA:.5:.95:.05', 'route_planning_accuracy', 'obj_appearance_order_accuracy'];
    const preferredSamplingValue = values => values.includes('fps_1') ? 'fps_1' : (values[0] || '');
    const activeRuns = () => runs.filter(r => !deletedRunIds.has(r.run_id));
    const persistDeletedRunIds = () => {{
      try {{
        if (typeof localStorage === 'undefined') return;
        localStorage.setItem(deletedRunIdsStorageKey, JSON.stringify([...deletedRunIds]));
      }} catch {{}}
    }};

    function initFilters() {{
      const models = [...new Set(activeRuns().map(r => r.model).filter(Boolean))].sort();
      const samplings = [...new Set(activeRuns().map(samplingLabel))].sort();
      const modelFilter = document.getElementById('modelFilter');
      const samplingFilter = document.getElementById('samplingFilter');
      const defaultSampling = preferredSamplingValue(samplings);
      modelFilter.innerHTML += models.map(m => `<option value="${{html(m)}}">${{html(m)}}</option>`).join('');
      samplingFilter.innerHTML += samplings.map(s => `<option value="${{html(s)}}">${{html(s)}}</option>`).join('');
      samplingFilter.value = defaultSampling;
      modelFilter.addEventListener('change', () => {{
        enforceFilterState(defaultSampling);
        updateFilterState();
        render();
      }});
      samplingFilter.addEventListener('change', () => {{
        enforceFilterState(defaultSampling);
        updateFilterState();
        render();
      }});
      document.getElementById('search').addEventListener('input', render);
      enforceFilterState(defaultSampling);
      updateFilterState();
    }}

    function enforceFilterState(defaultSampling) {{
      const modelFilter = document.getElementById('modelFilter');
      const samplingFilter = document.getElementById('samplingFilter');
      if (!modelFilter.value && !samplingFilter.value && defaultSampling) samplingFilter.value = defaultSampling;
    }}

    function updateFilterState() {{
      const modelFilter = document.getElementById('modelFilter');
      const samplingFilter = document.getElementById('samplingFilter');
      modelFilter.options[0].disabled = !samplingFilter.value;
      samplingFilter.options[0].disabled = !modelFilter.value;
    }}

    function latestPerModelSampling(items) {{
      const seen = new Map();
      for (const r of items) {{
        const key = `${{r.model}}||${{samplingLabel(r)}}`;
        if (!seen.has(key) || String(r.timestamp).localeCompare(String(seen.get(key).timestamp)) > 0) seen.set(key, r);
      }}
      return [...seen.values()].sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));
    }}

    function filteredRuns() {{
      const q = document.getElementById('search').value.toLowerCase();
      const m = document.getElementById('modelFilter').value;
      const s = document.getElementById('samplingFilter').value;
      return activeRuns().filter(r => (!m || r.model === m) && (!s || samplingLabel(r) === s) && (!q || JSON.stringify(r).toLowerCase().includes(q)));
    }}

    function renderBarChart(svgId, items) {{
      const svg = document.getElementById(svgId);
      svg.innerHTML = '';
      const width = svg.clientWidth || 1100, height = svg.clientHeight || 500;
      const padLeft = 58, padRight = 28, padTop = 84, padBottom = 78;
      const chartItems = latestPerModelSampling(items).slice(0, 12);
      if (!chartItems.length) {{
        svg.insertAdjacentHTML('beforeend', `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle" font-size="13" fill="#667085">暂无记录</text>`);
        return;
      }}

      const metricSet = new Set(chartItems.flatMap(r => Object.keys(r.metrics || {{}})));
      const metricKeys = [...preferredMetrics.filter(k => metricSet.has(k)), ...[...metricSet].filter(k => !preferredMetrics.includes(k)).sort()].slice(0, 12);
      if (!metricKeys.length) {{
        svg.insertAdjacentHTML('beforeend', `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle" font-size="13" fill="#667085">暂无可展示指标</text>`);
        return;
      }}
      const plotW = width - padLeft - padRight;
      const plotH = height - padTop - padBottom;
      const groupW = plotW / Math.max(1, metricKeys.length);
      const barW = Math.max(5, Math.min(30, groupW / (chartItems.length + 1)));
      const visibleBaselineKinds = [...new Set(metricKeys.flatMap(key => (baselines[key] || []).map(item => item.kind)))];

      [0, 20, 40, 60, 80, 100].forEach(score => {{
        const y = padTop + plotH * (1 - score / 100);
        const stroke = score === 0 ? '#d7dce3' : '#eef1f5';
        svg.insertAdjacentHTML('beforeend', `<line x1="${{padLeft}}" y1="${{y}}" x2="${{width-padRight}}" y2="${{y}}" stroke="${{stroke}}"/>`);
        svg.insertAdjacentHTML('beforeend', `<text x="${{padLeft - 12}}" y="${{y + 4}}" text-anchor="end" font-size="11" fill="#667085">${{score}}</text>`);
      }});
      svg.insertAdjacentHTML('beforeend', `<line x1="${{padLeft}}" y1="${{height-padBottom}}" x2="${{width-padRight}}" y2="${{height-padBottom}}" stroke="#d7dce3"/>`);
      svg.insertAdjacentHTML('beforeend', `<line x1="${{padLeft}}" y1="${{padTop}}" x2="${{padLeft}}" y2="${{height-padBottom}}" stroke="#d7dce3"/>`);

      chartItems.forEach((r, seriesIndex) => {{
        const color = palette[seriesIndex % palette.length];
        metricKeys.forEach((key, metricIndex) => {{
          const rawVal = metric(r, key);
          const val = Math.max(0, Math.min(100, Number(rawVal || 0)));
          const h = plotH * val / 100;
          const baseX = padLeft + metricIndex * groupW + (groupW - barW * chartItems.length) / 2;
          const x = baseX + seriesIndex * barW;
          const y = height - padBottom - h;
          const safeBarW = Math.max(2, barW - 2);
          svg.insertAdjacentHTML('beforeend', `<rect x="${{x}}" y="${{y}}" width="${{safeBarW}}" height="${{h}}" fill="${{color}}"><title>${{html(seriesLabel(r))}} ${{html(metricLabel(key))}}: ${{fmtScore(rawVal)}}</title></rect>`);
          if (rawVal !== null) {{
            const labelY = Math.max(padTop + 13, y - 6);
            svg.insertAdjacentHTML('beforeend', `<text x="${{x + safeBarW / 2}}" y="${{labelY}}" text-anchor="middle" font-size="11" fill="#344054">${{html(fmtScore(rawVal))}}</text>`);
          }}
        }});
      }});

      metricKeys.forEach((key, metricIndex) => {{
        const groupBaselines = baselines[key] || [];
        groupBaselines.forEach((baseline, baselineIndex) => {{
          if (typeof baseline.value !== 'number') return;
          const style = baselineStyles[baseline.kind] || baselineStyles.default;
          const val = Math.max(0, Math.min(100, Number(baseline.value)));
          const y = padTop + plotH * (1 - val / 100);
          const x1 = padLeft + metricIndex * groupW + Math.max(6, groupW * 0.08);
          const x2 = padLeft + (metricIndex + 1) * groupW - Math.max(6, groupW * 0.08);
          const label = baseline.label || style.label;
          const title = `${{metricLabel(key)}} ${{label}}: ${{fmtScore(baseline.value)}}`;
          svg.insertAdjacentHTML('beforeend', `<line x1="${{x1}}" y1="${{y}}" x2="${{x2}}" y2="${{y}}" stroke="${{style.color}}" stroke-width="${{style.width}}" stroke-dasharray="${{style.dash}}"><title>${{html(title)}}</title></line>`);
          if (groupW > 100) {{
            const labelY = Math.max(padTop + 12, y - 5 - baselineIndex * 12);
            svg.insertAdjacentHTML('beforeend', `<text x="${{x2}}" y="${{labelY}}" text-anchor="end" font-size="10" fill="${{style.color}}">${{html(fmtScore(baseline.value))}}</text>`);
          }}
        }});
      }});

      metricKeys.forEach((key, i) => {{
        const label = metricLabel(key);
        const x = padLeft + i * groupW + groupW / 2;
        svg.insertAdjacentHTML('beforeend', `<text x="${{x}}" y="${{height - 40}}" text-anchor="middle" font-size="12" fill="#475467">${{html(label)}}<title>${{html(key)}}</title></text>`);
      }});

      visibleBaselineKinds.forEach((kind, i) => {{
        const style = baselineStyles[kind] || baselineStyles.default;
        const y = 22 + i * 18;
        svg.insertAdjacentHTML('beforeend', `<line x1="${{padLeft}}" y1="${{y}}" x2="${{padLeft + 24}}" y2="${{y}}" stroke="${{style.color}}" stroke-width="${{style.width}}" stroke-dasharray="${{style.dash}}" />`);
        svg.insertAdjacentHTML('beforeend', `<text x="${{padLeft + 32}}" y="${{y + 4}}" font-size="12" fill="#1f2937">${{html(style.label)}}</text>`);
      }});

      chartItems.forEach((r, i) => {{
        const color = palette[i % palette.length];
        const y = 22 + i * 20;
        const legendX = Math.max(padLeft, width - 360);
        svg.insertAdjacentHTML('beforeend', `<line x1="${{legendX}}" y1="${{y}}" x2="${{legendX + 24}}" y2="${{y}}" stroke="${{color}}" stroke-width="6" />`);
        svg.insertAdjacentHTML('beforeend', `<text x="${{legendX + 32}}" y="${{y + 4}}" font-size="12" fill="#1f2937">${{html(seriesLabel(r))}}  Token: ${{html(fmt(token(r, 'total_tokens')))}} </text>`);
      }});
    }}

    function renderTable(items) {{
      const metricKeys = [...new Set(items.flatMap(r => Object.keys(r.metrics || {{}})))].filter(k => !['tabulated_keys','tabulated_results'].includes(k));
      const preferred = preferredMetrics;
      const keys = [...preferred.filter(k => metricKeys.includes(k)), ...metricKeys.filter(k => !preferred.includes(k)).sort()];
      const visibleMetricKeys = keys.slice(0, 10);
      const head = ['时间', '模型', '采样', 'Token'];
      const header = head.map(h => `<th>${{h}}</th>`).join('') + visibleMetricKeys.map(k => `<th title="${{html(k)}}">${{html(metricLabel(k))}}</th>`).join('') + '<th>操作</th>';
      const rows = items.map(r => `<tr>
        <td>${{html(r.timestamp)}}</td><td>${{html(r.model)}}</td><td><span class="pill">${{html(samplingLabel(r))}}</span></td>
        <td>${{html(fmt(token(r, 'total_tokens')))}} </td>
        ${{visibleMetricKeys.map(k => `<td>${{html(fmt(metric(r, k)))}}</td>`).join('')}}
        <td><button class="danger-btn" type="button" data-delete-run="${{html(r.run_id)}}">删除</button></td>
      </tr>`).join('');
      document.getElementById('runsTable').innerHTML = items.length ? `<thead><tr>${{header}}</tr></thead><tbody>${{rows}}</tbody>` : `<tbody><tr><td class="empty" colspan="${{head.length + visibleMetricKeys.length + 1}}">暂无记录</td></tr></tbody>`;
    }}

    function render() {{
      renderBarChart('metricChart', filteredRuns());
      renderTable(activeRuns());
    }}

    function initDeleteDialog() {{
      const table = document.getElementById('runsTable');
      const modal = document.getElementById('deleteModal');
      const target = document.getElementById('deleteTarget');
      const cancelButton = document.getElementById('cancelDelete');
      const confirmButton = document.getElementById('confirmDelete');

      table.addEventListener('click', event => {{
        const button = event.target.closest('[data-delete-run]');
        if (!button) return;
        pendingDeleteRunId = button.getAttribute('data-delete-run');
        const run = activeRuns().find(item => item.run_id === pendingDeleteRunId);
        target.textContent = run ? `确认删除 ${{run.timestamp}} / ${{run.model}} / ${{samplingLabel(run)}} 这条评估记录？` : '确认删除这条评估记录？';
        modal.hidden = false;
        confirmButton.focus();
      }});

      function closeModal() {{
        pendingDeleteRunId = null;
        modal.hidden = true;
      }}

      cancelButton.addEventListener('click', closeModal);
      modal.addEventListener('click', event => {{
        if (event.target === modal) closeModal();
      }});
      confirmButton.addEventListener('click', () => {{
        if (pendingDeleteRunId) {{
          deletedRunIds.add(pendingDeleteRunId);
          persistDeletedRunIds();
        }}
        closeModal();
        render();
      }});
      document.addEventListener('keydown', event => {{
        if (event.key === 'Escape' && !modal.hidden) closeModal();
      }});
    }}

    initFilters();
    initDeleteDialog();
    render();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Update static evaluation dashboard from an lmms-eval results.json file.")
    parser.add_argument("result_file", nargs="?", help="Path to results.json")
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--html_path", default=str(DEFAULT_HTML_PATH))
    parser.add_argument("--baseline_excel", default=None, help="Optional Excel question bank used to calculate baseline reference lines.")
    args = parser.parse_args()
    if args.result_file:
        result = update_dashboard_from_result_file(args.result_file, args.data_path, args.html_path, args.baseline_excel)
    else:
        result = refresh_dashboard(args.data_path, args.html_path, args.baseline_excel)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
