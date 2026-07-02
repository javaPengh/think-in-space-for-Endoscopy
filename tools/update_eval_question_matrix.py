import argparse
import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_DATA_PATH = DOCS_DIR / "eval_question_matrix.json"
DEFAULT_HTML_PATH = DOCS_DIR / "eval_question_matrix.html"
MRA_METRIC = "MRA:.5:.95:.05"
SOURCE_ID_KEYS = ("question_id", "id", "ID", "Question_ID", "questionId")
NUMERIC_QUESTION_TYPES = {
    "counting(object)",
    "counting(action)",
    "polyp_size_estimation(ref)",
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
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
    rows = [row for row in matrix_data.get("rows", []) if row.get("sample_path") != sample_path_text]
    rows.extend(run_rows)
    rows.sort(key=lambda row: (str(row.get("timestamp", "")), str(row.get("data_version", "default")), str(row.get("model", "")), str(row.get("sampling_strategy", "")), _sortable_doc_id(row.get("doc_id"))))

    matrix_data = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(matrix_data, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_question_matrix_html(matrix_data), encoding="utf-8")
    return {"data_path": str(data_path), "html_path": str(html_path), "row_count": len(run_rows)}


def refresh_question_matrix_html(data_path=None, html_path=None):
    data_path = Path(data_path or DEFAULT_DATA_PATH).expanduser().resolve()
    html_path = Path(html_path or DEFAULT_HTML_PATH).expanduser().resolve()
    matrix_data = _read_matrix_data(data_path)
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
        source_doc = score_doc if score_doc else doc
        eval_doc_id = log.get("doc_id")
        doc_id = _source_doc_id(source_doc, doc, eval_doc_id)
        answer_type = source_doc.get("answer_type") or _infer_answer_type(source_doc.get("question_type"))
        score = _score_percent(score_doc, answer_type)
        is_correct = _choice_is_correct(score_doc, answer_type)
        is_scored = _numeric_is_scored(score, answer_type)
        natural_prediction = score_doc.get("natural_prediction")
        if natural_prediction is None:
            natural_prediction = _first_scalar(log.get("resps"))

        restricted_prediction = score_doc.get("restricted_prediction")
        extraction_was_attempted = score_doc and ("restricted_prediction" in score_doc or "prediction" in score_doc)
        if restricted_prediction is None:
            restricted_prediction = score_doc.get("prediction")
        if restricted_prediction is None and extraction_was_attempted:
            restricted_prediction = "提取失败"
        if restricted_prediction is None:
            restricted_prediction = _first_scalar(log.get("filtered_resps"))
        row_id_seed = f"{sample_path_text}|{doc_id}"
        rows.append(
            {
                "row_id": hashlib.sha256(row_id_seed.encode("utf-8")).hexdigest()[:12],
                "timestamp": timestamp,
                "model": model,
                "data_version": data_version,
                "sampling_strategy": sampling_strategy,
                "note": run_note,
                "question_type": source_doc.get("question_type", ""),
                "answer_type": answer_type or "",
                "doc_id": doc_id,
                "question": source_doc.get("question", ""),
                "natural_prediction": natural_prediction if natural_prediction is not None else "",
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
    rows = data.get("rows", [])
    generated_at = html.escape(str(data.get("updated_at", "")))
    data_version_options = _select_options(sorted({row.get("data_version", "default") for row in rows}))
    model_options = _select_options(sorted({row.get("model", "") for row in rows if row.get("model")}))
    sampling_options = _select_options(sorted({row.get("sampling_strategy", "") for row in rows if row.get("sampling_strategy")}))
    matrix_rows_json = _json_for_script(rows)
    row_count = len(rows)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VSI-Bench 题目级对错矩阵</title>
  <style>
    :root {{
      color-scheme: light;
      --border: #d8dee8;
      --header: #eef3fb;
      --text: #162033;
      --muted: #5c6b82;
      --accent: #2563eb;
      --ok: #047857;
      --bad: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: var(--text);
      background: #f6f8fb;
    }}
    header {{
      padding: 24px 28px 14px;
      background: #fff;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      padding: 18px 28px 28px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
    }}
    input, select {{
      height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      padding: 0 10px;
      color: var(--text);
      font-size: 13px;
    }}
    input {{
      min-width: 260px;
      flex: 1 1 300px;
    }}
    select {{
      min-width: 160px;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
    }}
    table {{
      width: 100%;
      min-width: 1400px;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 9px;
      vertical-align: top;
      font-size: 13px;
      line-height: 1.45;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--header);
      text-align: left;
      white-space: nowrap;
      color: #23314a;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .nowrap {{
      white-space: nowrap;
    }}
    .text {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .status {{
      display: inline-block;
      min-width: 52px;
      font-weight: 700;
    }}
    .status.ok {{ color: var(--ok); }}
    .status.bad {{ color: var(--bad); }}
    .score {{
      color: var(--accent);
      font-weight: 700;
    }}
    .empty {{
      padding: 32px;
      text-align: center;
      color: var(--muted);
    }}
    .pager {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pager-controls {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    button {{
      height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 0 12px;
      cursor: pointer;
      font-size: 13px;
    }}
    button:disabled {{
      color: #98a2b3;
      cursor: not-allowed;
    }}
  </style>
</head>
<body>
  <header>
    <h1>VSI-Bench 题目级对错矩阵</h1>
    <div class="meta">共 <span id="visible-count">{row_count}</span> / {row_count} 条记录，更新时间：{generated_at}</div>
  </header>
  <main>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索题号、模型、题目、输出、真实答案">
      <select id="data-version-filter">
        <option value="">全部数据版本</option>
        {data_version_options}
      </select>
      <select id="model-filter">
        <option value="">全部模型</option>
        {model_options}
      </select>
      <select id="sampling-filter">
        <option value="">全部采样策略</option>
        {sampling_options}
      </select>
    </div>
    <div class="pager">
      <div>当前页 <span id="page-info">0 / 0</span></div>
      <div class="pager-controls">
        <label for="page-size">每页</label>
        <select id="page-size">
          <option value="50">50</option>
          <option value="100" selected>100</option>
          <option value="200">200</option>
          <option value="500">500</option>
        </select>
        <button id="prev-page" type="button">上一页</button>
        <button id="next-page" type="button">下一页</button>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <colgroup>
          <col style="width: 112px">
          <col style="width: 120px">
          <col style="width: 145px">
          <col style="width: 100px">
          <col style="width: 180px">
          <col style="width: 80px">
          <col style="width: 150px">
          <col style="width: 86px">
          <col style="width: 92px">
          <col style="width: 68px">
          <col style="width: 260px">
          <col style="width: 300px">
          <col style="width: 120px">
          <col style="width: 110px">
          <col style="width: 220px">
        </colgroup>
        <thead>
          <tr>
            <th>时间</th>
            <th>数据版本</th>
            <th>模型</th>
            <th>采样策略</th>
            <th>备注</th>
            <th>题号</th>
            <th>题型</th>
            <th>答案类型</th>
            <th>是否正确/得分</th>
            <th>分数</th>
            <th>原题目</th>
            <th>自然输出</th>
            <th>受限输出</th>
            <th>真实答案</th>
            <th>选项内容</th>
          </tr>
        </thead>
        <tbody id="matrix-body"></tbody>
      </table>
      <div id="empty" class="empty" hidden>没有匹配记录</div>
    </div>
  </main>
  <script id="matrix-data" type="application/json">{matrix_rows_json}</script>
  <script>
    const allRows = JSON.parse(document.getElementById("matrix-data").textContent || "[]");
    let filteredRows = allRows;
    let currentPage = 1;
    let filterTimer = null;

    function rowSearchText(row) {{
      if (row._searchText !== undefined) return row._searchText;
      row._searchText = [
        row.timestamp,
        row.data_version,
        row.model,
        row.sampling_strategy,
        row.note,
        row.doc_id,
        row.question_type,
        row.answer_type,
        row.question,
        row.natural_prediction,
        row.restricted_prediction,
        row.ground_truth,
        row.options
      ].map((value) => String(value ?? "")).join("\\n").toLowerCase()
      return row._searchText;
    }}

    const matrixBody = document.getElementById("matrix-body");
    const search = document.getElementById("search");
    const dataVersionFilter = document.getElementById("data-version-filter");
    const modelFilter = document.getElementById("model-filter");
    const samplingFilter = document.getElementById("sampling-filter");
    const pageSize = document.getElementById("page-size");
    const pageInfo = document.getElementById("page-info");
    const prevPage = document.getElementById("prev-page");
    const nextPage = document.getElementById("next-page");
    const visibleCount = document.getElementById("visible-count");
    const empty = document.getElementById("empty");

    function html(value) {{
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));
    }}

    function scoreText(row) {{
      return row.score === null || row.score === undefined || row.score === "" ? "" : Number(row.score).toFixed(2);
    }}

    function status(row) {{
      if (row.is_correct === true) return ["正确", "ok"];
      if (row.is_correct === false) return ["错误", "bad"];
      if (row.is_scored === true) return ["有得分", "ok"];
      if (row.is_scored === false) return ["无得分", "bad"];
      return ["", ""];
    }}

    function renderRow(row) {{
      const [statusText, statusClass] = status(row);
      return `<tr>
  <td class="nowrap">${{html(row.timestamp)}}</td>
  <td class="nowrap">${{html(row.data_version || "default")}}</td>
  <td class="nowrap">${{html(row.model)}}</td>
  <td class="nowrap">${{html(row.sampling_strategy)}}</td>
  <td class="text">${{html(row.note)}}</td>
  <td class="nowrap">${{html(row.doc_id)}}</td>
  <td>${{html(row.question_type)}}</td>
  <td class="nowrap">${{html(row.answer_type)}}</td>
  <td class="nowrap"><span class="status ${{statusClass}}">${{html(statusText)}}</span></td>
  <td class="nowrap score">${{html(scoreText(row))}}</td>
  <td class="text">${{html(row.question)}}</td>
  <td class="text">${{html(row.natural_prediction)}}</td>
  <td class="text">${{html(row.restricted_prediction)}}</td>
  <td class="text">${{html(row.ground_truth)}}</td>
  <td class="text">${{html(row.options)}}</td>
</tr>`;
    }}

    function applyFilters() {{
      const keyword = search.value.trim().toLowerCase();
      const dataVersion = dataVersionFilter.value;
      const model = modelFilter.value;
      const sampling = samplingFilter.value;
      filteredRows = allRows.filter((row) => {{
        const matchesDataVersion = !dataVersion || row.data_version === dataVersion;
        const matchesModel = !model || row.model === model;
        const matchesSampling = !sampling || row.sampling_strategy === sampling;
        const matchesKeyword = !keyword || rowSearchText(row).includes(keyword);
        return matchesDataVersion && matchesModel && matchesSampling && matchesKeyword;
      }});
      currentPage = 1;
      renderPage();
    }}

    function renderPage() {{
      const perPage = Number(pageSize.value) || 100;
      const total = filteredRows.length;
      const totalPages = Math.max(1, Math.ceil(total / perPage));
      currentPage = Math.min(Math.max(currentPage, 1), totalPages);
      const start = (currentPage - 1) * perPage;
      const pageRows = filteredRows.slice(start, start + perPage);
      matrixBody.innerHTML = pageRows.map(renderRow).join("");
      visibleCount.textContent = total;
      pageInfo.textContent = total ? `${{start + 1}}-${{Math.min(start + perPage, total)}} / ${{total}}` : "0 / 0";
      prevPage.disabled = currentPage <= 1;
      nextPage.disabled = currentPage >= totalPages;
      empty.hidden = total !== 0;
    }}

    function scheduleFilters() {{
      clearTimeout(filterTimer);
      filterTimer = setTimeout(applyFilters, 120);
    }}

    search.addEventListener("input", scheduleFilters);
    dataVersionFilter.addEventListener("change", applyFilters);
    modelFilter.addEventListener("change", applyFilters);
    samplingFilter.addEventListener("change", applyFilters);
    pageSize.addEventListener("change", () => {{
      currentPage = 1;
      renderPage();
    }});
    prevPage.addEventListener("click", () => {{
      currentPage -= 1;
      renderPage();
    }});
    nextPage.addEventListener("click", () => {{
      currentPage += 1;
      renderPage();
    }});
    renderPage();
  </script>
</body>
</html>
"""


def _render_row(row):
    status_text, status_class = _status_label(row)
    score = row.get("score")
    score_text = "" if score is None else f"{float(score):.2f}"
    return f"""<tr data-data-version="{_attr(row.get('data_version', 'default'))}" data-model="{_attr(row.get('model'))}" data-sampling="{_attr(row.get('sampling_strategy'))}">
  <td class="nowrap">{_cell(row.get('timestamp'))}</td>
  <td class="nowrap">{_cell(row.get('data_version', 'default'))}</td>
  <td class="nowrap">{_cell(row.get('model'))}</td>
  <td class="nowrap">{_cell(row.get('sampling_strategy'))}</td>
  <td class="text">{_cell(row.get('note'))}</td>
  <td class="nowrap">{_cell(row.get('doc_id'))}</td>
  <td>{_cell(row.get('question_type'))}</td>
  <td class="nowrap">{_cell(row.get('answer_type'))}</td>
  <td class="nowrap"><span class="status {status_class}">{html.escape(status_text)}</span></td>
  <td class="nowrap score">{html.escape(score_text)}</td>
  <td class="text">{_cell(row.get('question'))}</td>
  <td class="text">{_cell(row.get('natural_prediction'))}</td>
  <td class="text">{_cell(row.get('restricted_prediction'))}</td>
  <td class="text">{_cell(row.get('ground_truth'))}</td>
  <td class="text">{_cell(row.get('options'))}</td>
</tr>"""


def _status_label(row):
    if row.get("is_correct") is True:
        return "正确", "ok"
    if row.get("is_correct") is False:
        return "错误", "bad"
    if row.get("is_scored") is True:
        return "有得分", "ok"
    if row.get("is_scored") is False:
        return "无得分", "bad"
    return "", ""


def _select_options(values):
    return "\n".join(f'<option value="{_attr(value)}">{_cell(value)}</option>' for value in values)


def _cell(value):
    return html.escape("" if value is None else str(value))


def _attr(value):
    return html.escape("" if value is None else str(value), quote=True)


def _json_for_script(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _read_matrix_data(data_path):
    if not data_path.exists():
        return {"rows": []}
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"rows": data}
    if "rows" not in data:
        data["rows"] = []
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
        for token in ["_natural", "_blind", "_file"]:
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
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description="Build VSI-Bench question-level answer matrix from a sample JSON file.")
    parser.add_argument("sample_file", nargs="?", help="Path to a task sample JSON file, e.g. logs/.../vsibench.json")
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH), help="Output JSON path.")
    parser.add_argument("--html_path", default=str(DEFAULT_HTML_PATH), help="Output HTML path.")
    parser.add_argument("--refresh_only", action="store_true", help="Regenerate HTML from the existing matrix JSON without reading a sample file.")
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
