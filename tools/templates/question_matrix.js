"use strict";

const choiceTypes = new Set(["mca", "mac", "mcq", "multiple_choice", "choice"]);
const numberTypes = new Set(["numeric", "numerical", "number"]);
const isChoice = row => choiceTypes.has(String(row.answer_type || "").toLowerCase());
const isNumeric = row => numberTypes.has(String(row.answer_type || "").toLowerCase());
const formatCount = value => Number(value).toLocaleString("zh-CN");
const modeText = row => row.answer_mode === "natural" ? "首轮自然回答" : "直接回答";
const runKey = row => row.sample_path || JSON.stringify([row.timestamp, row.model, row.sampling_strategy, row.answer_mode, row.note]);
const questionKey = row => JSON.stringify([row.data_version, String(row.doc_id), row.question, row.options, row.ground_truth]);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[ch]));
}

function scoreValue(row) {
  return row.score === null || row.score === undefined || row.score === "" || !Number.isFinite(Number(row.score)) ? null : Number(row.score);
}

function resultKind(row) {
  if (isChoice(row)) return row.is_correct === true ? "correct" : row.is_correct === false ? "incorrect" : "unscored";
  const score = scoreValue(row);
  if (!isNumeric(row) || score === null) return "unscored";
  return score === 0 ? "zero" : score >= 100 ? "full" : "partial";
}

function resultLabel(row) {
  return {
    correct: ["正确", "ok"], incorrect: ["错误", "bad"], zero: ["MRA 0", "bad"],
    full: ["MRA 100", "ok"], partial: [`MRA ${scoreValue(row)?.toFixed(1)}`, "partial"], unscored: ["未评分", "neutral"]
  }[resultKind(row)];
}

function explanationState(row) {
  const records = row.answer_explanations || [];
  if (records.some(record => String(record.explanation ?? "").trim())) return "available";
  return records.length ? "empty" : "missing";
}

function parseTimestamp(value) {
  const normalized = String(value ?? "").trim().replace(/T(\d{2})-(\d{2})-(\d{2})/, "T$1:$2:$3");
  const parsed = Date.parse(normalized);
  return Number.isNaN(parsed) ? null : parsed;
}

function rowTime(row) {
  const parsed = parseTimestamp(row.timestamp);
  if (parsed !== null) return parsed;
  // Historical CLI filenames use MMDD_HHmm, with the year in logs/YYYYMMDD.
  const date = String(row.sample_path || "").match(/logs[\/\\](\d{4})(\d{2})(\d{2})[\/\\]/);
  const time = String(row.timestamp || "").match(/^\d{4}_(\d{2})(\d{2})$/);
  return date && time ? parseTimestamp(`${date[1]}-${date[2]}-${date[3]}T${time[1]}:${time[2]}:00`) : null;
}

function matchesFilters(row, filters, searchText = null) {
  for (const [field, value] of [["data_version", filters.version], ["model", filters.model], ["sampling_strategy", filters.sampling], ["question_type", filters.type], ["answer_mode", filters.mode]]) {
    if (value && row[field] !== value) return false;
  }
  const kind = resultKind(row);
  if (filters.status === "attention" && !["incorrect", "zero"].includes(kind)) return false;
  if (filters.status && filters.status !== "attention" && filters.status !== kind) return false;
  if (filters.explanation && filters.explanation !== explanationState(row)) return false;
  if (filters.start !== null && filters.start !== undefined || filters.end !== null && filters.end !== undefined) {
    const time = rowTime(row);
    if (time === null || filters.start != null && time < filters.start || filters.end != null && time > filters.end) return false;
  }
  if (filters.keyword) {
    const text = searchText ? searchText(row) : JSON.stringify(row).toLowerCase();
    if (!text.includes(filters.keyword.toLowerCase())) return false;
  }
  return true;
}

function statistics(rows) {
  let choices = 0, correct = 0, numeric = 0, numericSum = 0, attention = 0, explained = 0, empty = 0;
  const runs = new Set(), questions = new Set();
  for (const row of rows) {
    const kind = resultKind(row);
    if (["correct", "incorrect"].includes(kind)) { choices++; if (kind === "correct") correct++; }
    if (isNumeric(row) && scoreValue(row) !== null) { numeric++; numericSum += scoreValue(row); }
    if (["incorrect", "zero"].includes(kind)) attention++;
    if (explanationState(row) === "available") explained++;
    if (explanationState(row) === "empty") empty++;
    runs.add(runKey(row)); questions.add(questionKey(row));
  }
  return {total: rows.length, choices, correct, numeric, numericSum, attention, explained, empty, runs: runs.size, questions: questions.size};
}

function groupMatrix(rows) {
  const runs = new Map(), questions = new Map();
  for (const row of rows) {
    const run = runKey(row), question = questionKey(row);
    if (!runs.has(run)) runs.set(run, row);
    if (!questions.has(question)) questions.set(question, {row, cells: new Map()});
    const cells = questions.get(question).cells;
    if (!cells.has(run)) cells.set(run, []);
    cells.get(run).push(row); // Retain duplicate source IDs instead of silently overwriting.
  }
  return {runs: [...runs.entries()], questions: [...questions.values()]};
}

function compareRows(a, b, sort) {
  const collate = (x, y) => String(x ?? "").localeCompare(String(y ?? ""), "zh-CN", {numeric: true});
  if (sort === "score") return (scoreValue(a) ?? Infinity) - (scoreValue(b) ?? Infinity) || collate(a.doc_id, b.doc_id);
  if (sort === "model") return collate(a.model, b.model) || collate(a.doc_id, b.doc_id);
  if (sort === "question") return collate(a.data_version, b.data_version) || collate(a.doc_id, b.doc_id);
  return (rowTime(b) ?? -Infinity) - (rowTime(a) ?? -Infinity) || collate(b.timestamp, a.timestamp) || collate(a.doc_id, b.doc_id);
}

function csvCell(value) {
  let text = String(value ?? "");
  if (/^\s*[=+@-]/.test(text)) text = "'" + text;
  return '"' + text.replace(/"/g, '""') + '"';
}

function exportCsv(rows) {
  const columns = [
    ["题号", "doc_id"], ["数据版本", "data_version"], ["模型", "model"], ["采样策略", "sampling_strategy"], ["时间", "timestamp"],
    ["首轮模式", "answer_mode"], ["题型", "question_type"], ["问题", "question"], ["选项", "options"], ["首次原始输出", "first_turn_prediction"],
    ["评分用答案", "restricted_prediction"], ["标准答案", "ground_truth"], ["首轮评分(0-100)", "score"], ["第二轮解释", "natural_prediction"],
    ["解释状态", "explanation_status"], ["所有第二轮记录", "answer_explanations"], ["备注", "note"], ["源日志", "sample_path"]
  ];
  return "\uFEFF" + [columns.map(([label]) => csvCell(label)).join(","), ...rows.map(row => columns.map(([, key]) => csvCell(key === "answer_explanations" ? JSON.stringify(row[key] || []) : row[key])).join(","))].join("\r\n");
}

function initQuestionMatrix() {
  const $ = id => document.getElementById(id);
  const payload = JSON.parse($("matrix-data").textContent);
  const allRows = payload.rows || [];
  allRows.forEach((row, index) => { row._index = index; });
  let filteredRows = [], matrix = null, page = 1, view = "list", selectedIndex = null, timer;
  const searchCache = new Map();
  const filterIds = ["version-filter", "model-filter", "type-filter", "status-filter", "explanation-filter", "sampling-filter", "mode-filter", "time-start", "time-end"];
  const h = escapeHtml;
  const text = (id, value) => { $(id).textContent = value; };
  const pill = row => { const [label, color] = resultLabel(row); return `<span class="pill ${color}">${h(label)}</span>`; };
  const answer = value => value === null || value === undefined || value === "" ? "—" : String(value);
  const firstExplanation = row => (row.answer_explanations || []).find(record => String(record.explanation || "").trim())?.explanation || "";
  const searchable = row => {
    if (!searchCache.has(row._index)) searchCache.set(row._index, [row.doc_id, row.model, row.timestamp, row.data_version, row.sampling_strategy, row.question_type, row.question, row.note, row.first_turn_prediction, row.restricted_prediction, row.ground_truth, row.options, ...(row.answer_explanations || []).map(record => record.explanation)].join("\n").toLowerCase());
    return searchCache.get(row._index);
  };
  for (const [id, field] of [["version-filter", "data_version"], ["model-filter", "model"], ["type-filter", "question_type"], ["sampling-filter", "sampling_strategy"]]) {
    $(id).innerHTML += [...new Set(allRows.map(row => row[field]).filter(Boolean))].sort().map(value => `<option value="${h(value)}">${h(value)}</option>`).join("");
  }
  text("updated-at", `数据更新：${payload.updated_at || "未知"}`);
  text("dataset-summary", `${formatCount(allRows.length)} 条历史记录 · 每次运行独立保留`);

  function renderStats() {
    const s = statistics(filteredRows);
    text("stat-records", formatCount(s.total));
    text("stat-scope", `${formatCount(s.questions)} 道题 · ${s.runs} 次运行`);
    text("stat-accuracy", s.choices ? `${(s.correct / s.choices * 100).toFixed(1)}%` : "—");
    text("stat-choice-count", `${formatCount(s.correct)} / ${formatCount(s.choices)} 条已评分选择题记录`);
    text("stat-mra", s.numeric ? (s.numericSum / s.numeric).toFixed(1) : "—");
    text("stat-numeric-count", `${formatCount(s.numeric)} 条数值题记录 · 满分 100`);
    text("stat-errors", formatCount(s.attention));
    text("stat-coverage", s.total ? `${(s.explained / s.total * 100).toFixed(1)}%` : "—");
    text("stat-explanations", `${formatCount(s.explained)} 条有解释 · ${formatCount(s.empty)} 条返回空解释`);
  }

  function applyFilters() {
    const filters = {version: $("version-filter").value, model: $("model-filter").value, type: $("type-filter").value, sampling: $("sampling-filter").value, mode: $("mode-filter").value, status: $("status-filter").value, explanation: $("explanation-filter").value, start: parseTimestamp($("time-start").value), end: parseTimestamp($("time-end").value), keyword: $("search").value.trim().toLowerCase()};
    filteredRows = allRows.filter(row => matchesFilters(row, filters, searchable)).sort((a, b) => compareRows(a, b, $("sort").value));
    matrix = null; page = 1; renderStats(); renderPage();
  }

  function questionCell(row) {
    return `<button type="button" class="question-link" data-row="${row._index}"><span class="question-id"># ${h(row.doc_id)}</span><span class="question-text">${h(answer(row.question))}</span></button><div class="secondary">${h(row.question_type || "未知题型")} · ${h(row.data_version)}</div>`;
  }

  function renderListRow(row) {
    const state = explanationState(row);
    const explain = state === "available" ? `<button type="button" class="explanation-link" data-row="${row._index}">阅读第二轮解释 ↗</button><div class="clamp explanation-preview">${h(firstExplanation(row).slice(0, 180))}</div>` : `<span class="pill neutral">${state === "empty" ? "返回空解释" : "未生成"}</span>`;
    return `<tr><td class="question-col">${questionCell(row)}</td><td class="run-col"><div class="model-name">${h(row.model)}</div><div class="secondary">${h(row.sampling_strategy)} · ${h(modeText(row))}</div><div class="secondary">${h(row.timestamp)}</div></td><td class="answer-col"><div class="answer-text clamp">${h(answer(row.restricted_prediction))}</div></td><td class="answer-col"><div class="answer-text clamp">${h(answer(row.ground_truth))}</div></td><td class="result-col">${pill(row)}<div class="score-line">${isChoice(row) ? "选择题" : isNumeric(row) ? "数值题 · MRA" : "未知类型"}</div></td><td class="explanation-col">${explain}</td><td><button type="button" class="open-button" data-row="${row._index}" aria-label="查看题目 ${h(row.doc_id)} 的详情">↗</button></td></tr>`;
  }

  function renderMatrixRow(group, runs) {
    return `<tr><td>${questionCell(group.row)}</td>${runs.map(([key]) => {
      const records = group.cells.get(key);
      if (!records) return '<td><span class="muted" title="此运行没有符合筛选条件的记录">—</span></td>';
      return `<td>${records.map(row => `<button class="matrix-cell" type="button" data-row="${row._index}" aria-label="查看 ${h(row.model)} 对题目 ${h(row.doc_id)} 的回答">${pill(row)}<span class="secondary">${explanationState(row) === "available" ? "有解释" : "无解释"}</span></button>`).join("")}</td>`;
    }).join("")}</tr>`;
  }

  function renderPage() {
    const size = Number($("page-size").value);
    if (view === "matrix" && !matrix) matrix = groupMatrix(filteredRows);
    const items = view === "matrix" ? matrix.questions : filteredRows;
    const pages = Math.max(1, Math.ceil(items.length / size));
    page = Math.max(1, Math.min(page, pages));
    const start = (page - 1) * size;
    $("results-table").className = view === "matrix" ? "matrix-table" : "list-table";
    if (view === "list") {
      $("results-head").innerHTML = '<tr><th>题目</th><th>模型 / 运行</th><th>首次答案（评分用）</th><th>标准答案</th><th>评分结果</th><th>自然输出 · 第二轮解释</th><th aria-label="操作"></th></tr>';
      $("results-body").innerHTML = items.slice(start, start + size).map(renderListRow).join("");
      text("view-note", "点击题目或右侧箭头查看原始输出、完整解释、选项和运行信息。统计按记录计算，包含重复评估运行。");
    } else {
      $("results-head").innerHTML = `<tr><th>题目 / 数据版本</th>${matrix.runs.map(([key, row], index) => `<th title="${h(key)}"><div class="model-name">${h(row.model)}</div><div class="secondary">${h(row.sampling_strategy)} · ${h(modeText(row))}</div><div class="secondary">${h(row.timestamp)} · 运行 ${index + 1}</div></th>`).join("")}</tr>`;
      $("results-body").innerHTML = items.slice(start, start + size).map(group => renderMatrixRow(group, matrix.runs)).join("");
      text("view-note", "每列对应一次运行；相同版本、题号、题目、选项及标准答案才对齐。横向滚动查看各模型，点击色块查看详情。“—”表示没有匹配记录，不代表答错。");
    }
    $("results-head").hidden = !items.length;
    $("empty").hidden = !!items.length;
    text("result-count", `${formatCount(filteredRows.length)} / ${formatCount(allRows.length)} 条记录${view === "matrix" ? ` · ${formatCount(items.length)} 道题 × ${matrix.runs.length} 次运行` : ""}`);
    text("page-info", items.length ? `${start + 1}–${Math.min(start + size, items.length)} / ${formatCount(items.length)} ${view === "matrix" ? "道题" : "条记录"} · 第 ${page} / ${pages} 页` : "0 条记录");
    $("prev").disabled = page <= 1; $("next").disabled = page >= pages; $("export").disabled = !filteredRows.length;
  }

  function explanationContent(row, index) {
    const record = (row.answer_explanations || [])[index];
    if (!record) return '<div class="muted prose">这条记录没有第二轮解释。需要在评估时开启 explain_answer，再导入对应日志。</div>';
    return `<div class="prompt">${h(record.prompt || "追问提示词未记录")}</div><div class="secondary">本轮被解释的答案</div><div class="prose">${h(answer(record.answer))}</div><hr style="border:0;border-top:1px solid var(--line);margin:15px 0"><div class="prose">${h(record.explanation || "模型返回了空解释。")}</div>`;
  }

  function showDetail(index) {
    const row = allRows[index];
    if (!row) return;
    selectedIndex = index;
    text("detail-title", `题目 #${row.doc_id}`);
    const records = row.answer_explanations || [];
    const pos = filteredRows.indexOf(row);
    $("detail-content").innerHTML = `<div class="detail-meta">${pill(row)}<span class="pill info">${h(row.model)}</span><span class="pill neutral">${h(row.data_version)}</span></div>
      <section class="detail-section"><h3>题目 · ${h(row.question_type)}</h3><div class="prose question-detail">${h(answer(row.question))}</div>${row.options ? `<div class="prose option-box">${h(row.options)}</div>` : ""}</section>
      <section class="detail-section answer-pair"><div class="answer-card"><h3>首次答案 · 评分用</h3><div class="prose">${h(answer(row.restricted_prediction))}</div></div><div class="answer-card truth"><h3>标准答案</h3><div class="prose">${h(answer(row.ground_truth))}</div></div></section>
      <section class="detail-section"><details><summary>查看第一轮原始输出 · ${h(modeText(row))}</summary><div class="prose answer-card">${h(answer(row.first_turn_prediction))}</div></details></section>
      <section class="detail-section"><div class="turn-label">TURN 02 / 回答后追问</div><h3>自然输出 · 第二轮解释</h3><div class="explanation-card">${records.length > 1 ? `<label for="repeat-selector">解释记录</label><select class="repeat-selector" id="repeat-selector">${records.map((record, i) => `<option value="${i}">请求 ${(record.request_index ?? 0) + 1} · 第 ${(record.repeat_index ?? i) + 1} 次生成</option>`).join("")}</select>` : ""}<div id="explanation-content">${explanationContent(row, 0)}</div></div><p class="detail-notice">第二轮解释不参与评分；解释可能包含事后合理化或改口，不等同于首次作答时的内部推理。</p></section>
      <section class="detail-section"><details><summary>运行信息与来源</summary><dl class="source-info"><dt>时间 / 采样策略</dt><dd>${h(row.timestamp)} · ${h(row.sampling_strategy)}</dd><dt>首轮模式 / 分数</dt><dd>${h(modeText(row))} · ${scoreValue(row) === null ? "未评分" : h(scoreValue(row))}</dd><dt>备注</dt><dd>${h(row.note || "无")}</dd><dt>源日志</dt><dd>${h(row.sample_path)}</dd><dt>评估内题号 / 记录 ID</dt><dd>${h(row.eval_doc_id ?? "未记录")} / ${h(row.row_id)}</dd></dl></details></section>`;
    if ($("repeat-selector")) $("repeat-selector").addEventListener("change", event => { $("explanation-content").innerHTML = explanationContent(row, Number(event.target.value)); });
    text("detail-position", `${pos + 1} / ${formatCount(filteredRows.length)}`);
    $("detail-prev").disabled = pos <= 0; $("detail-next").disabled = pos < 0 || pos >= filteredRows.length - 1;
    $("detail").scrollTop = 0;
    if (!$("detail").open) $("detail").showModal();
  }

  function resetFilters() {
    clearTimeout(timer);
    [...filterIds, "search"].forEach(id => { $(id).value = ""; });
    applyFilters();
  }
  filterIds.forEach(id => $(id).addEventListener("change", applyFilters));
  $("sort").addEventListener("change", applyFilters);
  $("search").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(applyFilters, 180); });
  $("reset").addEventListener("click", resetFilters); $("empty-reset").addEventListener("click", resetFilters);
  $("page-size").addEventListener("change", () => { page = 1; renderPage(); });
  $("prev").addEventListener("click", () => { page--; renderPage(); });
  $("next").addEventListener("click", () => { page++; renderPage(); });
  for (const name of ["list", "matrix"]) $("view-" + name).addEventListener("click", () => {
    view = name; page = 1;
    for (const item of ["list", "matrix"]) $("view-" + item).setAttribute("aria-pressed", String(item === view));
    renderPage();
  });
  $("results-body").addEventListener("click", event => {
    const button = event.target.closest("button[data-row]");
    if (button) showDetail(Number(button.dataset.row));
  });
  $("close-detail").addEventListener("click", () => $("detail").close());
  $("detail").addEventListener("click", event => { if (event.target === $("detail")) $("detail").close(); });
  for (const [id, step] of [["detail-prev", -1], ["detail-next", 1]]) $(id).addEventListener("click", () => {
    const pos = filteredRows.indexOf(allRows[selectedIndex]);
    const row = filteredRows[pos + step];
    if (row) showDetail(row._index);
  });
  $("export").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([exportCsv(filteredRows)], {type: "text/csv;charset=utf-8"}));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `question-analysis-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  applyFilters();
}

if (typeof module !== "undefined" && module.exports) module.exports = {escapeHtml, scoreValue, resultKind, resultLabel, explanationState, rowTime, matchesFilters, statistics, groupMatrix, compareRows, exportCsv};
if (typeof document !== "undefined") initQuestionMatrix();
