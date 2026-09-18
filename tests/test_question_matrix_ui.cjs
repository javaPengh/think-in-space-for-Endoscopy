// No browser required: verify analysis functions and page wiring with a minimal DOM stub.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ui = require("../tools/templates/question_matrix.js");

const base = {doc_id: 1, data_version: "v1", question: "Which side?", options: "A / B", ground_truth: "A", model: "qwen", sampling_strategy: "uniform_16f", timestamp: "2026-09-18T12:00:00", sample_path: "run1.json", answer_type: "multiple_choice", score: 0, is_correct: false, answer_mode: "restricted", restricted_prediction: "B", first_turn_prediction: "B", answer_explanations: [{answer: "B", prompt: "Please explain your answer step by step.", explanation: "I looked left.", request_index: 0, repeat_index: 0}, {answer: "B", explanation: "Second attempt", request_index: 0, repeat_index: 1}]};
const rows = [base, {...base, doc_id: 2, is_correct: true, score: 100, answer_explanations: []}, {...base, sample_path: "run2.json", model: "internvl", answer_type: "numeric", is_correct: null, score: 50, answer_explanations: []}, {...base, doc_id: 3, answer_type: "numeric", is_correct: null, score: 0, answer_explanations: [{explanation: ""}]}, {...base, doc_id: 4, score: null, is_correct: null, answer_explanations: []}];

assert.equal(ui.resultKind(rows[2]), "partial");
assert.equal(ui.resultKind(rows[3]), "zero");
assert.equal(ui.resultKind(rows[4]), "unscored");
assert.equal(ui.explanationState(rows[1]), "missing");
assert.equal(ui.explanationState(rows[3]), "empty");
assert.equal(ui.matchesFilters(rows[2], {status: "attention"}), false);
assert.equal(ui.matchesFilters(rows[3], {status: "attention"}), true);
assert.equal(ui.matchesFilters(rows[0], {keyword: "second attempt"}), true);
assert.equal(ui.matchesFilters(rows[0], {explanation: "missing"}), false);
assert.equal(ui.matchesFilters({...base, timestamp: "unknown"}, {start: 1}), false);
assert.equal(ui.matchesFilters(rows[0], {start: Date.parse("2026-09-19T00:00:00")}), false);
assert.equal(ui.rowTime({...base, timestamp: "0716_1142", sample_path: "logs/20260716/run/file.json"}), Date.parse("2026-07-16T11:42:00"));
assert.equal(ui.rowTime({...base, timestamp: "2026-09-18T12-00-00"}), Date.parse("2026-09-18T12:00:00"));
const stats = ui.statistics(rows);
assert.deepEqual([stats.choices, stats.correct, stats.numeric, stats.numericSum, stats.attention, stats.explained, stats.empty], [2, 1, 2, 50, 2, 1, 1]);
const groups = ui.groupMatrix([base, {...base, sample_path: "run2.json"}, {...base, data_version: "v2"}, {...base, question: "Different question"}, {...base}]);
assert.equal(groups.runs.length, 2);
assert.equal(groups.questions.length, 3);
assert.equal(groups.questions[0].cells.get("run1.json").length, 2);
assert.deepEqual([...rows].sort((a, b) => ui.compareRows(a, b, "score")).map(row => row.score), [0, 0, 50, 100, null]);
assert.equal(ui.escapeHtml('<script>"&'), "&lt;script&gt;&quot;&amp;");
const csv = ui.exportCsv([{...base, question: '=HYPERLINK("bad")', natural_prediction: "line1\nline2"}]);
assert.ok(csv.startsWith("\uFEFF"));
assert.ok(csv.includes("'=HYPERLINK"));
assert.ok(csv.includes('"line1\nline2"'));
assert.ok(csv.includes("Second attempt"));
console.log("PASS analysis logic: scoring, filters, provenance, alignment, sorting, CSV");

const elements = new Map();
class Element {
  constructor(id) { this.id = id; this.value = ""; this.textContent = ""; this.events = {}; this.attributes = {}; this.open = false; this._html = ""; }
  set innerHTML(value) {
    this._html = value;
    for (const match of value.matchAll(/\bid="([^"]+)"/g)) if (!elements.has(match[1])) elements.set(match[1], new Element(match[1]));
  }
  get innerHTML() { return this._html; }
  addEventListener(type, callback) { this.events[type] = callback; }
  setAttribute(name, value) { this.attributes[name] = value; }
  showModal() { this.open = true; }
  close() { this.open = false; }
  fire(type, event = {target: this}) { this.events[type](event); }
}
const template = fs.readFileSync(path.join(__dirname, "../tools/templates/question_matrix.html"), "utf8");
for (const match of template.matchAll(/\bid="([^"]+)"/g)) elements.set(match[1], new Element(match[1]));
const $ = id => elements.get(id) || null;
$("page-size").value = "50"; $("sort").value = "newest";
$("matrix-data").textContent = JSON.stringify({rows, updated_at: "test"});
const script = fs.readFileSync(path.join(__dirname, "../tools/templates/question_matrix.js"), "utf8");
vm.runInNewContext(script, {document: {getElementById: $, body: {}}, console, setTimeout, clearTimeout, Blob, URL});
assert.equal($("stat-records").textContent, "5");
assert.equal($("stat-accuracy").textContent, "50.0%");
assert.equal($("stat-mra").textContent, "25.0");
assert.ok($("results-body").innerHTML.includes("阅读第二轮解释"));
$("view-matrix").fire("click");
assert.equal($("results-table").className, "matrix-table");
assert.equal($("view-matrix").attributes["aria-pressed"], "true");
assert.ok($("results-head").innerHTML.includes("internvl"));
$("results-body").fire("click", {target: {closest: () => ({dataset: {row: "0"}})}});
assert.equal($("detail").open, true);
assert.ok($("detail-content").innerHTML.includes("I looked left."));
$("repeat-selector").value = "1"; $("repeat-selector").fire("change");
assert.ok($("explanation-content").innerHTML.includes("Second attempt"));
$("close-detail").fire("click");
assert.equal($("detail").open, false);
$("status-filter").value = "incorrect"; $("status-filter").fire("change");
assert.equal($("stat-records").textContent, "1");
$("model-filter").value = "missing"; $("model-filter").fire("change");
assert.equal($("empty").hidden, false);
assert.equal($("export").disabled, true);
$("reset").fire("click");
assert.equal($("stat-records").textContent, "5");
$("view-list").fire("click");
$("page-size").value = "2"; $("page-size").fire("change");
assert.equal(($("results-body").innerHTML.match(/<tr>/g) || []).length, 2);
$("next").fire("click");
assert.ok($("page-info").textContent.includes("第 2 / 3 页"));
console.log("PASS page wiring: overview, list/matrix switch, details/repeats, filters, empty state, pagination");
