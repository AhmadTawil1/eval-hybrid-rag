"use strict";

const $ = (id) => document.getElementById(id);
const MODES = ["dense", "sparse", "hybrid"];
const SET_LABEL = { direct: "Direct", multihop: "Multi-hop", trap: "Trap: no answer exists" };
const TYPE_LABEL = {
  temporal_query: "about timing",
  comparison_query: "comparison",
  inference_query: "inference",
  null_query: "unanswerable",
};
const DEFAULT_QUESTION = "q11";

// Everything from the data files (answers, article passages) is inserted as text nodes, never as HTML.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

function seconds(ms) {
  return `${(ms / 1000).toFixed(1)} s`;
}

const state = { questions: [], filtered: [], index: 0 };

function passageList(passages) {
  const box = el("div");
  passages.forEach((p, i) => {
    const heading = el("div", { class: "src" }, `#${i + 1} ${p.label}`);
    if (p.needed) heading.append(el("span", { class: "badge" }, "needed evidence"));
    box.append(el("div", { class: p.needed ? "passage needed" : "passage" }, heading, el("p", {}, p.text)));
  });
  return box;
}

function statusLine(question, data) {
  if (question.set === "trap") {
    return data.refused
      ? el("p", { class: "status good" }, "Correct: it refused, as nothing in the collection answers this")
      : el("p", { class: "status bad" }, "Wrong: it gave an answer to an unanswerable question");
  }
  return data.refused
    ? el("p", { class: "status neutral" }, "Refused (\"Insufficient information.\")")
    : el("p", { class: "status neutral" }, "Answered");
}

function modeCard(mode, question) {
  const data = question.modes[mode];
  const card = el("div", { class: "card", "data-mode": mode });
  card.append(el("h3", {}, mode, el("span", { class: "meta" }, seconds(data.latency_ms))));
  card.append(statusLine(question, data));

  if (question.set !== "trap") {
    card.append(
      el("p", { class: "evidence" }, `Search found ${data.evidence_found} of ${question.evidence_needed} needed passages`),
    );
  }
  card.append(el("p", { class: data.refused ? "answer refusal" : "answer" }, data.answer));

  if (data.ragas) {
    const r = data.ragas;
    card.append(
      el(
        "p",
        { class: "scores" },
        `RAGAS: faithfulness ${r.faithfulness.toFixed(2)} · answer relevance ${r.answer_relevancy.toFixed(2)} · ` +
          `context precision ${r.context_precision.toFixed(2)} · context recall ${r.context_recall.toFixed(2)}`,
      ),
    );
  }
  if (data.citations.length) {
    const list = el("ul");
    for (const c of data.citations) list.append(el("li", {}, `${c.label} `, el("code", {}, c.chunk_id)));
    card.append(el("h4", {}, "Sources cited"), list);
  }
  card.append(
    el("details", {}, el("summary", {}, `The ${data.passages.length} passages the model saw`), passageList(data.passages)),
  );
  return card;
}

function showQuestion() {
  const question = state.filtered[state.index];
  $("question-select").value = question.id;
  $("position").textContent = `${state.index + 1} of ${state.filtered.length}`;
  $("prev").disabled = state.index === 0;
  $("next").disabled = state.index === state.filtered.length - 1;

  const tags = el(
    "div",
    { class: "tags" },
    el("span", { class: "tag" }, question.id),
    el("span", { class: "tag" }, SET_LABEL[question.set]),
    el("span", { class: "tag" }, TYPE_LABEL[question.type] || question.type),
  );
  if (question.set !== "trap") {
    tags.append(el("span", { class: "tag" }, `needs ${question.evidence_needed} passages from different articles`));
  }
  const reference =
    question.set === "trap"
      ? "Expected answer: \"Insufficient information.\" (the collection does not contain the answer)"
      : `Reference answer from the dataset: ${question.reference}`;

  $("detail").replaceChildren(
    el("div", { class: "qbox" }, tags, el("p", { class: "question" }, question.query), el("p", { class: "reference" }, reference)),
    el("div", { class: "modes" }, ...MODES.map((mode) => modeCard(mode, question))),
  );
}

function rebuildQuestionList(keepId) {
  const filter = $("set-filter").value;
  state.filtered = state.questions.filter((q) => filter === "all" || q.set === filter);
  const select = $("question-select");
  select.replaceChildren(
    ...state.filtered.map((q) => {
      const text = q.query.length > 95 ? `${q.query.slice(0, 95)}...` : q.query;
      return el("option", { value: q.id }, `${q.id} · ${text}`);
    }),
  );
  const found = state.filtered.findIndex((q) => q.id === keepId);
  state.index = found >= 0 ? found : 0;
  showQuestion();
}

function selectFeatured(id) {
  $("set-filter").value = "all";
  rebuildQuestionList(id);
  $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function initExplorer() {
  let data;
  try {
    data = await loadJson("data/explorer.json");
  } catch {
    $("detail").replaceChildren(el("p", { class: "notice error" }, "Could not load the recorded answers. If you opened this file directly, serve the folder with a web server instead."));
    return;
  }
  state.questions = data.questions;
  $("generator").textContent = data.generator;
  $("set-filter").addEventListener("change", () => rebuildQuestionList(state.filtered[state.index]?.id));
  $("question-select").addEventListener("change", (event) => {
    state.index = state.filtered.findIndex((q) => q.id === event.target.value);
    showQuestion();
  });
  $("prev").addEventListener("click", () => {
    state.index -= 1;
    showQuestion();
  });
  $("next").addEventListener("click", () => {
    state.index += 1;
    showQuestion();
  });
  for (const button of document.querySelectorAll("[data-featured]")) {
    button.addEventListener("click", () => selectFeatured(button.dataset.featured));
  }
  rebuildQuestionList(DEFAULT_QUESTION);
}

async function initResults() {
  const lead = $("results-lead");
  let results;
  try {
    results = await loadJson("data/results.json");
  } catch {
    lead.textContent = "Could not load the benchmark results.";
    return;
  }
  lead.textContent =
    `${results.n} questions from MultiHop-RAG (${results.n_answerable} answerable, scored by RAGAS; ` +
    `${results.n_trap} unanswerable, checked for a correct refusal), top ${results.k} passages per question. ` +
    `Generator ${results.models.generator}, judge ${results.models.judge}.`;

  const columns = [
    ["Faithfulness", (m) => m.ragas.faithfulness, (m) => m.ragas.faithfulness.toFixed(3), true],
    ["Answer relevance", (m) => m.ragas.answer_relevancy, (m) => m.ragas.answer_relevancy.toFixed(3), true],
    ["Context precision", (m) => m.ragas.context_precision, (m) => m.ragas.context_precision.toFixed(3), true],
    ["Context recall", (m) => m.ragas.context_recall, (m) => m.ragas.context_recall.toFixed(3), true],
    ["Trap refusal", (m) => m.trap_refusal.rate, (m) => `${m.trap_refusal.refused}/${m.trap_refusal.n}`, true],
    [
      "Latency mean / p95",
      (m) => m.latency_ms.mean,
      (m) => `${(m.latency_ms.mean / 1000).toFixed(1)} s / ${(m.latency_ms.p95 / 1000).toFixed(1)} s`,
      false,
    ],
  ];
  const entries = Object.entries(results.modes);
  const best = columns.map(([, value, , higher]) => {
    const values = entries.map(([, m]) => value(m));
    return higher ? Math.max(...values) : Math.min(...values);
  });
  const head = el("tr", {}, el("th", { scope: "col" }, "Mode"), ...columns.map(([name]) => el("th", { scope: "col" }, name)));
  const rows = entries.map(([mode, m]) =>
    el(
      "tr",
      {},
      el("th", { scope: "row" }, mode),
      ...columns.map(([, value, display], i) => el("td", value(m) === best[i] ? { class: "best" } : {}, display(m))),
    ),
  );
  $("results-table").replaceChildren(el("thead", {}, head), el("tbody", {}, ...rows));
  $("results-note").textContent =
    "Bold = best in column. RAGAS scores are averages over the answerable questions; with this few questions, gaps of 0.05 to 0.10 are within noise.";
}

async function initRepoLink() {
  try {
    const site = await loadJson("data/site.json");
    if (site.repo_url) {
      $("repo-link").replaceChildren("Source code: ", el("a", { href: site.repo_url }, site.repo_url));
    }
  } catch {
    /* the link is optional */
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initExplorer();
  initResults();
  initRepoLink();
});
