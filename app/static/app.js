"use strict";

const $ = (id) => document.getElementById(id);
const MODES = ["dense", "sparse", "hybrid"];
const EXAMPLES = [
  "Who founded the studio behind the game Cocoon?",
  "What happened to customers whose savings were stuck when the crypto platform imploded?",
  "Considering the information from a Bloomberg article discussing the release date of the latest Apple MacBook Pro and a CNET article detailing the new features of the same device, which letter represents the first character of the feature that is both newly introduced in the latest model according to CNET and is specifically mentioned as being anticipated prior to the release date in the Bloomberg article?",
];

// Everything that comes from the server (model answers, article text) is inserted as text nodes,
// never as HTML, so a passage cannot inject markup or script.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

function detailText(data) {
  if (!data || !data.detail) return "";
  if (Array.isArray(data.detail)) return "Please check your question: it needs 1 to 500 characters.";
  return String(data.detail);
}

async function api(path, body) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Could not reach the server. Please try again.");
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    /* non-JSON error body */
  }
  if (!response.ok) {
    throw new Error(detailText(data) || `The server returned an error (${response.status}).`);
  }
  return data;
}

function seconds(ms) {
  return `${(ms / 1000).toFixed(ms < 1000 ? 2 : 1)} s`;
}

function startTimer(meta) {
  const started = Date.now();
  const tick = () => {
    meta.textContent = `waiting ${Math.floor((Date.now() - started) / 1000)} s (the model thinks before answering, about 10 s)`;
  };
  tick();
  const id = setInterval(tick, 500);
  return () => clearInterval(id);
}

function passagesList(passages) {
  const box = el("div");
  for (const p of passages) {
    box.append(
      el(
        "div",
        { class: "passage" },
        el("div", { class: "src" }, `#${p.rank} ${p.section_header || p.source_file}`),
        el("div", { class: "muted small" }, `score ${p.score.toFixed(3)}`),
        el("p", {}, p.text),
      ),
    );
  }
  return box;
}

function renderAnswer(card, mode, data) {
  card.replaceChildren(el("h3", {}, mode, el("span", { class: "meta" }, seconds(data.latency_ms))));
  if (data.generation === "paused_daily_limit") {
    card.append(el("p", { class: "notice warn" }, data.answer), el("h4", {}, "Retrieved passages"), passagesList(data.passages));
    return;
  }
  card.append(el("p", { class: "answer" }, data.answer));
  if (data.citations.length) {
    const list = el("ul");
    for (const c of data.citations) {
      list.append(el("li", {}, `${c.section_header || c.source_file} `, el("code", {}, c.chunk_id)));
    }
    card.append(el("h4", {}, "Sources cited"), list);
  }
}

function renderError(card, mode, error) {
  card.replaceChildren(el("h3", {}, mode), el("p", { class: "notice error" }, error.message));
}

function setBusy(busy) {
  for (const id of ["ask", "compare", "search"]) $(id).disabled = busy;
  if (!busy && limits && !limits.generation_enabled) {
    $("ask").disabled = true;
    $("compare").disabled = true;
  }
}

let limits = null;

async function refreshLimits() {
  const note = $("limits");
  try {
    limits = await api("/api/v1/limits");
  } catch {
    note.className = "notice";
    note.textContent = "Could not load the usage limits.";
    return;
  }
  note.className = "notice";
  if (!limits.generation_enabled) {
    note.classList.add("warn");
    note.textContent = "Answer generation is switched off on this server, so only \"Search only\" works.";
  } else if (limits.daily_answers_remaining <= 0) {
    note.classList.add("warn");
    note.textContent =
      "Today's answer budget for this demo is used up. \"Ask\" now shows the retrieved passages without an AI-written answer, and \"Search only\" is unaffected.";
  } else {
    note.textContent =
      `Limits: ${limits.answers_per_visitor_per_hour} answers per visitor per hour, ` +
      `${limits.daily_answers_remaining} of ${limits.daily_answers_limit} answers left today. "Search only" is free.`;
  }
  setBusy(false);
}

function currentQuestion() {
  const question = $("question").value.trim();
  if (!question) {
    $("output").className = "";
    $("output").replaceChildren(el("p", { class: "notice error" }, "Please type a question first."));
    return null;
  }
  return question;
}

async function askOne(mode, question) {
  const card = el("div", { class: "card", "data-mode": mode });
  const meta = el("span", { class: "meta" });
  card.append(el("h3", {}, mode, meta));
  const stop = startTimer(meta);
  const result = api("/api/v1/query", { query: question, mode })
    .then((data) => renderAnswer(card, mode, data))
    .catch((error) => renderError(card, mode, error))
    .finally(stop);
  return { card, done: result };
}

async function onAsk(event) {
  event.preventDefault();
  const question = currentQuestion();
  if (!question) return;
  setBusy(true);
  const { card, done } = await askOne($("mode").value, question);
  $("output").className = "";
  $("output").replaceChildren(card);
  await done;
  await refreshLimits();
}

async function onCompare() {
  const question = currentQuestion();
  if (!question) return;
  setBusy(true);
  const runs = await Promise.all(MODES.map((mode) => askOne(mode, question)));
  $("output").className = "compare";
  $("output").replaceChildren(...runs.map((run) => run.card));
  await Promise.all(runs.map((run) => run.done));
  await refreshLimits();
}

async function onSearch() {
  const question = currentQuestion();
  if (!question) return;
  setBusy(true);
  const mode = $("mode").value;
  const card = el("div", { class: "card", "data-mode": mode }, el("h3", {}, `${mode}: retrieved passages`));
  $("output").className = "";
  $("output").replaceChildren(card);
  try {
    const data = await api("/api/v1/search", { query: question, mode, k: 5 });
    card.replaceChildren(
      el("h3", {}, `${mode}: retrieved passages`, el("span", { class: "meta" }, seconds(data.latency_ms))),
      passagesList(data.results),
    );
  } catch (error) {
    renderError(card, mode, error);
  }
  setBusy(false);
}

function updateCounter() {
  $("counter").textContent = `${$("question").value.length} / 500`;
}

async function loadResults() {
  const lead = $("results-lead");
  let results;
  try {
    results = await api("/api/v1/metrics");
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

document.addEventListener("DOMContentLoaded", () => {
  $("ask-form").addEventListener("submit", onAsk);
  $("compare").addEventListener("click", onCompare);
  $("search").addEventListener("click", onSearch);
  $("question").addEventListener("input", updateCounter);
  for (const button of document.querySelectorAll("[data-example]")) {
    button.addEventListener("click", () => {
      $("question").value = EXAMPLES[Number(button.dataset.example)];
      updateCounter();
      $("question").focus();
    });
  }
  updateCounter();
  refreshLimits();
  loadResults();
});
