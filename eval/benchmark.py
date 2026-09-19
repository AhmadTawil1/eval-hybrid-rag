import json
import pathlib
import sys

import numpy as np
from ragas import EvaluationDataset, evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

from app.config import get_settings
from app.embeddings import MODEL_NAME as EMBEDDING_MODEL
from app.generation import INSUFFICIENT
from app.pipeline import query
from eval.judge import get_judge_embeddings, get_judge_llm

DATASET_PATH = pathlib.Path("eval/test_dataset.json")
OUT_DIR = pathlib.Path("data/processed")
RESULTS_PATH = pathlib.Path("eval/results.json")
MODES = ["dense", "sparse", "hybrid"]
K = 5
METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_dataset() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def collect(mode: str, items: list[dict]) -> list[dict]:
    rows = []
    for n, item in enumerate(items, start=1):
        row = {
            "id": item["id"],
            "set": item["set"],
            "user_input": item["query"],
            "reference": item["answer"],
        }
        try:
            result = query(item["query"], mode, K)
            row.update(
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                citations=[c["chunk_id"] for c in result["citations"]],
                latency_ms=result["latency_ms"],
                error=None,
            )
        except Exception as e:
            row.update(
                response=None,
                retrieved_contexts=[],
                citations=[],
                latency_ms=None,
                error=f"{type(e).__name__}: {e}",
            )
        rows.append(row)
        status = row["error"] or f"{row['latency_ms']:.0f} ms"
        print(f"[{mode}] {n}/{len(items)} {item['id']} {status}", flush=True)
    return rows


def save_collected(mode: str, rows: list[dict]) -> None:
    with open(OUT_DIR / f"benchmark_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def load_collected(mode: str) -> list[dict]:
    with open(OUT_DIR / f"benchmark_{mode}.json", encoding="utf-8") as f:
        return json.load(f)


def score_mode(mode: str) -> dict:
    rows = [r for r in load_collected(mode) if r["set"] != "trap" and not r["error"]]
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), ContextPrecision(), ContextRecall()],
        llm=get_judge_llm(),
        embeddings=get_judge_embeddings(),
        raise_exceptions=False,
        show_progress=False,
        run_config=RunConfig(timeout=120, max_retries=2, max_workers=8),
        token_usage_parser=get_token_usage_for_openai,
    )
    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in rows])
    tokens = result.total_tokens()
    scored = {
        "mode": mode,
        "n_questions": len(rows),
        "means": {m: float(df[m].mean()) for m in METRIC_COLUMNS},
        "n_scored": {m: int(df[m].notna().sum()) for m in METRIC_COLUMNS},
        "judge_tokens": {"input": tokens.input_tokens, "output": tokens.output_tokens},
        "rows": json.loads(df[["id", *METRIC_COLUMNS]].to_json(orient="records")),
    }
    with open(OUT_DIR / f"ragas_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    return scored


def compute_stats(mode: str) -> dict:
    rows = [r for r in load_collected(mode) if not r["error"]]
    latencies = [r["latency_ms"] for r in rows]
    trap = [r for r in rows if r["set"] == "trap"]
    answerable = [r for r in rows if r["set"] != "trap"]
    refused = lambda r: r["response"].strip() == INSUFFICIENT
    return {
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "n": len(latencies),
        },
        "trap_refusal": {"refused": sum(map(refused, trap)), "n": len(trap)},
        "false_refusal": {"refused": sum(map(refused, answerable)), "n": len(answerable)},
    }


def build_results() -> dict:
    items = load_dataset()
    settings = get_settings()
    modes = {}
    for mode in MODES:
        with open(OUT_DIR / f"ragas_{mode}.json", encoding="utf-8") as f:
            ragas_means = json.load(f)["means"]
        stats = compute_stats(mode)
        modes[mode] = {
            "ragas": {m: round(v, 4) for m, v in ragas_means.items()},
            "latency_ms": {"mean": round(stats["latency_ms"]["mean"], 1), "p95": round(stats["latency_ms"]["p95"], 1)},
            "trap_refusal": {**stats["trap_refusal"], "rate": stats["trap_refusal"]["refused"] / stats["trap_refusal"]["n"]},
            "false_refusal": {**stats["false_refusal"], "rate": stats["false_refusal"]["refused"] / stats["false_refusal"]["n"]},
        }
    return {
        "dataset": "MultiHop-RAG",
        "n": len(items),
        "n_answerable": sum(1 for i in items if i["set"] != "trap"),
        "n_trap": sum(1 for i in items if i["set"] == "trap"),
        "k": K,
        "models": {
            "generator": settings.llm_model,
            "judge": settings.judge_model,
            "judge_embeddings": settings.judge_embedding_model,
            "retrieval_embeddings": EMBEDDING_MODEL,
        },
        "modes": modes,
    }


def save_results(results: dict) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


TABLE_COLUMNS = [
    ("Faithfulness", lambda m: m["ragas"]["faithfulness"], lambda m: f"{m['ragas']['faithfulness']:.3f}", True),
    ("Answer relevance", lambda m: m["ragas"]["answer_relevancy"], lambda m: f"{m['ragas']['answer_relevancy']:.3f}", True),
    ("Context precision", lambda m: m["ragas"]["context_precision"], lambda m: f"{m['ragas']['context_precision']:.3f}", True),
    ("Context recall", lambda m: m["ragas"]["context_recall"], lambda m: f"{m['ragas']['context_recall']:.3f}", True),
    ("Trap refusal", lambda m: m["trap_refusal"]["rate"], lambda m: f"{m['trap_refusal']['refused']}/{m['trap_refusal']['n']}", True),
    (
        "Latency mean / p95",
        lambda m: m["latency_ms"]["mean"],
        lambda m: f"{m['latency_ms']['mean'] / 1000:.1f} s / {m['latency_ms']['p95'] / 1000:.1f} s",
        False,
    ),
]


def format_table(results: dict) -> str:
    modes = results["modes"]
    header = "| Mode | " + " | ".join(name for name, *_ in TABLE_COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(TABLE_COLUMNS) + 1)]
    best = {}
    for name, value, _, higher_is_better in TABLE_COLUMNS:
        values = [value(m) for m in modes.values()]
        best[name] = max(values) if higher_is_better else min(values)
    for mode, m in modes.items():
        cells = []
        for name, value, display, _ in TABLE_COLUMNS:
            text = display(m)
            cells.append(f"**{text}**" if value(m) == best[name] else text)
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"n={results['n']} ({results['n_answerable']} answerable scored by RAGAS, {results['n_trap']} trap), "
        f"k={results['k']}. Bold = best in column."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["collect", "score", "results", "table"]
    if "collect" in steps:
        items = load_dataset()
        for mode in MODES:
            rows = collect(mode, items)
            save_collected(mode, rows)
            failed = sum(1 for r in rows if r["error"])
            print(f"[{mode}] collected {len(rows)} rows, {failed} failed", flush=True)
    if "score" in steps:
        for mode in MODES:
            scored = score_mode(mode)
            print(f"[{mode}] ragas means: {scored['means']} | n_scored: {scored['n_scored']}", flush=True)
    if "stats" in steps:
        for mode in MODES:
            print(f"[{mode}] {compute_stats(mode)}", flush=True)
    if "results" in steps:
        save_results(build_results())
        print(f"saved {RESULTS_PATH}", flush=True)
    if "table" in steps:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            print(format_table(json.load(f)))
