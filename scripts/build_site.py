"""Build the data and copy the media the static site (site/) needs.

Reads the saved benchmark outputs in data/processed/ (git-ignored, produced by
`python -m eval.benchmark`) and writes committed, deployable files under site/.
Optional: SITE_REPO_URL=https://github.com/you/repo python -m scripts.build_site
"""

import json
import os
import pathlib
import shutil

from app.chunking import load_chunks
from app.generation import INSUFFICIENT

SITE = pathlib.Path("site")
PROCESSED = pathlib.Path("data/processed")
MODES = ["dense", "sparse", "hybrid"]
RAGAS_KEYS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def read_json(path: pathlib.Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_explorer(results: dict) -> dict:
    dataset = read_json(pathlib.Path("eval/test_dataset.json"))
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    id_by_text: dict[str, str] = {}
    for c in chunks:
        id_by_text.setdefault(c["text"], c["chunk_id"])

    rows = {m: {r["id"]: r for r in read_json(PROCESSED / f"benchmark_{m}.json")} for m in MODES}
    ragas = {m: {r["id"]: r for r in read_json(PROCESSED / f"ragas_{m}.json")["rows"]} for m in MODES}

    questions = []
    for item in dataset:
        needed = set(item["expected_chunk_ids"])
        entry = {
            "id": item["id"],
            "set": item["set"],
            "type": item["question_type"],
            "query": item["query"],
            "reference": item["answer"],
            "evidence_needed": len(needed),
            "modes": {},
        }
        for mode in MODES:
            row = rows[mode][item["id"]]
            if row["error"]:
                raise ValueError(f"{mode} {item['id']} has an error row: {row['error']}")
            retrieved_ids = [id_by_text[text] for text in row["retrieved_contexts"]]
            mode_entry = {
                "answer": row["response"],
                "refused": row["response"].strip() == INSUFFICIENT,
                "latency_ms": round(row["latency_ms"]),
                "citations": [{"chunk_id": c, "label": by_id[c]["section_header"]} for c in row["citations"]],
                "evidence_found": len(needed & set(retrieved_ids)),
                "passages": [
                    {
                        "chunk_id": cid,
                        "label": by_id[cid]["section_header"],
                        "text": by_id[cid]["text"],
                        "needed": cid in needed,
                    }
                    for cid in retrieved_ids
                ],
            }
            scores = ragas[mode].get(item["id"])
            if scores:
                mode_entry["ragas"] = {k: round(scores[k], 3) for k in RAGAS_KEYS}
            entry["modes"][mode] = mode_entry
        questions.append(entry)

    return {
        "k": results["k"],
        "generator": results["models"]["generator"],
        "questions": questions,
    }


def main() -> None:
    results = read_json(pathlib.Path("eval/results.json"))
    explorer = build_explorer(results)
    write_json(SITE / "data" / "explorer.json", explorer)
    write_json(SITE / "data" / "results.json", results)
    write_json(SITE / "data" / "site.json", {"repo_url": os.environ.get("SITE_REPO_URL") or None})

    media = {
        "eval/charts/02_ragas_metrics.png": "img/02_ragas_metrics.png",
        "eval/charts/03_refusal_behavior.png": "img/03_refusal_behavior.png",
        "eval/charts/04_latency.png": "img/04_latency.png",
        "eval/charts/05_evidence_retrieved.png": "img/05_evidence_retrieved.png",
        "eval/charts/06_answerable_outcomes.png": "img/06_answerable_outcomes.png",
        "assets/architecture.png": "img/architecture.png",
        "paper/experiment_report.pdf": "report.pdf",
    }
    for source, target in media.items():
        destination = SITE / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    size_kb = (SITE / "data" / "explorer.json").stat().st_size / 1024
    print(f"explorer: {len(explorer['questions'])} questions x {len(MODES)} modes, {size_kb:.0f} KB")
    print(f"copied {len(media)} media files into {SITE}/")


if __name__ == "__main__":
    main()
