import json
import pathlib
import random
from collections import defaultdict

from rapidfuzz import fuzz

from app.chunking import load_chunks

QUERIES_PATH = pathlib.Path("data/raw/MultiHopRAG.json")
DATASET_PATH = pathlib.Path("eval/test_dataset.json")
FUZZY_THRESHOLD = 85


def load_queries() -> list[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def group_by_type(queries: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for q in queries:
        groups[q["question_type"]].append(q)
    return dict(groups)


SAMPLE_PLAN = [
    ("direct", "temporal_query", 5),
    ("direct", "comparison_query", 5),
    ("multihop", "inference_query", 10),
    ("trap", "null_query", 5),
]


def sample_items(groups: dict[str, list[dict]]) -> list[dict]:
    random.seed(42)
    items = []
    for set_name, question_type, n in SAMPLE_PLAN:
        for q in random.sample(groups[question_type], n):
            items.append({**q, "set": set_name})
    return items


def group_chunks_by_url(chunks: list[dict]) -> dict[str, list[dict]]:
    by_url = defaultdict(list)
    for c in chunks:
        by_url[c["url"]].append(c)
    return dict(by_url)


def find_chunks_for_fact(fact: str, candidates: list[dict]) -> list[dict]:
    exact = [c for c in candidates if fact in c["text"]]
    if exact:
        return exact
    return [c for c in candidates if fuzz.partial_ratio(fact, c["text"]) >= FUZZY_THRESHOLD]


def map_expected_chunks(item: dict, chunks_by_url: dict[str, list[dict]]) -> tuple[list[str], list[dict]]:
    if item["question_type"] == "null_query":
        return [], []

    expected, unmatched = [], []
    for ev in item["evidence_list"]:
        matched = find_chunks_for_fact(ev["fact"], chunks_by_url.get(ev["url"], []))
        if not matched:
            unmatched.append(ev)
        expected.extend(c["chunk_id"] for c in matched)
    return list(dict.fromkeys(expected)), unmatched


if __name__ == "__main__":
    groups = group_by_type(load_queries())
    items = sample_items(groups)
    print(f"sampled {len(items)} items")
    for set_name, question_type, n in SAMPLE_PLAN:
        picked = [i for i in items if i["set"] == set_name and i["question_type"] == question_type]
        print(f"  {set_name:9} {question_type:17} {len(picked)}")

    chunks_by_url = group_chunks_by_url(load_chunks())
    total_facts = unmatched_facts = 0
    for item in items:
        item["expected_chunk_ids"], unmatched = map_expected_chunks(item, chunks_by_url)
        total_facts += len(item["evidence_list"])
        unmatched_facts += len(unmatched)
        for ev in unmatched:
            print(f"  NO CHUNK: {ev['url']} | {ev['fact'][:80]!r}")
    print(f"evidence facts with no chunk: {unmatched_facts} / {total_facts}")

    dataset = [
        {
            "id": f"q{n:02d}",
            "query": item["query"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "set": item["set"],
            "expected_chunk_ids": item["expected_chunk_ids"],
        }
        for n, item in enumerate(items, start=1)
    ]
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"saved {len(dataset)} items to {DATASET_PATH}")
