import time
from functools import lru_cache

from app.chunking import load_chunks
from app.generation import generate
from app.retrieval import dense, hybrid, sparse

SEARCHERS = {"dense": dense.search, "sparse": sparse.search, "hybrid": hybrid.search}


@lru_cache(maxsize=1)
def get_chunks_by_id() -> dict[str, dict]:
    return {c["chunk_id"]: c for c in load_chunks()}


def search(query: str, mode: str, k: int = 5) -> list[dict]:
    if mode not in SEARCHERS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(SEARCHERS)}")
    chunks_by_id = get_chunks_by_id()
    return [
        {
            "rank": rank,
            "chunk_id": chunk_id,
            "source_file": chunks_by_id[chunk_id]["source_file"],
            "section_header": chunks_by_id[chunk_id]["section_header"],
            "url": chunks_by_id[chunk_id]["url"],
            "text": chunks_by_id[chunk_id]["text"],
            "score": float(score),
        }
        for rank, (chunk_id, score) in enumerate(SEARCHERS[mode](query, k), start=1)
    ]


def query(query: str, mode: str, k: int = 5) -> dict:
    if mode not in SEARCHERS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(SEARCHERS)}")

    start = time.perf_counter()
    chunks_by_id = get_chunks_by_id()
    hits = SEARCHERS[mode](query, k)
    chunks = [chunks_by_id[chunk_id] for chunk_id, _ in hits]

    grounded = generate(query, chunks)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "answer": grounded.answer,
        "citations": [
            {
                "chunk_id": cid,
                "source_file": chunks_by_id[cid]["source_file"],
                "section_header": chunks_by_id[cid]["section_header"],
            }
            for cid in grounded.referenced_chunks
        ],
        "contexts": [c["text"] for c in chunks],
        "latency_ms": latency_ms,
    }
