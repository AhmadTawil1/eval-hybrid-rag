import pathlib
import pickle
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

BM25_PATH = pathlib.Path("data/processed/bm25.pkl")


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def build_bm25(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    chunk_ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return bm25, chunk_ids


def save_bm25(bm25: BM25Okapi, chunk_ids: list[str], path: pathlib.Path = BM25_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)


def load_bm25(path: pathlib.Path = BM25_PATH) -> tuple[BM25Okapi, list[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


@lru_cache(maxsize=1)
def get_bm25() -> tuple[BM25Okapi, list[str]]:
    return load_bm25()


def search(query: str, k: int = 10) -> list[tuple[str, float]]:
    bm25, chunk_ids = get_bm25()
    scores = bm25.get_scores(tokenize(query))
    top = scores.argsort()[::-1][:k]
    return [(chunk_ids[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    from app.chunking import load_chunks

    chunks = load_chunks()
    bm25, chunk_ids = build_bm25(chunks)
    save_bm25(bm25, chunk_ids)
    print(f"built and saved BM25 over {len(chunk_ids)} chunks to {BM25_PATH}")
