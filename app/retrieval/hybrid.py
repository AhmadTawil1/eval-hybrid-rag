from app.retrieval import dense, sparse

RRF_K = 60
CANDIDATES = 50


def rrf_fuse(dense_ids: list[str], bm25_ids: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in (dense_ids, bm25_ids):
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def search(query: str, k: int = 10) -> list[tuple[str, float]]:
    dense_ids = [cid for cid, _ in dense.search(query, CANDIDATES)]
    bm25_ids = [cid for cid, _ in sparse.search(query, CANDIDATES)]
    return rrf_fuse(dense_ids, bm25_ids)[:k]
