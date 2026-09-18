import sys

from app.chunking import load_chunks
from app.retrieval import dense, hybrid, sparse

TOP = 5
POOL = hybrid.CANDIDATES

QUERIES = {
    "A (exact keywords)": "Sam Bankman-Fried FTX trial October 3 2023",
    "B (paraphrase)": "What happened to customers whose savings were stuck when the crypto platform imploded?",
}


def snippet(text: str, n: int = 90) -> str:
    return text.replace("\n", " ")[:n]


def run(label: str, query: str, chunks_by_id: dict[str, dict]) -> None:
    dense_ids = [cid for cid, _ in dense.search(query, POOL)]
    bm25_ids = [cid for cid, _ in sparse.search(query, POOL)]
    fused = hybrid.rrf_fuse(dense_ids, bm25_ids)

    print(f"\n{'=' * 100}\n{label}: {query}\n{'=' * 100}")
    for mode, ids in (("DENSE", dense_ids), ("BM25", bm25_ids), ("HYBRID", [c for c, _ in fused])):
        print(f"\n-- {mode} top {TOP}")
        for cid in ids[:TOP]:
            d = dense_ids.index(cid) + 1 if cid in dense_ids else "-"
            b = bm25_ids.index(cid) + 1 if cid in bm25_ids else "-"
            print(f"  {cid}  dense#{d:<3} bm25#{b:<3} {snippet(chunks_by_id[cid]['text'])}")

    top_h = [c for c, _ in fused[:TOP]]
    only_dense = [c for c in top_h if c in dense_ids[:TOP] and c not in bm25_ids[:TOP]]
    only_bm25 = [c for c in top_h if c in bm25_ids[:TOP] and c not in dense_ids[:TOP]]
    print(f"\nhybrid top {TOP}: {len(set(top_h))} distinct | from dense top-{TOP} only: {len(only_dense)} | from bm25 top-{TOP} only: {len(only_bm25)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    chunks_by_id = {c["chunk_id"]: c for c in load_chunks()}
    if len(sys.argv) > 1:
        queries = {f"custom {i + 1}": q for i, q in enumerate(sys.argv[1:])}
    else:
        queries = QUERIES
    for label, query in queries.items():
        run(label, query, chunks_by_id)
