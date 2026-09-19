import json
import pathlib

CORPUS_PATH = pathlib.Path("data/raw/corpus.json")


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def verify_indexes(bm25_ids: list[str]) -> None:
    from app.retrieval.dense import COLLECTION_NAME, get_qdrant_client

    client = get_qdrant_client()
    qdrant_ids, offset = set(), None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME, limit=1000, offset=offset, with_payload=["chunk_id"], with_vectors=False
        )
        qdrant_ids.update(p.payload["chunk_id"] for p in points)
        if offset is None:
            break

    bm25_set = set(bm25_ids)
    if bm25_set != qdrant_ids:
        raise RuntimeError(
            f"index mismatch: {len(bm25_set - qdrant_ids)} ids only in BM25, "
            f"{len(qdrant_ids - bm25_set)} only in Qdrant"
        )


def main(force: bool = False) -> None:
    from app.chunking import chunk_articles, save_chunks
    from app.retrieval import dense, sparse

    corpus = load_corpus()
    print(f"loaded {len(corpus)} articles")

    chunks = chunk_articles(corpus)
    save_chunks(chunks)
    print(f"saved {len(chunks)} chunks")

    dense.create_collection()
    if force or not dense.index_is_current(len(chunks)):
        dense.embed_and_upsert(chunks)
        print(f"dense: upserted {len(chunks)} chunks into '{dense.COLLECTION_NAME}'")
    else:
        print(f"dense: '{dense.COLLECTION_NAME}' already holds {len(chunks)} points, skipping embedding (--force to redo)")

    bm25, chunk_ids = sparse.build_bm25(chunks)
    sparse.save_bm25(bm25, chunk_ids)
    print(f"bm25: indexed {len(chunk_ids)} chunks")

    verify_indexes(chunk_ids)
    print("verified: BM25 and Qdrant chunk_ids match")


if __name__ == "__main__":
    import sys

    main(force="--force" in sys.argv[1:])
