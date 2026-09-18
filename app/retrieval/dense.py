import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.embeddings import get_embedding_model

COLLECTION_NAME = "mhrag_chunks"
VECTOR_SIZE = 384
BATCH_SIZE = 64
NAMESPACE = uuid.NAMESPACE_URL


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")


def create_collection() -> None:
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def embed_and_upsert(chunks: list[dict], batch_size: int = BATCH_SIZE) -> None:
    client = get_qdrant_client()
    model = get_embedding_model()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = model.encode(
            [c["text"] for c in batch],
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        points = [
            PointStruct(
                id=chunk_id_to_point_id(chunk["chunk_id"]),
                vector=vector.tolist(),
                payload=chunk,
            )
            for chunk, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query: str, k: int = 10) -> list[tuple[str, float]]:
    model = get_embedding_model()
    vector = model.encode(query, normalize_embeddings=True).tolist()

    result = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=k,
        with_payload=["chunk_id"],
    )
    return [(point.payload["chunk_id"], point.score) for point in result.points]


if __name__ == "__main__":
    from app.chunking import load_chunks

    create_collection()
    chunks = load_chunks()
    embed_and_upsert(chunks)
    print(f"upserted {len(chunks)} chunks into '{COLLECTION_NAME}'")
