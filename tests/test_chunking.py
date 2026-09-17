from app.chunking import chunk_articles

SAMPLE_CORPUS = [
    {
        "title": "Sample Article",
        "url": "https://example.com/sample",
        "published_at": "2024-01-01 00:00:00",
        "source": "Example News",
        "body": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 60,
    },
    {
        "title": "Second Article",
        "url": "https://example.com/second",
        "published_at": "2024-02-02 00:00:00",
        "source": "Example News",
        "body": "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " * 40,
    },
]


def test_same_input_produces_same_ids_twice():
    ids_1 = [c["chunk_id"] for c in chunk_articles(SAMPLE_CORPUS)]
    ids_2 = [c["chunk_id"] for c in chunk_articles(SAMPLE_CORPUS)]
    assert ids_1 == ids_2
    assert len(set(ids_1)) == len(ids_1)


def test_no_chunk_longer_than_512_chars():
    chunks = chunk_articles(SAMPLE_CORPUS)
    assert len(chunks) > 0
    assert all(len(c["text"]) <= 512 for c in chunks)
