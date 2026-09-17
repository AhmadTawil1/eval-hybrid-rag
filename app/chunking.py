import hashlib
import json
import pathlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

CHUNKS_PATH = pathlib.Path("data/processed/chunks.jsonl")


def make_chunk_id(url: str, position: int, text: str) -> str:
    return hashlib.sha1(f"{url}|{position}|{text}".encode()).hexdigest()[:16]


def chunk_articles(corpus: list[dict]) -> list[dict]:
    chunks = []
    for article in corpus:
        section_header = article["title"]
        if article.get("source"):
            section_header = f"{section_header} — {article['source']}"

        for position, text in enumerate(splitter.split_text(article["body"])):
            chunks.append(
                {
                    "chunk_id": make_chunk_id(article["url"], position, text),
                    "text": text,
                    "position": position,
                    "source_file": article["title"] or article["url"],
                    "section_header": section_header,
                    "url": article["url"],
                    "published_at": article["published_at"],
                    "source": article["source"],
                }
            )
    return chunks


def save_chunks(chunks: list[dict], path: pathlib.Path = CHUNKS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from app.ingest import load_corpus

    corpus = load_corpus()
    chunks = chunk_articles(corpus)
    save_chunks(chunks)
    print(len(chunks))
