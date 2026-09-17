import json
import pathlib

CORPUS_PATH = pathlib.Path("data/raw/corpus.json")


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    corpus = load_corpus()
    print(len(corpus))
