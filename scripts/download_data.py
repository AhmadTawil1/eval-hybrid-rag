import json, pathlib
from datasets import load_dataset

out = pathlib.Path("data/raw"); out.mkdir(parents=True, exist_ok=True)

corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")

json.dump(corpus.to_list(),  open(out / "corpus.json", "w", encoding="utf-8"),       ensure_ascii=False)
json.dump(queries.to_list(), open(out / "MultiHopRAG.json", "w", encoding="utf-8"),  ensure_ascii=False)
print("articles:", len(corpus), "| queries:", len(queries))