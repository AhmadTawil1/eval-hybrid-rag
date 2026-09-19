# eval-hybrid-rag

## Limitations and next steps

### What the first benchmark showed

Benchmark: 50 questions from MultiHop-RAG (20 direct, 20 multi-hop, 10 trap), 3 retrieval modes (dense, sparse BM25, hybrid RRF), `k=5`, generator `gpt-5-mini`, RAGAS judge `gpt-5.6-terra`. Full numbers are in `eval/results.json`.

The main finding is that **retrieval recall, not the generator, is the bottleneck**:

- On average only 17% (dense), 21% (sparse) and 26% (hybrid) of the evidence chunks a question needs appeared in the top 5, and only 1 of the 40 answerable questions had all of its evidence retrieved, in every mode.
- The generator refused 22 to 26 of the 40 answerable questions ("Insufficient information."). In hybrid mode, none of the 22 refusals had all the needed evidence retrieved, so the refusals followed what retrieval delivered.
- Example: a question comparing a Fortune article with a TechCrunch article got 5 chunks that all came from the TechCrunch article, so a comparison was impossible.
- On the questions that were answered, RAGAS scores were much higher (faithfulness about 0.6 to 0.7, answer relevance about 0.6). The averages over all 40 are low mainly because refusals score 0 on answer relevance.
- Hybrid retrieved the most evidence and had the best context recall (0.625 vs 0.525 for dense), but with 40 questions per mode, gaps of 0.05 to 0.10 are within noise.

### Ideas to improve the experiment

Each idea is a hypothesis to test, not a result.

| Idea | Why it might help | How to measure |
|---|---|---|
| Larger `k` (10, 20) | Multi-hop questions need 2 to 5 chunks from different articles, and 5 rarely holds them all. | Context recall and false-refusal rate. Watch precision, token cost and latency. |
| Spread retrieval across articles (cap chunks per URL, or MMR) | The top 5 can fill up with chunks from one article and miss the second source. | Share of questions with all evidence retrieved. |
| Query decomposition | Split a multi-source question into one sub-query per source or entity, retrieve for each, then merge. | Same as above, plus answer quality on the multi-hop set. |
| Retrieval metrics on exact chunks | We already have `expected_chunk_ids`. Recall@k and hit rate would separate retrieval quality from generation quality and need no LLM judge. | Recall@k per mode. |
| Larger evaluation set | 50 questions is noisy. About 100 costs roughly $9 at current judge prices. | Bootstrap confidence intervals on the metric means. |
| Repeat runs | `gpt-5-mini` only accepts its default temperature, so runs are not fully repeatable. | Run-to-run spread of each metric. |
| An answer-correctness metric | None of the four RAGAS metrics compares the answer with the reference answer. | Exact match for yes/no questions, or RAGAS `answer_correctness`. |
| Tune the retrieval settings | RRF `k=60`, the candidate pool of 50, chunk size 512 and overlap 64, and the embedding model were all fixed defaults. A cross-encoder reranker is another option. | Context recall and precision per setting. |
| Cross-check the judge | Generator and judge are both OpenAI models, so mild same-vendor bias is possible. | Re-score a sample with a judge from another vendor. |
| Stricter grounding | Two trap questions (`q47`, `q48`) got plausible, cited but unsupported answers instead of a refusal. Inline `[chunk-id]` citations were also inconsistent with earlier models. | Trap refusal rate and citation coverage after a prompt or verification-pass change. |
| Report retrieval and LLM time separately | With a reasoning model the LLM call (10 to 12 s) hides retrieval-time differences between modes. | Latency split per step, or a lower reasoning effort or non-reasoning model for latency runs. |

### Known dataset limitations

- The direct set is almost all yes/no answers (19 of 20), so it says little about free-form answers.
- Nine of the 20 multi-hop answers are the same entity ("Sam Bankman-Fried"), so that set covers fewer topics than its size suggests.
- The exact-chunk evidence check is a strict lower bound: a different chunk containing the same information does not count as retrieved.
