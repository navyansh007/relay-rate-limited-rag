# Decisions log

Running log of design decisions, the measurements that changed them, and the surprises
along the way. It is the reasoning behind the numbers in the README.

| Date | Decision / observation | Why | Evidence |
|---|---|---|---|
| 2026-11-09 | Python 3.12 + uv, asyncio + httpx | Streaming + header access, one event loop for pipeline, load generator and stand-in | — |
| 2026-11-09 | Real Groq calls only behind an explicit `--real` flag + spend guard | Daily budget is shared by everything; dev must default to the stand-in | Free-tier daily limits |
| 2026-11-09 | Groq docs: `x-ratelimit-*-requests` = **per day**, `x-ratelimit-*-tokens` = **per minute**; `retry-after` only on 429 | RPM and TPD are NOT exposed in headers → must be found by probing / console limits page | console.groq.com/docs/rate-limits (fetched 2026-11-09) |
| 2026-11-09 | Free-tier table not in public docs (only Developer plan shown) | Limit values must come from headers + deliberate probing, confidence graded accordingly | same |
| 2026-11-09 | Candidate models: `llama-3.1-8b-instant` (understand/verify), `openai/gpt-oss-120b` / `gpt-oss-20b` (draft) | gpt-oss has published prices ($0.15/$0.60 and $0.075/$0.30 per M) → cost per 1k answers is computable; but they are reasoning models: hidden reasoning tokens delay first visible token and consume TPM. Final pick after probes | console.groq.com/docs/models |
| 2026-11-09 | Papers verified by the `arXiv:<id>v<N>` stamp on page 1 (all three match) | Answers must match the exact paper versions; v7 abstract says EN-FR BLEU 41.8 — a model answering from memory of v1 may say 41.0 | `relay.ingest.download` |
| 2026-11-09 | Sections from the PDF outline (hyperref TOC), not font heuristics | Heading fonts differ per paper; the TOC is exact. Heading must have its number on/just before the line: Transformer's Figure 2 labels repeat "Scaled Dot-Product Attention" verbatim and first mis-assigned §3.2.1 text to §3.2.2 (wrong citation) | caught by inspecting section dump |
| 2026-11-09 | Bibliography excluded from the index | Reference titles ("…attention…", "…tokenization…") match queries lexically but contain no answers — naive-retriever trap #1 candidate | ~55k chars of references across papers |
| 2026-11-09 | Chunks never cross section boundaries; tables hard-split by words | One citable location per chunk; tables have no sentence punctuation and produced a 756-token "sentence" | chunk stats |
| 2026-11-09 | Hybrid BM25 + bge-small dense, RRF fusion, optional MiniLM-L6 cross-encoder (all local) | BM25 catches exact numbers/names ("Mimi", "41.8"); dense catches paraphrase | `relay.retrieval` |
| 2026-11-09 | **Measured: re-rank is not free.** Warm hybrid search ≈4 ms; re-ranking 20 candidates ≈650 ms (200-tok pieces) / 920 ms (300) / 1110 ms (500), M-series CPU. 10 candidates ≈240–540 ms. Cold first call ≈10 s (model load) → must warm up at startup | Re-rank trades TTFT + shared CPU for fewer tokens; under bursts the CPU re-ranker is a queue of its own. Candidate pool size becomes a load knob | ad-hoc timing run, 12 queries per setting |
| 2026-11-09 | Observed: "What frame rate does Mimi operate at?" never surfaces §3.3 (where it is defined); hits are eval tables mentioning Mimi | Naive-retriever trap candidate: definitional passages lose to tables that repeat the term many times | retrieval smoke test |
