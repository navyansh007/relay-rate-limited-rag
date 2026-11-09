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
