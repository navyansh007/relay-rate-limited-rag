# The Relay

A rate-limit-aware assistant that answers questions about three arXiv papers
with citations, under real free-tier rate limits. It runs against
Groq's free tier, or against a local stand-in service that copies Groq's
interface and can be slow, fail, refuse at limits and degrade under load.

## Layout

| Path | What |
|---|---|
| `src/relay/ingest.py` | Downloads the pinned paper versions, splits them into section-aware citable chunks |
| `src/relay/retrieval.py` | Hybrid BM25 + dense retrieval, optional cross-encoder re-rank (the tool) |
| `src/relay/pipeline.py` | understand → retrieve → decide (branch/loop) → draft → verify |
| `src/relay/verify.py` | Per-sentence citation and number checks |
| `src/relay/limiter.py`, `retry.py` | Header-corrected rate limiter; budgeted retries |
| `src/relay/service.py` | Admission queue, coalescing, cache |
| `src/relay/standin.py` | Groq-compatible stand-in model service |
| `src/relay/probes.py`, `calibrate.py` | Measure the real service; fit the stand-in to it |
| `src/relay/loadgen.py` | Open-loop load scenarios (`config/scenarios/*.yaml`) |
| `src/relay/evaluate.py`, `grade.py` | Quality runs per retrieval setting; human-reviewed grades |
| `src/relay/analysis.py` | Tables 2a, 2b, 3, computed only from `results/results.csv` |
| `config/default.yaml` | Every tunable (models, limits, retrieval, retries, deadlines) |
| `DECISIONS.md` | Decisions, measurements that changed them, assistant overrides |

## Setup

```bash
uv sync
cp .env.example .env            # add your own GROQ_API_KEY
uv run python -m relay.ingest   # downloads papers, verifies versions, builds chunks
uv run pytest -q
```

## Reproduce

```bash
# 1. Learn the real service (small, spend-guarded)
uv run python -m relay.probes latency --n 40 --gap 3
uv run python -m relay.probes rpm --n 40 --window 20 && uv run python -m relay.probes recover
uv run python -m relay.calibrate            # fits config/standin_calibration.yaml

# 2. Stand-in (separate terminal)
uv run python -m relay.standin

# 3. Quality per retrieval setting (real), then review grades
uv run python -m relay.evaluate --real --settings s1_small_many,s2_big_few,s3_rerank
uv run python -m relay.grade propose        # fill the `final` column in eval/grades.csv
uv run python -m relay.grade apply

# 4. Load scenarios
uv run python -m relay.loadgen config/scenarios/<name>.yaml [--real]

# 5. Tables
uv run python -m relay.analysis
```

One question, with any config override:

```bash
uv run python -m relay.app "What frame rate does Mimi run at?" --real \
    --set retrieval.setting=s1_small_many --set pipeline.verify_mode=strict
```

The API key is read from `.env` and never logged. Real-service runs require
an explicit `--real` flag, and probes stop at a per-run token ceiling.
