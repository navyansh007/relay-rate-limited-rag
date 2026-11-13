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
| `config/scenarios/*.yaml` | Load shapes: `real_steady`, `real_burst` (real); `steady`, `burst`, `burst_cache`, `overload`, `overload_nolimiter`, `overload_naive`, `overload_deadline8`, `rpm_halved`, `day_exhausted` (stand-in) |
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

# 4. Load scenarios (stand-in must be running for stand-in scenarios)
uv run python -m relay.loadgen config/scenarios/real_steady.yaml config/scenarios/real_burst.yaml --real
uv run python -m relay.loadgen config/scenarios/{steady,burst,burst_cache,overload,overload_nolimiter,overload_naive,overload_deadline8,rpm_halved,day_exhausted}.yaml

# 5. Daily limit (last thing on a day: spends gpt-oss-20b's daily tokens)
uv run python -m relay.probes day --model openai/gpt-oss-20b --prompt-tokens 6000 --max-total 600000

uv run python -m relay.analysis
```

One question, with any config override:

```bash
uv run python -m relay.app "What frame rate does Mimi run at?" --real \
    --set retrieval.setting=s1_small_many --set pipeline.verify_mode=strict
```

The API key is read from `.env` and never logged. Real-service runs require
an explicit `--real` flag, and probes stop at a per-run token ceiling.

## Notes on the data

- Run load scenarios on mains power or under `caffeinate -i`: a sleeping laptop pauses the
  load generator's clock but not the wall-clock rate limits. The summary line reports
  `clock_drift_s`; a run with drift is invalid (see `raw/invalid_runs_mac_slept.csv`).
- Rows set aside, with the reason, are kept in `raw/`: `aborted_real_steady_s3.csv`
  (retrieval default changed mid-run), `invalid_runs_mac_slept.csv`,
  `superseded_day_exhausted_v1.csv` (reply text improved, scenario re-run).
- `eval` rows from run 1 with status `rate_limited` are real but ungraded: they are
  limiter fallbacks, not model answers (see DECISIONS.md, 2026-11-09).
