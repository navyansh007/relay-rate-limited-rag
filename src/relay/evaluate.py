"""Runs every eval question under every chosen retrieval setting.

Requests run one at a time with a generous deadline, so the limiter *waits*
for per-minute capacity instead of shedding (quality runs must not be
degraded by our own load). Tight-retrieval degradation is disabled so each
row really used the setting it is labelled with.

Rows go to ``results/results.csv`` with scenario ``eval`` and the setting in
``retrieval_setting``; answers go to the sidecar for ``relay.grade``.

Typical usage example:

    uv run python -m relay.evaluate --settings s1_small_many,s2_big_few,s3_rerank          # stand-in
    uv run python -m relay.evaluate --settings s3_rerank --real --ids seed-t1,seed-u1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
import uuid

import yaml

from relay.app import apply_overrides, build
from relay.config import ROOT, load_config
from relay.limiter import Limiter
from relay.pipeline import Pipeline
from relay.results import ANSWERS_JSONL, RESULTS_CSV, ResultsWriter

# Quality runs are sequential: when a limit is hit the right move is to wait it
# out, not to give up. The retry budget protects against retry storms under
# overload and wrongly cut these runs short (first eval run), so it is relaxed here.
EVAL_SETS = ["pipeline.tight_headroom=0", "retry.max_retries=6", "retry.budget_ratio=100"]


def completed_pairs(scenario: str) -> set[tuple[str, str]]:
    """(question_id, setting) pairs that already have a non-rate-limited eval row.

    Args:
        scenario: Eval scenario label.

    Returns:
        Pairs to skip when re-running only what is missing.
    """
    if not RESULTS_CSV.exists():
        return set()
    with open(RESULTS_CSV) as f:
        rows = {r["request_id"]: r for r in csv.DictReader(f) if r["scenario"] == scenario}
    done = set()
    with open(ANSWERS_JSONL) as f:
        for line in f:
            a = json.loads(line)
            r = rows.get(a["request_id"])
            if r and r["status"] == "ok":
                done.add((a["question_id"], r["retrieval_setting"]))
    return done


async def run(
    questions: list[dict],
    settings: list[str],
    real: bool,
    sets: list[str],
    gap_s: float,
    scenario: str,
    only_missing: bool = False,
) -> None:
    """Runs the evaluation.

    Args:
        questions: Eval question records.
        settings: Retrieval setting names.
        real: Use the real service.
        sets: Extra config overrides.
        gap_s: Pause between requests (spreads load across the minute).
        scenario: Scenario label written to the CSV.
        only_missing: Skip (question, setting) pairs that already have an ok row.
    """
    base = apply_overrides(load_config(), [*EVAL_SETS, *sets])
    retrievers = Pipeline.load_retrievers(base)
    shared = Limiter(base["limits"], enabled=base.get("limiter", {}).get("enabled", True))
    systems = {}
    for s in settings:
        cfg = apply_overrides(load_config(), [*EVAL_SETS, *sets, f"retrieval.setting={s}"])
        systems[s] = build(cfg, real, retrievers=retrievers, limiter=shared)
    skip = completed_pairs(scenario) if only_missing else set()
    writer = ResultsWriter()
    total = len(questions) * len(settings)
    print(
        f"{scenario}: {len(questions)} questions x {len(settings)} settings = {total} requests "
        f"({'real' if real else 'stand-in'})"
    )
    used = 0
    try:
        for q in questions:
            for s in settings:
                if (q["id"], s) in skip:
                    continue
                system = systems[s]
                rid = f"{scenario}-{s}-{q['id']}-{uuid.uuid4().hex[:4]}"
                wall = time.time()
                o = await system.pipeline.run(q["question"], rid, wall + 180)
                writer.write(o, scenario, system.source, wall, q["id"])
                used += o.tokens_in + o.tokens_out
                print(
                    f"  {q['id']:10s} {s:14s} {o.status:12s} {o.answer_kind:10s} "
                    f"ttft={o.first_token_ms or 0:6.0f} tok={o.tokens_in + o.tokens_out:5d} "
                    f"| {o.text[:90]!r}"
                )
                await asyncio.sleep(gap_s)
    finally:
        for system in systems.values():
            await system.aclose()
        print(f"tokens used: {used}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate answer quality per retrieval setting.")
    ap.add_argument("--settings", default="s1_small_many,s2_big_few,s3_rerank")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--ids", default="")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--scenario", default="eval")
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="run only (question, setting) pairs without an ok row",
    )
    a = ap.parse_args()
    ids = [i for i in a.ids.split(",") if i]
    with open(ROOT / "eval/questions.yaml") as f:
        questions = [q for q in yaml.safe_load(f) if not ids or q["id"] in ids]
    asyncio.run(
        run(questions, a.settings.split(","), a.real, a.set, a.gap, a.scenario, a.only_missing)
    )


if __name__ == "__main__":
    main()
