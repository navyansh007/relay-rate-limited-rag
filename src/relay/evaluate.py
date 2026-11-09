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
import time
import uuid

import yaml

from relay.app import apply_overrides, build
from relay.config import ROOT, load_config
from relay.pipeline import Pipeline
from relay.results import ResultsWriter


async def run(
    questions: list[dict],
    settings: list[str],
    real: bool,
    sets: list[str],
    gap_s: float,
    scenario: str,
) -> None:
    """Runs the evaluation.

    Args:
        questions: Eval question records.
        settings: Retrieval setting names.
        real: Use the real service.
        sets: Extra config overrides.
        gap_s: Pause between requests (spreads load across the minute).
        scenario: Scenario label written to the CSV.
    """
    base = apply_overrides(load_config(), sets)
    retrievers = Pipeline.load_retrievers(base)
    systems = {}
    for s in settings:
        cfg = apply_overrides(
            load_config(), [*sets, f"retrieval.setting={s}", "pipeline.tight_headroom=0"]
        )
        systems[s] = build(cfg, real, retrievers=retrievers)
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
    a = ap.parse_args()
    ids = [i for i in a.ids.split(",") if i]
    with open(ROOT / "eval/questions.yaml") as f:
        questions = [q for q in yaml.safe_load(f) if not ids or q["id"] in ids]
    asyncio.run(run(questions, a.settings.split(","), a.real, a.set, a.gap, a.scenario))


if __name__ == "__main__":
    main()
