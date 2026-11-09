"""Open-loop load generator: runs one scenario and appends to results.csv.

Arrival times are computed up front from the scenario and requests are fired
at those times whether or not earlier ones have finished (open loop). A
closed-loop generator would wait for replies before sending more, hiding
exactly the queueing a burst causes (coordinated omission).

Each simulated user has a patience drawn from a lognormal distribution. When
it runs out the request is cancelled and recorded as ``timeout`` (given up);
the pipeline's in-flight model calls are cancelled with it, so abandoned
requests stop spending tokens.

Scenario file keys (see ``config/scenarios/*.yaml``):

* ``name``, ``source`` (``stand-in`` | ``real``), ``seed``
* ``arrivals``: list of phases, each ``{kind: poisson, rate_per_min, from_s, to_s}``
  or ``{kind: burst, at_s, n, over_s}``
* ``questions``: ``{file, zipf_s, ids}`` (skewed popularity; optional id filter)
* ``patience``: ``{median_s, sigma, min_s, max_s}``
* ``standin``: config patch sent to the stand-in before the run
* ``set``: config overrides, ``["retry.policy=naive", ...]``
* ``max_requests``: hard cap (required for real runs)
* ``output_dir``: where rows go (default ``results``; use ``results/scratch``
  for runs that must not appear in the submitted results)

Typical usage example:

    uv run python -m relay.loadgen config/scenarios/steady.yaml
    uv run python -m relay.loadgen config/scenarios/real_steady.yaml --real
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import httpx
import yaml

from relay.app import apply_overrides, build
from relay.config import RESULTS_DIR, ROOT, load_config
from relay.pipeline import ST_TIMEOUT, Outcome
from relay.results import ResultsWriter
from relay.service import Service


def arrival_times(phases: list[dict], rng: random.Random) -> list[float]:
    """Computes request arrival offsets (seconds) for a scenario.

    Args:
        phases: Arrival phases from the scenario.
        rng: Random source.

    Returns:
        Sorted offsets from scenario start.
    """
    times: list[float] = []
    for ph in phases:
        if ph["kind"] == "poisson":
            rate_s = ph["rate_per_min"] / 60.0
            t = ph.get("from_s", 0.0)
            while True:
                t += rng.expovariate(rate_s)
                if t >= ph["to_s"]:
                    break
                times.append(t)
        elif ph["kind"] == "burst":
            times += [ph["at_s"] + rng.uniform(0, ph["over_s"]) for _ in range(ph["n"])]
        else:
            raise ValueError(f"unknown arrival kind {ph['kind']}")
    return sorted(times)


def load_questions(spec: dict) -> list[dict]:
    """Loads the scenario's question pool.

    Args:
        spec: ``questions`` section of the scenario.

    Returns:
        Question records in file order.
    """
    with open(ROOT / spec["file"]) as f:
        qs = yaml.safe_load(f)
    ids = spec.get("ids")
    return [q for q in qs if not ids or q["id"] in ids]


def zipf_picker(n: int, s: float, rng: random.Random):
    """Returns a sampler of indices in ``[0, n)`` with Zipf(s) popularity.

    The popularity order is shuffled once so the most popular question is not
    always the first in the file.

    Args:
        n: Number of items.
        s: Zipf exponent; 0 means uniform.
        rng: Random source.

    Returns:
        A zero-argument function returning an index.
    """
    order = list(range(n))
    rng.shuffle(order)
    weights = [1.0 / (r + 1) ** s for r in range(n)]
    return lambda: order[rng.choices(range(n), weights=weights)[0]]


def patience_sampler(spec: dict, rng: random.Random):
    """Returns a sampler of user patience in seconds.

    Args:
        spec: ``patience`` section of the scenario.
        rng: Random source.

    Returns:
        A zero-argument function.
    """

    def sample() -> float:
        v = spec["median_s"] * math.exp(rng.gauss(0, spec["sigma"]))
        return min(spec["max_s"], max(spec["min_s"], v))

    return sample


def run_scenario(path: Path, allow_real: bool) -> dict:
    """Runs one scenario end to end and appends its summary.

    Setup (stand-in reset and patch) and teardown (stats, summary file) are
    synchronous; only the request traffic runs on the event loop.

    Args:
        path: Scenario YAML.
        allow_real: Must be True to run a ``source: real`` scenario.

    Returns:
        Summary counts for the run.

    Raises:
        SystemExit: If a real scenario is run without ``--real`` or without a cap.
    """
    with open(path) as f:
        sc = yaml.safe_load(f)
    real = sc["source"] == "real"
    if real and not allow_real:
        raise SystemExit("this scenario uses the real service: pass --real")
    if real and not sc.get("max_requests"):
        raise SystemExit("real scenarios must set max_requests")
    cfg = apply_overrides(load_config(), sc.get("set", []))
    admin = cfg["provider"]["standin_base_url"].rsplit("/openai", 1)[0] + "/admin"
    if not real:
        httpx.post(f"{admin}/reset", timeout=5).raise_for_status()
        if sc.get("standin"):
            httpx.post(f"{admin}/config", json=sc["standin"], timeout=5).raise_for_status()

    summary = asyncio.run(_drive(sc, cfg, real))
    if not real:
        summary["standin_peak_queue"] = httpx.get(f"{admin}/stats", timeout=5).json()["peak_queued"]
    with open(RESULTS_DIR / "scenario_summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    return summary


async def _drive(sc: dict, cfg: dict, real: bool) -> dict:
    """Fires the scenario's requests on schedule and records each outcome.

    Args:
        sc: Parsed scenario.
        cfg: Configuration with scenario overrides applied.
        real: Whether to use the real service.

    Returns:
        Summary counts for the run.
    """
    rng = random.Random(sc.get("seed", 0))
    system = build(cfg, real)
    svc = Service(system.pipeline, cfg["service"])
    out_dir = ROOT / sc.get("output_dir", "results")
    writer = ResultsWriter(out_dir / "results.csv", out_dir / "answers.jsonl")
    questions = load_questions(sc["questions"])
    pick = zipf_picker(len(questions), sc["questions"].get("zipf_s", 0.0), rng)
    patience = patience_sampler(sc["patience"], rng)
    offsets = arrival_times(sc["arrivals"], rng)
    if sc.get("max_requests"):
        offsets = offsets[: sc["max_requests"]]
    plan = [(off, questions[pick()], patience()) for off in offsets]
    name = sc["name"]
    deadline_s = cfg["service"]["deadline_s"]
    span = offsets[-1] if offsets else 0
    print(f"{name}: {len(plan)} requests over {span:.0f}s ({system.source})")

    counts: dict[str, int] = {}
    start = time.perf_counter() + 0.5

    async def one(i: int, off: float, q: dict, patience_s: float) -> None:
        await asyncio.sleep(max(0.0, start + off - time.perf_counter()))
        rid = f"{name}-{i:05d}"
        wall, t0 = time.time(), time.perf_counter()
        sink: dict = {}
        task = asyncio.create_task(
            svc.handle(q["question"], rid, wall + deadline_s, t0=t0, sink=sink)
        )
        info: dict = {}
        try:
            o, info = await asyncio.wait_for(task, timeout=patience_s)
        except TimeoutError:
            o = sink.get("outcome") or Outcome(rid, q["question"])
            o.status = ST_TIMEOUT
            o.total_ms = (time.perf_counter() - t0) * 1000
            info = {"gave_up": True}
        info["patience_s"] = round(patience_s, 2)
        writer.write(o, name, system.source, wall, q["id"], info)
        counts[o.status] = counts.get(o.status, 0) + 1

    try:
        await asyncio.gather(*(one(i, off, q, p) for i, (off, q, p) in enumerate(plan)))
    finally:
        await system.aclose()
    return {
        "scenario": name,
        "source": system.source,
        "requests": len(plan),
        "duration_s": round(span, 1),
        "statuses": counts,
        "peak_admission_queue": svc.peak_queued,
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Run load scenarios.")
    ap.add_argument("scenarios", nargs="+", type=Path)
    ap.add_argument("--real", action="store_true", help="allow scenarios with source: real")
    a = ap.parse_args()
    for path in a.scenarios:
        run_scenario(path, a.real)


if __name__ == "__main__":
    main()
