"""Builds the results tables from ``results/results.csv`` alone.

Every number is recomputable from the CSV. Definitions
(repeated under the tables in the output):

* Percentiles are nearest-rank over rows with ``status == ok`` and a
  non-empty ``first_token_ms``; ``n`` is that row count.
* Duration is from the first arrival to the last completion.
* Offered rate is requests divided by the arrival window, per minute.
* "Timed out or given up" is ``status == timeout``.
* Peak queue length is the peak number of requests in the system at once,
  computed by sweeping ``[start_time, start_time + total_ms]`` intervals.
* Tokens per answer is mean ``tokens_in + tokens_out`` over ``ok`` rows.
* Cost per 1,000 answers prices all tokens of ``ok`` rows at the draft
  model's list price (an upper bound: understand/verify calls use a cheaper
  model) and also states that the free tier actually cost $0.

Typical usage example:

    uv run python -m relay.analysis                  # tables 2a, 2b, 3 as markdown
    uv run python -m relay.analysis --csv other.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from relay.config import load_config
from relay.results import RESULTS_CSV

# Core shapes first (steady, burst, overload, halved, day), then variants.
ROW_ORDER = [
    "real_steady",
    "steady",
    "real_burst",
    "burst",
    "burst_cache",
    "overload",
    "overload_deadline8",
    "overload_nolimiter",
    "overload_naive",
    "rpm_halved",
    "day_exhausted",
]


def pct(xs: list[float], p: float) -> float | None:
    """Nearest-rank percentile.

    Args:
        xs: Values.
        p: Percentile in (0, 100].

    Returns:
        The value at rank ``ceil(p/100 * n)``, or None for an empty list.
    """
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def peak_in_system(rows: list[dict]) -> int:
    """Peak number of overlapping requests.

    Args:
        rows: CSV rows of one scenario.

    Returns:
        Maximum concurrency.
    """
    events = []
    for r in rows:
        s = float(r["start_time"])
        events += [(s, 1), (s + float(r["total_ms"]) / 1000, -1)]
    cur = peak = 0
    for _, d in sorted(events, key=lambda e: (e[0], e[1])):
        cur += d
        peak = max(peak, cur)
    return peak


def _fmt3(vals: list[float | None]) -> str:
    return " / ".join("—" if v is None else f"{v:.0f}" for v in vals)


def _pc(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}" if d else "not measured"


def scenario_rows(rows: list[dict], price: tuple[float, float]) -> tuple[list[str], list[str]]:
    """Formats Table 2a and 2b rows, one per (scenario, source).

    Args:
        rows: All CSV rows.
        price: Draft model price per 1M tokens (input, output).

    Returns:
        Markdown lines for table 2a and 2b.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["scenario"].startswith("eval"):
            continue
        groups[(r["scenario"], r["source"])].append(r)
    a, b = [], []
    order = {name: i for i, name in enumerate(ROW_ORDER)}
    for (scen, src), rs in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0])):
        starts = [float(r["start_time"]) for r in rs]
        ends = [float(r["start_time"]) + float(r["total_ms"]) / 1000 for r in rs]
        window = max(starts) - min(starts)
        duration = max(ends) - min(starts)
        rate = len(rs) / window * 60 if window > 0 else float("nan")
        ok = [r for r in rs if r["status"] == "ok"]
        ft = [float(r["first_token_ms"]) for r in ok if r["first_token_ms"]]
        tt = [float(r["total_ms"]) for r in ok]
        a.append(
            f"| {scen} | {src} | {len(rs)} | {duration:.0f} | {rate:.1f} | "
            f"{_fmt3([pct(ft, 50), pct(ft, 95), pct(ft, 99)])} (n={len(ft)}) | "
            f"{_fmt3([pct(tt, 50), pct(tt, 95), pct(tt, 99)])} (n={len(tt)}) |"
        )
        n = len(rs)
        count = lambda st, rs=rs: sum(r["status"] == st for r in rs)
        retries = sum(int(r["retries"]) for r in rs) / n
        tok = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in ok]
        cost = (
            1000
            * sum(int(r["tokens_in"]) * price[0] + int(r["tokens_out"]) * price[1] for r in ok)
            / 1e6
            / len(ok)
            if ok
            else None
        )
        b.append(
            f"| {scen} | {_pc(count('ok'), n)} | {_pc(count('rate_limited'), n)} | "
            f"{_pc(count('timeout'), n)} | {retries:.2f} | {peak_in_system(rs)} | "
            f"{(sum(tok) / len(tok)) if tok else 0:.0f} | "
            + (f"${cost:.2f} (free tier: $0)" if cost is not None else "not measured")
            + " |"
        )
    return a, b


def retrieval_rows(rows: list[dict]) -> list[str]:
    """Formats Table 3 from graded eval rows, one per retrieval setting.

    Args:
        rows: All CSV rows.

    Returns:
        Markdown lines.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["scenario"].startswith("eval") and r["answer_grade"]:
            groups[(r["retrieval_setting"], r["source"])].append(r)
    cfg = load_config()["retrieval"]["settings"]
    out = []
    for (setting, src), rs in sorted(groups.items()):
        s = cfg.get(setting, {})
        label = (
            f"{setting}: {s.get('piece_tokens')} tok, {s.get('fetch')} pieces, "
            f"re-rank {'yes' if s.get('rerank') else 'no'} ({src})"
        )
        n = len(rs)
        g = lambda x, rs=rs: sum(r["answer_grade"] == x for r in rs)
        ok = [r for r in rs if r["status"] == "ok"]
        tok = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in rs]
        ft = [float(r["first_token_ms"]) for r in ok if r["first_token_ms"]]
        tt = [float(r["total_ms"]) for r in ok]
        out.append(
            f"| {label} | {n} | {_pc(g('correct'), n)} | {_pc(g('wrong'), n)} | "
            f"{_pc(g('refused_correctly'), n)} | {sum(tok) / n:.0f} | "
            f"{_fmt3([pct(ft, 50), pct(ft, 95)])} | {_fmt3([pct(tt, 50), pct(tt, 95)])} |"
        )
    return out


def main() -> None:
    """Prints Tables 2a, 2b and 3 as markdown."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=RESULTS_CSV)
    a = ap.parse_args()
    with open(a.csv) as f:
        rows = list(csv.DictReader(f))
    cfg = load_config()
    draft = cfg["models"]["draft"]
    price = tuple(cfg["prices"][draft])
    t2a, t2b = scenario_rows(rows, price)
    print("### Table 2a: Latency\n")
    print(
        "| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) "
        "| First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |"
    )
    print("|---|---|---|---|---|---|---|")
    print("\n".join(t2a))
    print("\n### Table 2b: Outcomes\n")
    print(
        "| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | "
        "Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |"
    )
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(t2b))
    print("\n### Table 3: Retrieval settings\n")
    print(
        "| Setting (piece size, pieces fetched, re-rank) | Questions asked | Correct and cited % "
        "| Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) "
        "| Total time p50 / p95 (ms) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(retrieval_rows(rows)) or "| not measured | | | | | | | |")
    print(
        f"\nPercentiles: nearest-rank over status=ok rows (n shown). Cost: all tokens at "
        f"{draft} list price ${price[0]}/${price[1]} per 1M in/out (upper bound). "
        "Peak queue length: peak requests in system, from start_time + total_ms."
    )


if __name__ == "__main__":
    main()
