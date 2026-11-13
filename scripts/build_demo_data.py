"""Bakes recorded real-service runs into the UI's replay file.

The demo UI can replay requests that really ran against Groq, with their
recorded timings, retrieved passages and rate-limit headers, so it works with
no API key and no index. This script collects those runs from the raw logs
and writes ``src/relay/ui/static/demo.json``.

Typical usage example:

    uv run python scripts/build_demo_data.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/relay/ui/static/demo.json"

# (request_id, short label, what it shows)
SESSIONS = [
    ("real_burst-00003", "Mimi frame rate", "A cited answer"),
    ("eval-s2_big_few-a07-9b18", "Mimi quantizers", "Two facts, two citations"),
    ("eval-s2_big_few-a08-bb3f", "Helium size", "Numbers checked against the source"),
    ("real_burst-00002", "H-Net ratio loss", "A mechanism, explained from §2.2.3"),
    ("real_steady-00010", "Moshi H100 count", "The papers do not say, so it refuses"),
    ("real_burst-00011", "Warmup steps, in a burst", "12 s waiting for the token budget"),
    ("real_burst-00006", "Training cost, in a burst", "Over budget: quoted sources, at once"),
]
PIECES = {"s1_small_many": (200, 40), "s2_big_few": (500, 80), "s3_rerank": (300, 60)}
PIECES["s4_tight"] = (500, 80)


def _jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _cite(c: dict) -> str:
    sec = "abstract" if c["section"] == "abstract" else f"§{c['section']}"
    return f"{c['paper']} {sec} p.{c['page']}"


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def main() -> None:
    """Collects sessions, the real burst and headline numbers."""
    answers = {a["request_id"]: a for a in _jsonl(ROOT / "results/answers.jsonl")}
    traces = {}
    for t in _jsonl(ROOT / "raw/traces.jsonl"):
        traces[t["request_id"]] = t  # last write wins (reruns relabel old ids)
    calls = _jsonl(ROOT / "raw/groq_calls.jsonl")
    with open(ROOT / "results/results.csv") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["request_id"]: r for r in rows}
    chunks: dict[str, dict] = {}
    for piece, overlap in set(PIECES.values()):
        path = ROOT / f"data/index/chunks_{piece}_{overlap}.jsonl"
        for c in _jsonl(path):
            chunks[c["chunk_id"]] = c

    sessions = []
    for rid, label, blurb in SESSIONS:
        t, a, r = traces[rid], answers[rid], by_id[rid]
        hits = next(
            (s["attrs"].get("hits", []) for s in t["spans"] if s["name"].startswith("retrieve")), []
        )
        passages = [
            {"cite": _cite(chunks[h]), "text": chunks[h]["text"][:520]} for h in hits if h in chunks
        ]
        # rate-limit headers seen by this request's calls: remaining tokens per model
        own = [c for c in calls if c["tag"].startswith(rid + ":")]
        budget = [
            {
                "model": c["model"].split("/")[-1],
                "step": c["tag"].split(":")[-1],
                "remaining": int(float(c["headers"].get("x-ratelimit-remaining-tokens", 0))),
                "limit": int(float(c["headers"].get("x-ratelimit-limit-tokens", 8000))),
            }
            for c in own
            if c.get("headers")
        ]
        sessions.append(
            {
                "id": rid,
                "label": label,
                "blurb": blurb,
                "question": a["question"],
                "source": "real Groq",
                "setting": r["retrieval_setting"],
                "status": r["status"],
                "answer_kind": a["answer_kind"],
                "text": a["text"],
                "first_token_ms": float(r["first_token_ms"]) if r["first_token_ms"] else None,
                "total_ms": float(r["total_ms"]),
                "tokens_in": int(r["tokens_in"]),
                "tokens_out": int(r["tokens_out"]),
                "evidence": (t["attrs"].get("evidence") or [None])[0],
                "understanding": t["attrs"].get("understanding", {}),
                "degraded": t["attrs"].get("degraded", []),
                "held": len(t["attrs"].get("held_sentences", []) or []),
                "spans": t["spans"],
                "totals": t["totals"],
                "passages": passages,
                "budget": budget,
            }
        )

    # the real 12-question burst, one lane per request
    burst_rows = sorted(
        (r for r in rows if r["scenario"] == "real_burst"), key=lambda r: float(r["start_time"])
    )
    t0 = float(burst_rows[0]["start_time"])
    lanes = []
    for r in burst_rows:
        tr = traces.get(r["request_id"], {"spans": []})
        lanes.append(
            {
                "id": r["request_id"],
                "question": answers.get(r["request_id"], {}).get("question", ""),
                "start_ms": round((float(r["start_time"]) - t0) * 1000, 1),
                "total_ms": float(r["total_ms"]),
                "first_token_ms": float(r["first_token_ms"]) if r["first_token_ms"] else None,
                "status": r["status"],
                "answer_kind": answers.get(r["request_id"], {}).get("answer_kind", "none"),
                "spans": [
                    {k: s[k] for k in ("kind", "name", "start_ms", "end_ms")} for s in tr["spans"]
                ],
            }
        )
    tpm = sorted(
        (
            round((c["started_at"] + c["total_ms"] / 1000 - t0) * 1000),
            int(float(c["headers"]["x-ratelimit-remaining-tokens"])),
        )
        for c in calls
        if c["tag"].startswith("real_burst")
        and c["model"].endswith("120b")
        and "x-ratelimit-remaining-tokens" in c.get("headers", {})
    )

    def outcome(name: str) -> dict:
        rs = [r for r in rows if r["scenario"] == name]
        n = len(rs)
        return {
            "scenario": name,
            "source": rs[0]["source"],
            "n": n,
            "answered": sum(r["status"] == "ok" for r in rs),
            "quoted": sum(r["status"] == "rate_limited" for r in rs),
            "gave_up": sum(r["status"] == "timeout" for r in rs),
        }

    steady_ok = [
        float(r["first_token_ms"])
        for r in rows
        if r["scenario"] == "real_steady" and r["status"] == "ok"
    ]
    graded = [r for r in rows if r["scenario"] == "eval" and r["answer_grade"]]
    unans = [r for r in graded if r["answer_grade"].startswith("refused_correctly")]
    stats = {
        "tpm": 8000,
        "tpd": 200000,
        "answers_per_min": 3.7,
        "steady_ttft_p50": round(_pct(steady_ok, 50)),
        "steady_ttft_p95": round(_pct(steady_ok, 95)),
        "refused_correctly": len(unans),
        "unanswerable_asked": 32,
        "real_429s_in_burst": sum(
            1 for c in calls if c["tag"].startswith("real_burst") and c["status"] != "ok"
        ),
        "outcomes": [
            outcome(s)
            for s in (
                "real_steady",
                "real_burst",
                "steady",
                "burst",
                "burst_cache",
                "overload",
                "rpm_halved",
                "day_exhausted",
            )
        ],
    }
    with open(ROOT / "eval/questions.yaml") as f:
        import yaml

        examples = [{"id": q["id"], "question": q["question"]} for q in yaml.safe_load(f)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "sessions": sessions,
                "burst": {"lanes": lanes, "tpm": tpm},
                "stats": stats,
                "examples": examples,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"wrote {OUT.relative_to(ROOT)}: {len(sessions)} sessions, {len(lanes)} burst lanes")


if __name__ == "__main__":
    main()
