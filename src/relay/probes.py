"""Calibration probes against the real Groq service.

Every attempt (headers, status, body, timings) is appended to
``raw/groq_calls.jsonl``. Probes are deliberately small; run them one at a time
and watch the spend, because the daily budget is shared with everything else.

Typical usage example:

    uv run python -m relay.probes latency     --model openai/gpt-oss-20b --n 40 --gap 3
    uv run python -m relay.probes concurrency --model openai/gpt-oss-20b --levels 1,2,4,8
    uv run python -m relay.probes rpm         --model openai/gpt-oss-20b --n 40 --window 20
    uv run python -m relay.probes tpm         --model openai/gpt-oss-20b --prompt-tokens 2500
    uv run python -m relay.probes recover     --model openai/gpt-oss-20b --poll 1 --max-s 90
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
import uuid

from relay.config import RAW_DIR, groq_api_key, load_config
from relay.llm_client import (
    OK,
    RL_DAY,
    RL_MINUTE,
    CallLogger,
    CallResult,
    LLMClient,
    parse_duration_s,
)

TINY = [{"role": "user", "content": "Reply with exactly the word: pong"}]
# ~4 characters per token is a rough English estimate; real counts come back in usage.
FILLER = "The quick brown fox jumps over the lazy dog near the riverbank today. "


class SpendGuard:
    """Stops a probe run once it has used a configured number of tokens.

    Attributes:
        max_tokens: Token ceiling for the run.
        used: Tokens consumed so far (prompt + completion).
    """

    def __init__(self, max_tokens: int):
        """Initialises the guard.

        Args:
            max_tokens: Token ceiling for the run.
        """
        self.max_tokens = max_tokens
        self.used = 0

    def add(self, r: CallResult) -> None:
        """Records the tokens consumed by one attempt.

        Args:
            r: The completed attempt.
        """
        self.used += r.tokens_in + r.tokens_out

    def check(self) -> None:
        """Aborts the run if the ceiling has been reached.

        Raises:
            SystemExit: If ``used`` is at or above ``max_tokens``.
        """
        if self.used >= self.max_tokens:
            raise SystemExit(f"spend guard: {self.used} tokens used (limit {self.max_tokens})")


def _client(cfg: dict) -> LLMClient:
    """Builds a real-service client with generous timeouts.

    Probes want to *observe* slow calls rather than cut them off, so the
    first-token and stall timeouts are much longer than in the pipeline.

    Args:
        cfg: Loaded configuration.

    Returns:
        A client that logs to ``raw/groq_calls.jsonl``.
    """
    p = cfg["provider"]
    return LLMClient(
        p["real_base_url"],
        api_key=groq_api_key(),
        source="real",
        connect_timeout_s=p["connect_timeout_s"],
        first_token_timeout_s=30.0,
        stall_timeout_s=30.0,
        logger=CallLogger(RAW_DIR / "groq_calls.jsonl"),
    )


def _extra(a: argparse.Namespace) -> dict | None:
    """Request extras for reasoning models.

    Args:
        a: Parsed CLI arguments.

    Returns:
        ``{"reasoning_effort": ...}`` or None.
    """
    return {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else None


def _pct(xs: list[float], p: float) -> float:
    """Returns the nearest-rank percentile of an already sorted list.

    Args:
        xs: Sorted, non-empty values.
        p: Percentile as a fraction in [0, 1].

    Returns:
        The selected value.
    """
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def _summary(name: str, results: list[CallResult]) -> None:
    """Prints a short latency/status summary for a batch of attempts.

    Args:
        name: Label for the batch.
        results: Attempts to summarise; must be non-empty.
    """
    ok = [r for r in results if r.status == OK]
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print(f"\n== {name}: n={len(results)} statuses={by_status}")
    if ok:
        ttft = sorted(r.ttft_ms or 0 for r in ok)
        tot = sorted(r.total_ms for r in ok)
        print(
            f"   ttft ms  p50={_pct(ttft, 0.5):.0f} p95={_pct(ttft, 0.95):.0f} max={ttft[-1]:.0f}"
        )
        print(f"   total ms p50={_pct(tot, 0.5):.0f} p95={_pct(tot, 0.95):.0f} max={tot[-1]:.0f}")
        print(
            f"   mean tokens in/out {statistics.mean(r.tokens_in for r in ok):.0f}"
            f"/{statistics.mean(r.tokens_out for r in ok):.0f}"
        )
    print(f"   last headers: {results[-1].headers}")
    for r in results:
        if r.status != OK:
            print(
                f"   first non-ok: {r.status} http={r.http_status} kind={r.limit_kind} "
                f"retry_after={r.retry_after_s} body={(r.error or '')[:300]}"
            )
            break


async def probe_latency(cli: LLMClient, a: argparse.Namespace, guard: SpendGuard, run: str) -> None:
    """Sends spaced-out sequential tiny requests to measure baseline latency.

    Args:
        cli: Real-service client.
        a: Parsed CLI arguments (uses ``n``, ``gap``, ``model``, ``max_tokens``).
        guard: Spend guard for this run.
        run: Run id used in attempt tags.
    """
    out = []
    for i in range(a.n):
        guard.check()
        r = await cli.chat(
            TINY, a.model, max_tokens=a.max_tokens, extra=_extra(a), tag=f"{run}:latency:{i}"
        )
        guard.add(r)
        out.append(r)
        print(
            f"  {i:3d} {r.status:20s} ttft={r.ttft_ms or 0:7.0f} total={r.total_ms:7.0f} "
            f"rem_tok={r.headers.get('x-ratelimit-remaining-tokens')} "
            f"rem_req={r.headers.get('x-ratelimit-remaining-requests')}"
        )
        await asyncio.sleep(a.gap)
    _summary("latency", out)


async def probe_concurrency(
    cli: LLMClient, a: argparse.Namespace, guard: SpendGuard, run: str
) -> None:
    """Fires k simultaneous requests per level to look for an in-flight cap.

    Args:
        cli: Real-service client.
        a: Parsed CLI arguments (uses ``levels``, ``gap``, ``model``, ``max_tokens``).
        guard: Spend guard for this run.
        run: Run id used in attempt tags.
    """
    for k in [int(x) for x in a.levels.split(",")]:
        guard.check()
        rs = await asyncio.gather(
            *[
                cli.chat(
                    TINY,
                    a.model,
                    max_tokens=a.max_tokens,
                    extra=_extra(a),
                    tag=f"{run}:conc{k}:{i}",
                )
                for i in range(k)
            ]
        )
        for r in rs:
            guard.add(r)
        _summary(f"concurrency={k}", list(rs))
        await asyncio.sleep(a.gap)


async def probe_rpm(cli: LLMClient, a: argparse.Namespace, guard: SpendGuard, run: str) -> None:
    """Spreads n tiny requests over a window to find the RPM wall and the 429 shape.

    Args:
        cli: Real-service client.
        a: Parsed CLI arguments (uses ``n``, ``window``, ``model``, ``max_tokens``).
        guard: Spend guard for this run.
        run: Run id used in attempt tags.
    """
    interval = a.window / a.n
    tasks = []
    for i in range(a.n):
        guard.check()
        tasks.append(
            asyncio.create_task(
                cli.chat(
                    TINY, a.model, max_tokens=a.max_tokens, extra=_extra(a), tag=f"{run}:rpm:{i}"
                )
            )
        )
        await asyncio.sleep(interval)
    rs = await asyncio.gather(*tasks)
    for r in rs:
        guard.add(r)
    for i, r in enumerate(rs):
        print(
            f"  {i:3d} t={r.started_at:.1f} {r.status:20s} kind={r.limit_kind} "
            f"retry_after={r.retry_after_s}"
        )
    _summary("rpm", list(rs))


async def probe_tpm(cli: LLMClient, a: argparse.Namespace, guard: SpendGuard, run: str) -> None:
    """Sends large prompts back to back to see which bucket trips first.

    Args:
        cli: Real-service client.
        a: Parsed CLI arguments (uses ``prompt_tokens``, ``n``, ``model``, ``max_tokens``).
        guard: Spend guard for this run.
        run: Run id used in attempt tags.
    """
    reps = max(1, a.prompt_tokens * 4 // len(FILLER))
    msgs = [{"role": "user", "content": FILLER * reps + "\nReply with: ok"}]
    out = []
    for i in range(a.n):
        guard.check()
        r = await cli.chat(
            msgs, a.model, max_tokens=a.max_tokens, extra=_extra(a), tag=f"{run}:tpm:{i}"
        )
        guard.add(r)
        out.append(r)
        print(
            f"  {i} {r.status:20s} in={r.tokens_in} kind={r.limit_kind} "
            f"retry_after={r.retry_after_s} headers={r.headers}"
        )
    _summary("tpm", out)


async def probe_recover(cli: LLMClient, a: argparse.Namespace, guard: SpendGuard, run: str) -> None:
    """Polls with tiny calls after a limit is hit, measuring time to recovery.

    Run this immediately after ``rpm`` or ``tpm`` has produced 429s.

    Args:
        cli: Real-service client.
        a: Parsed CLI arguments (uses ``poll``, ``max_s``, ``model``, ``max_tokens``).
        guard: Spend guard for this run.
        run: Run id used in attempt tags.
    """
    t0 = time.time()
    first_fail = None
    i = 0
    while time.time() - t0 < a.max_s:
        guard.check()
        r = await cli.chat(
            TINY, a.model, max_tokens=a.max_tokens, extra=_extra(a), tag=f"{run}:recover:{i}"
        )
        guard.add(r)
        i += 1
        if r.status != OK and first_fail is None:
            first_fail = time.time()
            print(
                f"  limited: {r.status} retry_after={r.retry_after_s} body={(r.error or '')[:200]}"
            )
        if r.status == OK and first_fail is not None:
            print(f"  recovered {time.time() - first_fail:.1f}s after first observed 429")
            return
        if r.status == OK and first_fail is None:
            print("  not currently limited; run right after `rpm` or `tpm`")
            return
        await asyncio.sleep(a.poll)
    print("  did not recover within max-s")


async def probe_day(cli: LLMClient, a: argparse.Namespace, guard: SpendGuard, run: str) -> None:
    """Spends one model's daily token budget to observe the real daily limit.

    Sends large prompts paced by the per-minute bucket (waiting out
    ``retry-after`` on minute 429s) until a daily 429 arrives or ``--max-total``
    tokens are spent. Prints the daily 429 body and headers, which no
    documentation shows. Run it last on a day: it leaves the model with
    almost no budget until the bucket refills.

    Args:
        cli: Real-service client.
        a: Parsed CLI arguments (uses ``prompt_tokens``, ``model``, ``max_total``).
        guard: Spend guard for this run.
        run: Run id used in attempt tags.
    """
    reps = max(1, a.prompt_tokens * 4 // len(FILLER))
    msgs = [{"role": "user", "content": FILLER * reps + "\nReply with: ok"}]
    spent, i, t0 = 0, 0, time.time()
    while spent < a.max_total:
        guard.check()
        r = await cli.chat(msgs, a.model, max_tokens=16, extra=_extra(a), tag=f"{run}:day:{i}")
        guard.add(r)
        spent += r.tokens_in + r.tokens_out
        i += 1
        elapsed = time.time() - t0
        if r.status == RL_DAY:
            print(f"  DAILY LIMIT after {spent} tokens, {i} calls, {elapsed:.0f}s")
            print(f"  body: {r.error}")
            print(f"  headers: {r.headers}")
            return
        if r.status == RL_MINUTE:
            await asyncio.sleep(r.retry_after_s or 5.0)
            continue
        print(
            f"  {i:4d} {r.status:8s} spent={spent:7d} t={elapsed:5.0f}s "
            f"rem_tok={r.headers.get('x-ratelimit-remaining-tokens')} "
            f"rem_req={r.headers.get('x-ratelimit-remaining-requests')}"
        )
        reset = parse_duration_s(r.headers.get("x-ratelimit-reset-tokens")) or 0.0
        if float(r.headers.get("x-ratelimit-remaining-tokens", 1e9)) < a.prompt_tokens * 1.2:
            await asyncio.sleep(reset)
    print(f"  stopped at --max-total ({spent} tokens) without a daily 429")


PROBES = {
    "latency": probe_latency,
    "concurrency": probe_concurrency,
    "rpm": probe_rpm,
    "tpm": probe_tpm,
    "recover": probe_recover,
    "day": probe_day,
}


async def amain() -> None:
    """Parses CLI arguments and runs the selected probe."""
    ap = argparse.ArgumentParser(description="Probe the real Groq service.")
    ap.add_argument("probe", choices=PROBES)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--prompt-tokens", type=int, default=2500)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--reasoning-effort", default="low", help="gpt-oss / qwen reasoning effort; '' to omit"
    )
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=90.0)
    ap.add_argument("--max-total", type=int, default=0, help="day probe: token ceiling")
    a = ap.parse_args()
    cfg = load_config()
    guard = SpendGuard(max(cfg["spend_guard"]["max_real_tokens_per_run"], a.max_total))
    run = f"probe-{uuid.uuid4().hex[:6]}"
    cli = _client(cfg)
    try:
        await PROBES[a.probe](cli, a, guard, run)
    finally:
        await cli.aclose()
        print(f"\nrun={run} tokens used this run: {guard.used}")


if __name__ == "__main__":
    asyncio.run(amain())
