"""Client-side view of the provider's rate limits, per model.

The limiter keeps its own estimate of each model's RPM, TPM, RPD and TPD
budgets and corrects it from every response, because the provider's limits
"can change without notice":

* ``x-ratelimit-limit-tokens`` / ``-requests`` replace the configured TPM / RPD.
* ``x-ratelimit-remaining-tokens`` / ``-requests`` pull the local estimate
  down when the server knows better (e.g. another process shares the key).
* A per-minute 429 blocks the model until ``retry-after``; a per-day 429
  marks it exhausted until the daily reset.

Callers ``acquire`` before every model call. Waiting is FIFO per model, and a
caller that cannot be served before its deadline is told so immediately
instead of queueing pointlessly (load shedding).

Typical usage example:

    limiter = Limiter(cfg["limits"])
    grant = await limiter.acquire(model, est_tokens=900, deadline_at=time.time() + 8)
    if grant.ok:
        result = await client.chat(...)
        limiter.observe(model, result, reserved=900)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from relay.llm_client import RL_DAY, RL_MINUTE, CallResult, parse_duration_s

DAY_S = 86400.0


@dataclass
class Grant:
    """Result of an ``acquire`` call.

    Attributes:
        ok: Whether the caller may make the call now.
        waited_ms: Time spent waiting for capacity.
        reason: Why the grant was refused (``"day"``, ``"deadline"``), or ``""``.
    """

    ok: bool
    waited_ms: float
    reason: str = ""


@dataclass
class ModelBudget:
    """Local estimate of one model's remaining budget.

    Attributes:
        rpm: Requests per minute.
        tpm: Tokens per minute.
        rpd: Requests per day.
        tpd: Tokens per day.
        tokens: Current TPM bucket level.
        updated: Last refill time.
        req_used: Requests counted against RPM; drains at ``rpm/60`` per second.
        day_requests: Requests counted against RPD; drains at ``rpd/86400`` per second
            (measured: Groq's daily budget refills continuously, not at midnight).
        day_tokens: Tokens counted against TPD; assumed to drain the same way.
        blocked_until: No calls before this time (from a per-minute 429).
        day_blocked_until: Model exhausted for the day until this time.
        lock: FIFO queue of waiters for this model.
    """

    rpm: int
    tpm: int
    rpd: int
    tpd: int
    tokens: float = 0.0
    updated: float = field(default_factory=time.time)
    req_used: float = 0.0
    day_requests: float = 0.0
    day_tokens: float = 0.0
    blocked_until: float = 0.0
    day_blocked_until: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def refill(self, now: float) -> None:
        """Advances all four buckets to ``now``.

        Args:
            now: Current unix time.
        """
        dt = max(0.0, now - self.updated)
        self.tokens = min(self.tpm, self.tokens + dt * self.tpm / 60.0)
        self.req_used = max(0.0, self.req_used - dt * self.rpm / 60.0)
        self.day_requests = max(0.0, self.day_requests - dt * self.rpd / 86400.0)
        self.day_tokens = max(0.0, self.day_tokens - dt * self.tpd / 86400.0)
        self.updated = now

    def wait_needed(self, now: float, est_tokens: int) -> float:
        """Seconds until a call of ``est_tokens`` fits every per-minute limit.

        Args:
            now: Current unix time.
            est_tokens: Tokens the call is expected to consume.

        Returns:
            Zero if the call may go now.
        """
        waits = [self.blocked_until - now]
        if self.req_used + 1 > self.rpm:
            waits.append((self.req_used + 1 - self.rpm) * 60.0 / self.rpm)
        need = min(est_tokens, self.tpm) - self.tokens
        if need > 0:
            waits.append(need * 60.0 / self.tpm)
        return max(0.0, *waits)

    def day_exhausted(self, now: float, est_tokens: int = 0) -> bool:
        """Whether the daily budget is (known to be) spent.

        Args:
            now: Current unix time.
            est_tokens: Tokens the next call would add.

        Returns:
            True if no more calls should be attempted today.
        """
        return (
            now < self.day_blocked_until
            or self.day_requests + 1 > self.rpd
            or self.day_tokens + est_tokens > self.tpd
        )


class Limiter:
    """Per-model budgets shared by every request in the process.

    Attributes:
        enabled: When False, ``acquire`` always grants immediately (used to
            show what happens without client-side limiting).
        budgets: Per-model state.
    """

    def __init__(self, limits: dict[str, dict], enabled: bool = True):
        """Initialises budgets from configured limits.

        Args:
            limits: ``{model: {rpm, tpm, rpd, tpd}}``.
            enabled: Whether to enforce limits client-side.
        """
        self.enabled = enabled
        self._limits = limits
        self.budgets: dict[str, ModelBudget] = {}

    def budget(self, model: str) -> ModelBudget:
        """Returns (creating if needed) the budget for a model.

        Args:
            model: Model id.

        Returns:
            The model's budget.
        """
        if model not in self.budgets:
            lim = self._limits.get(model) or {"rpm": 30, "tpm": 6000, "rpd": 1000, "tpd": 100000}
            self.budgets[model] = ModelBudget(
                lim["rpm"], lim["tpm"], lim["rpd"], lim["tpd"], tokens=lim["tpm"]
            )
        return self.budgets[model]

    async def acquire(self, model: str, est_tokens: int, deadline_at: float) -> Grant:
        """Waits (FIFO) until a call fits the per-minute limits, then charges it.

        Args:
            model: Model id.
            est_tokens: Expected tokens (prompt estimate + max completion).
            deadline_at: Unix time after which waiting is pointless.

        Returns:
            A grant; ``ok`` is False when the day is exhausted or the caller
            would not be served before its deadline.
        """
        t0 = time.time()
        if not self.enabled:
            return Grant(True, 0.0)
        b = self.budget(model)
        if b.day_exhausted(t0, est_tokens):
            return Grant(False, 0.0, "day")
        try:
            await asyncio.wait_for(b.lock.acquire(), timeout=max(0.0, deadline_at - t0))
        except TimeoutError:
            return Grant(False, (time.time() - t0) * 1000, "deadline")
        try:
            while True:
                now = time.time()
                b.refill(now)
                if b.day_exhausted(now, est_tokens):
                    return Grant(False, (now - t0) * 1000, "day")
                wait = b.wait_needed(now, est_tokens)
                if wait <= 0:
                    b.tokens -= est_tokens
                    b.req_used += 1
                    b.day_requests += 1
                    b.day_tokens += est_tokens
                    return Grant(True, (now - t0) * 1000)
                if now + wait > deadline_at:
                    return Grant(False, (now - t0) * 1000, "deadline")
                await asyncio.sleep(min(wait, 0.5))
        finally:
            b.lock.release()

    def estimated_wait_s(self, model: str, est_tokens: int) -> float:
        """Estimates how long a new call would wait, ignoring queued callers.

        Args:
            model: Model id.
            est_tokens: Expected tokens.

        Returns:
            Seconds; ``inf`` if the day is exhausted.
        """
        b = self.budget(model)
        now = time.time()
        b.refill(now)
        if b.day_exhausted(now, est_tokens):
            return float("inf")
        return b.wait_needed(now, est_tokens)

    def headroom(self, model: str) -> float:
        """Fraction of the per-minute token bucket currently available.

        Args:
            model: Model id.

        Returns:
            Value in [0, 1]; 0 when blocked or exhausted.
        """
        b = self.budget(model)
        now = time.time()
        b.refill(now)
        if b.day_exhausted(now) or now < b.blocked_until:
            return 0.0
        return max(0.0, b.tokens / b.tpm)

    def day_exhausted(self, model: str) -> bool:
        """Whether a model's daily budget is known to be spent.

        Args:
            model: Model id.

        Returns:
            True if exhausted.
        """
        return self.budget(model).day_exhausted(time.time())

    def day_reset_in_s(self, model: str) -> float | None:
        """Seconds until a day-blocked model is expected to reset.

        Args:
            model: Model id.

        Returns:
            Seconds, or None if not day-blocked.
        """
        left = self.budget(model).day_blocked_until - time.time()
        return left if left > 0 else None

    def observe(self, model: str, result: CallResult, reserved: int) -> None:
        """Corrects the local estimate from a completed attempt.

        Args:
            model: Model id.
            result: The attempt, including response headers.
            reserved: Tokens charged at ``acquire`` time for this attempt.
        """
        b = self.budget(model)
        now = time.time()
        h = result.headers
        actual = result.tokens_in + result.tokens_out
        if result.http_status == 200 and actual:
            delta = reserved - actual  # refund over-reservation
            b.tokens = min(b.tpm, b.tokens + delta)
            b.day_tokens = max(0, b.day_tokens - delta)
        elif result.http_status is not None and result.http_status != 200:
            # Rejected or failed before generation: refund the reservation.
            b.tokens = min(b.tpm, b.tokens + reserved)
            b.day_tokens = max(0, b.day_tokens - reserved)
        if "x-ratelimit-limit-tokens" in h:
            b.tpm = int(float(h["x-ratelimit-limit-tokens"]))
        if "x-ratelimit-limit-requests" in h:
            b.rpd = int(float(h["x-ratelimit-limit-requests"]))
        if "x-ratelimit-remaining-tokens" in h:
            b.tokens = min(b.tokens, float(h["x-ratelimit-remaining-tokens"]))
        if "x-ratelimit-remaining-requests" in h:
            b.day_requests = max(
                b.day_requests, b.rpd - int(float(h["x-ratelimit-remaining-requests"]))
            )
        if result.status == RL_MINUTE:
            b.blocked_until = max(b.blocked_until, now + (result.retry_after_s or 1.0))
            if "x-ratelimit-remaining-tokens" not in h:
                # No server figure: assume the bucket is empty. With the header
                # present it is authoritative, and zeroing as well would make the
                # limiter wait out a refill on top of the retry-after backoff.
                b.tokens = min(b.tokens, 0.0)
        elif result.status == RL_DAY:
            reset = parse_duration_s(h.get("x-ratelimit-reset-requests")) or result.retry_after_s
            b.day_blocked_until = now + (reset or DAY_S)
