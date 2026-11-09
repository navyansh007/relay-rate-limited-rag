"""Retry policies for model calls, with per-request accounting.

Three policies, selected by ``retry.policy`` in the config:

* ``budgeted`` (default): retry only retryable failures (per-minute 429, 5xx,
  timeout before the first token); honour ``retry-after``; full-jitter
  exponential backoff; at most ``max_retries``; never when the remaining
  deadline cannot cover the backoff plus a typical call; and only while a
  process-wide retry budget (a fraction of first attempts) has credit.
* ``naive``: up to 5 immediate retries, ignoring ``retry-after`` and budgets.
  Exists to measure how retrying makes overload worse.
* ``none``: a single attempt.

Per-day 429s and failures after the first visible token are never retried:
waiting does not help the former, and the user has already seen the latter.

Typical usage example:

    caller = RetryingCaller(client, limiter, RetryPolicy.from_config(cfg["retry"]))
    outcome = await caller.call(messages, model, max_tokens=300, deadline_at=t + 10)
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from relay.limiter import Limiter
from relay.llm_client import OK, RETRYABLE, CallResult, LLMClient, TokenCallback


@dataclass
class RetryPolicy:
    """Retry configuration.

    Attributes:
        policy: ``"budgeted"``, ``"naive"`` or ``"none"``.
        max_retries: Retry cap for ``budgeted``.
        base_backoff_s: First backoff ceiling.
        max_backoff_s: Backoff ceiling cap.
        budget_ratio: Retries allowed per first attempt, process-wide.
    """

    policy: str = "budgeted"
    max_retries: int = 2
    base_backoff_s: float = 0.25
    max_backoff_s: float = 4.0
    budget_ratio: float = 0.2

    @classmethod
    def from_config(cls, cfg: dict) -> RetryPolicy:
        """Builds a policy from the ``retry`` config section.

        Args:
            cfg: The ``retry`` section.

        Returns:
            The policy.
        """
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})


@dataclass
class Attempt:
    """One attempt plus the time spent waiting before it.

    Attributes:
        result: The call result.
        limiter_wait_ms: Time waiting for limiter capacity before the attempt.
        backoff_ms: Backoff sleep before the attempt (0 for the first).
    """

    result: CallResult
    limiter_wait_ms: float
    backoff_ms: float


@dataclass
class CallOutcome:
    """All attempts made for one logical model call.

    Attributes:
        attempts: Attempts in order.
        final: The last attempt's result, or None if none was made.
        refused: Why no (further) attempt was made: ``"day"``, ``"deadline"``,
            ``"budget"``, ``"max_retries"``, ``"not_retryable"`` or ``""``.
        refused_wait_ms: Limiter wait before a refused (never sent) attempt.
    """

    attempts: list[Attempt] = field(default_factory=list)
    final: CallResult | None = None
    refused: str = ""
    refused_wait_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the final attempt succeeded."""
        return self.final is not None and self.final.status == OK

    @property
    def retries(self) -> int:
        """Number of attempts beyond the first."""
        return max(0, len(self.attempts) - 1)

    @property
    def tokens_in(self) -> int:
        """Prompt tokens across all attempts."""
        return sum(a.result.tokens_in for a in self.attempts)

    @property
    def tokens_out(self) -> int:
        """Completion tokens across all attempts."""
        return sum(a.result.tokens_out for a in self.attempts)


class RetryBudget:
    """Process-wide token bucket limiting retries to a fraction of attempts.

    Attributes:
        ratio: Credit earned per first attempt.
        credit: Current credit (capped at 10).
    """

    def __init__(self, ratio: float):
        """Initialises the budget.

        Args:
            ratio: Credit earned per first attempt.
        """
        self.ratio = ratio
        self.credit = 1.0

    def deposit(self) -> None:
        """Earns credit for a first attempt."""
        self.credit = min(10.0, self.credit + self.ratio)

    def withdraw(self) -> bool:
        """Spends one credit for a retry if available.

        Returns:
            True if the retry is allowed.
        """
        if self.credit >= 1.0:
            self.credit -= 1.0
            return True
        return False


class RetryingCaller:
    """Makes model calls through the limiter with a retry policy.

    Attributes:
        client: The streaming client.
        limiter: Shared limiter.
        policy: Retry policy.
        budget: Shared retry budget.
        typical_call_s: Assumed duration of a call, used to decide whether a
            retry can still finish before the deadline.
    """

    def __init__(
        self,
        client: LLMClient,
        limiter: Limiter,
        policy: RetryPolicy,
        typical_call_s: float = 1.5,
        rng: random.Random | None = None,
        expected_out_tokens: int = 200,
    ):
        """Initialises the caller.

        Args:
            client: The streaming client.
            limiter: Shared limiter.
            policy: Retry policy.
            typical_call_s: Assumed duration of one call.
            rng: Random source for jitter.
            expected_out_tokens: Completion size reserved per call at admission.
        """
        self.client = client
        self.limiter = limiter
        self.policy = policy
        self.budget = RetryBudget(policy.budget_ratio)
        self.typical_call_s = typical_call_s
        self.rng = rng or random.Random()
        self.expected_out_tokens = expected_out_tokens

    def _backoff(self, attempt_no: int, result: CallResult) -> float:
        """Computes the sleep before retry number ``attempt_no``.

        Args:
            attempt_no: 1 for the first retry.
            result: The failed attempt.

        Returns:
            Seconds to sleep.
        """
        if self.policy.policy == "naive":
            return 0.05
        ceiling = min(self.policy.max_backoff_s, self.policy.base_backoff_s * 2 ** (attempt_no - 1))
        jitter = self.rng.uniform(0, ceiling)
        return max(jitter, result.retry_after_s or 0.0)

    async def call(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        deadline_at: float,
        est_prompt_tokens: int,
        on_token: TokenCallback | None = None,
        step: str = "",
        tag: str = "",
        extra: dict | None = None,
    ) -> CallOutcome:
        """Calls the model, retrying per policy, until success or give-up.

        Args:
            messages: Chat messages.
            model: Model id.
            max_tokens: Completion cap.
            deadline_at: Unix time by which the caller needs an answer.
            est_prompt_tokens: Prompt size estimate for limiter reservation.
            on_token: Streaming callback.
            step: Pipeline step name.
            tag: Log label.
            extra: Extra request-body fields.

        Returns:
            All attempts and the reason for stopping.
        """
        out = CallOutcome()
        # Groq charges prompt + actual output, not max_tokens (measured), so
        # reserve a typical completion and let the limiter settle afterwards.
        reserve = est_prompt_tokens + min(max_tokens, self.expected_out_tokens)
        backoff = 0.0
        max_retries = {"budgeted": self.policy.max_retries, "naive": 5, "none": 0}[
            self.policy.policy
        ]
        while True:
            grant = await self.limiter.acquire(model, reserve, deadline_at)
            if not grant.ok:
                out.refused = grant.reason
                out.refused_wait_ms = grant.waited_ms
                return out
            if not out.attempts:
                self.budget.deposit()
            remaining = deadline_at - time.time()
            res = await self.client.chat(
                messages,
                model,
                max_tokens=max_tokens,
                on_token=on_token,
                tag=tag,
                step=step,
                extra=extra,
                deadline_s=remaining,
            )
            self.limiter.observe(model, res, reserve)
            out.attempts.append(Attempt(res, grant.waited_ms, backoff * 1000))
            out.final = res
            if res.status == OK:
                return out
            if res.status not in RETRYABLE or res.ttft_ms is not None:
                out.refused = "not_retryable"
                return out
            n = len(out.attempts)
            if n > max_retries:
                out.refused = "max_retries"
                return out
            backoff = self._backoff(n, res)
            if self.policy.policy == "budgeted":
                if time.time() + backoff + self.typical_call_s > deadline_at:
                    out.refused = "deadline"
                    return out
                if not self.budget.withdraw():
                    out.refused = "budget"
                    return out
            await asyncio.sleep(backoff)
