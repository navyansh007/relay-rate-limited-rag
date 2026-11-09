"""Front door of the assistant: admission control, coalescing and caching.

* **Admission**: at most ``max_concurrent`` requests run the pipeline at once;
  the rest wait in a bounded FIFO queue. A request that arrives to a full
  queue is turned away immediately (``rate_limited``) instead of waiting for
  an answer it will give up on.
* **Coalescing**: identical questions in flight at the same time share one
  pipeline run. Bursts follow incidents, so many people ask the same thing.
* **Cache**: completed answers (not refusals caused by limits) are reused
  for ``cache_ttl_s``. A cache hit costs no tokens.

Typical usage example:

    svc = Service(system.pipeline, cfg["service"])
    outcome, info = await svc.handle(question, rid, deadline_at, t0=arrival)
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace

from relay.pipeline import ST_OK, ST_RATE_LIMITED, Outcome, Pipeline, TextCallback


class LeaderGone(Exception):
    """The request that owned a coalesced pipeline run gave up before it finished."""


def normalise(question: str) -> str:
    """Canonical form of a question for cache and coalescing keys.

    Args:
        question: Raw question.

    Returns:
        Lower-cased question with punctuation and extra spaces removed.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.-]", " ", question.lower())).strip()


class Service:
    """Admission, coalescing and caching in front of a pipeline.

    Attributes:
        pipeline: The pipeline.
        max_concurrent: Pipeline runs allowed at once.
        max_queue: Requests allowed to wait for a slot.
        cache_enabled: Whether to cache and coalesce.
        cache_ttl_s: Cache entry lifetime.
        queued: Requests currently waiting for a slot.
        peak_queued: Highest ``queued`` seen.
    """

    def __init__(self, pipeline: Pipeline, cfg: dict):
        """Initialises the service.

        Args:
            pipeline: The pipeline.
            cfg: The ``service`` config section.
        """
        self.pipeline = pipeline
        self.max_concurrent = cfg["max_concurrent"]
        self.max_queue = cfg["max_queue"]
        self.cache_enabled = cfg["cache"]
        self.cache_ttl_s = cfg["cache_ttl_s"]
        self._slots = asyncio.Semaphore(self.max_concurrent)
        self._cache: dict[str, tuple[float, Outcome]] = {}
        self._inflight: dict[str, asyncio.Future] = {}
        self.queued = 0
        self.peak_queued = 0

    def _from_shared(self, shared: Outcome, rid: str, t0: float, on_text) -> Outcome:
        """Builds this request's outcome from another request's result.

        Args:
            shared: The cached or coalesced outcome.
            rid: This request's id.
            t0: This request's arrival (``perf_counter`` clock).
            on_text: User callback.

        Returns:
            A copy with this request's timings and zero token cost.
        """
        now_ms = (time.perf_counter() - t0) * 1000
        if on_text and shared.text:
            on_text(shared.text)
        return replace(
            shared,
            request_id=rid,
            first_token_ms=now_ms if shared.text else None,
            total_ms=now_ms,
            retries=0,
            tokens_in=0,
            tokens_out=0,
            degraded=[*shared.degraded],
        )

    async def handle(
        self,
        question: str,
        request_id: str,
        deadline_at: float,
        on_text: TextCallback | None = None,
        t0: float | None = None,
        sink: dict | None = None,
    ) -> tuple[Outcome, dict]:
        """Answers one request through admission, cache and coalescing.

        Args:
            question: The question.
            request_id: Request id.
            deadline_at: Unix time the user gives up.
            on_text: User-visible text callback.
            t0: Arrival on the ``perf_counter`` clock.
            sink: Passed to ``Pipeline.run`` to expose the in-progress outcome.

        Returns:
            ``(outcome, info)`` where ``info`` records ``cache`` (``hit``,
            ``coalesced`` or ``miss``) and ``queue_wait_ms``.
        """
        t0 = t0 if t0 is not None else time.perf_counter()
        key = normalise(question)
        if self.cache_enabled:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < self.cache_ttl_s:
                return self._from_shared(hit[1], request_id, t0, on_text), {
                    "cache": "hit",
                    "queue_wait_ms": 0.0,
                }
            fut = self._inflight.get(key)
            if fut is not None:
                try:
                    shared = await asyncio.shield(fut)
                    return self._from_shared(shared, request_id, t0, on_text), {
                        "cache": "coalesced",
                        "queue_wait_ms": 0.0,
                    }
                except LeaderGone:
                    pass  # run it ourselves

        if self.queued >= self.max_queue and self._slots.locked():
            o = Outcome(
                request_id,
                question,
                status=ST_RATE_LIMITED,
                answer_kind="none",
                refusal_reason="queue_full",
            )
            msg = (
                "We're getting more questions than we can answer right now. "
                "Please try again in a minute."
            )
            o.text, o.first_token_ms = msg, (time.perf_counter() - t0) * 1000
            if on_text:
                on_text(msg)
            o.total_ms = o.first_token_ms
            return o, {"cache": "miss", "queue_wait_ms": 0.0, "rejected": "queue_full"}

        fut = None
        if self.cache_enabled:
            fut = asyncio.get_running_loop().create_future()
            self._inflight[key] = fut
        self.queued += 1
        self.peak_queued = max(self.peak_queued, self.queued)
        try:
            await self._slots.acquire()
        finally:
            self.queued -= 1
        queue_wait_ms = (time.perf_counter() - t0) * 1000
        try:
            o = await self.pipeline.run(
                question,
                request_id,
                deadline_at,
                on_text,
                t0=t0,
                queue_wait_ms=queue_wait_ms,
                sink=sink,
            )
        except BaseException:
            if fut is not None and not fut.done():
                fut.set_exception(LeaderGone())
                fut.exception()  # mark retrieved: followers may not exist
            raise
        finally:
            self._slots.release()
            if fut is not None:
                self._inflight.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(o)
        if self.cache_enabled and o.status == ST_OK:
            self._cache[key] = (time.time(), o)
        return o, {"cache": "miss", "queue_wait_ms": round(queue_wait_ms, 1)}
