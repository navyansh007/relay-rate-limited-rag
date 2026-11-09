"""Per-request latency spans, classified for the "where the time goes" analysis.

Every span is one of three kinds:

* ``wait``: time the request could not make progress (admission queue,
  limiter wait for rate-limit capacity, re-rank CPU queue).
* ``work``: time spent doing useful work (tool call, successful model call).
* ``retry``: time lost to failure (failed attempts and backoff sleeps).

Spans are stored with millisecond offsets from the request's arrival, and
written to ``raw/traces.jsonl`` keyed by ``request_id``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Span:
    """A timed interval within a request.

    Attributes:
        kind: ``"wait"``, ``"work"`` or ``"retry"``.
        name: What happened, e.g. ``"draft.attempt"`` or ``"limiter.draft"``.
        start_ms: Offset from request arrival.
        end_ms: Offset from request arrival.
        attrs: Extra details (model, status, ttft, tokens...).
    """

    kind: str
    name: str
    start_ms: float
    end_ms: float
    attrs: dict = field(default_factory=dict)


class Trace:
    """Collects spans and request-level attributes for one request.

    Attributes:
        request_id: Request identifier.
        t0: Arrival time (``time.perf_counter`` clock).
        spans: Recorded spans.
        attrs: Request-level attributes.
    """

    def __init__(self, request_id: str, t0: float | None = None):
        """Starts a trace.

        Args:
            request_id: Request identifier.
            t0: Arrival time on the ``perf_counter`` clock; defaults to now.
        """
        self.request_id = request_id
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.spans: list[Span] = []
        self.attrs: dict = {}

    def now_ms(self) -> float:
        """Milliseconds since arrival.

        Returns:
            Elapsed time.
        """
        return (time.perf_counter() - self.t0) * 1000

    def add(self, kind: str, name: str, start_ms: float, end_ms: float, **attrs) -> None:
        """Records a span.

        Args:
            kind: ``"wait"``, ``"work"`` or ``"retry"``.
            name: Span name.
            start_ms: Start offset.
            end_ms: End offset.
            **attrs: Extra details.
        """
        self.spans.append(Span(kind, name, round(start_ms, 1), round(end_ms, 1), attrs))

    def totals(self) -> dict[str, float]:
        """Sums span durations by kind.

        Returns:
            ``{"wait": ms, "work": ms, "retry": ms}``.
        """
        out = {"wait": 0.0, "work": 0.0, "retry": 0.0}
        for s in self.spans:
            out[s.kind] += s.end_ms - s.start_ms
        return {k: round(v, 1) for k, v in out.items()}

    def to_dict(self) -> dict:
        """Serialises the trace.

        Returns:
            JSON-ready dict.
        """
        return {
            "request_id": self.request_id,
            "attrs": self.attrs,
            "totals": self.totals(),
            "spans": [asdict(s) for s in self.spans],
        }


class TraceWriter:
    """Appends traces to a JSONL file.

    Attributes:
        path: Destination file, or None to disable.
    """

    def __init__(self, path: Path | None):
        """Initialises the writer.

        Args:
            path: Destination file, or None to disable.
        """
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: Trace) -> None:
        """Appends one trace.

        Args:
            trace: The trace to store.
        """
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
