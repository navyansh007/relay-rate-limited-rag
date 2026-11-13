"""Streaming client for any OpenAI-compatible chat endpoint (Groq or the stand-in).

One call to `LLMClient.chat` is exactly one HTTP attempt. Retries live in
`relay.retry`, not here, so every attempt is measured and logged on its own.

Timing points recorded per attempt, in milliseconds since the attempt started:

* ``first_byte_ms``: first SSE chunk of any kind, including hidden reasoning.
* ``ttft_ms``: first *visible* content token.
* ``total_ms``: stream finished or failed.

Typical usage example:

    client = LLMClient(base_url, api_key=key, logger=CallLogger(path))
    result = await client.chat(messages, "llama-3.1-8b-instant", max_tokens=64)
    if result.status == OK:
        print(result.text)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

RATE_HEADERS = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)

# e.g. "... on tokens per minute (TPM): Limit 6000, Used 5800, Requested 900 ..."
_LIMIT_RE = re.compile(r"on (requests|tokens) per (minute|day)", re.IGNORECASE)

# Status values for a single attempt.
OK = "ok"
RL_MINUTE = "rate_limited_minute"
RL_DAY = "rate_limited_day"
SERVER_ERROR = "server_error"
TIMEOUT = "timeout"
ERROR = "error"
RETRYABLE = {RL_MINUTE, SERVER_ERROR, TIMEOUT}

TokenCallback = Callable[[str], Awaitable[None] | None]


def message_key(messages: list[dict]) -> str:
    """Hashes a message list; links real replies to stand-in replay.

    Args:
        messages: OpenAI-format messages.

    Returns:
        Hex SHA-256 of the canonical JSON encoding.
    """
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class CallResult:
    """Outcome and measurements of a single model-call attempt.

    Attributes:
        model: Model id the call was made against.
        source: ``"real"`` for Groq or ``"stand-in"`` for the local service.
        status: One of the module-level status constants (``OK``, ``RL_MINUTE``...).
        http_status: HTTP status code, or None if no response was received.
        started_at: Unix timestamp (seconds) when the attempt began.
        first_byte_ms: Time to the first SSE chunk of any kind.
        ttft_ms: Time to the first visible content token.
        total_ms: Time until the stream ended or the attempt failed.
        text: Visible content received (partial if the stream broke).
        reasoning_chars: Characters of hidden reasoning streamed before content.
        tokens_in: Prompt tokens reported by the server's usage block.
        tokens_out: Completion tokens reported by the server's usage block.
        finish_reason: Server-reported finish reason, e.g. ``"stop"`` or ``"length"``.
        limit_kind: For 429s, which bucket tripped, e.g. ``"tokens/minute"``.
        retry_after_s: Parsed ``retry-after`` header, if present.
        headers: Rate-limit related response headers, verbatim.
        error: Error body or exception description, truncated.
        tag: Caller-supplied label (probe name, pipeline step, request id).
        msg_key: Hash of the request messages (see ``message_key``).
        prompt_chars: Total characters in the request messages.
    """

    model: str
    source: str
    status: str = ERROR
    http_status: int | None = None
    started_at: float = 0.0
    first_byte_ms: float | None = None
    ttft_ms: float | None = None
    total_ms: float = 0.0
    text: str = ""
    reasoning_chars: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reason: str | None = None
    limit_kind: str | None = None
    retry_after_s: float | None = None
    headers: dict = field(default_factory=dict)
    error: str | None = None
    tag: str = ""
    msg_key: str = ""
    prompt_chars: int = 0

    def to_json(self) -> str:
        """Serialises the result as a single JSON line.

        Returns:
            The JSON encoding of all fields.
        """
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_duration_s(value: str | None) -> float | None:
    """Parses a Groq duration string into seconds.

    Groq reset headers use Go-style durations such as ``"7.66s"``,
    ``"2m59.56s"`` or ``"120ms"``; ``retry-after`` is a plain number.

    Args:
        value: Raw header value, or None.

    Returns:
        The duration in seconds, or None if the value is empty or unparseable.
    """
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    total, matched = 0.0, False
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", value):
        matched = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if matched else None


def classify_429(body: str) -> tuple[str, str | None]:
    """Distinguishes a per-minute from a per-day rate-limit rejection.

    The distinction matters because a per-minute 429 is worth waiting out while
    a per-day 429 is not.

    Args:
        body: Response body of the 429.

    Returns:
        A ``(status, limit_kind)`` tuple. ``status`` is ``RL_DAY`` or
        ``RL_MINUTE``; ``limit_kind`` is e.g. ``"tokens/minute"``, or None when
        the body does not say (treated as per-minute).
    """
    m = _LIMIT_RE.search(body or "")
    if not m:
        return RL_MINUTE, None
    what, per = m.group(1).lower(), m.group(2).lower()
    return (RL_DAY if per == "day" else RL_MINUTE), f"{what}/{per}"


# Groq error bodies name the account's organisation id; raw logs are submitted.
_ORG_ID = re.compile(r"org_[0-9a-z]{20,}")


class CallLogger:
    """Appends every attempt to a JSONL file: the raw evidence behind every reported number.

    Attributes:
        path: Destination file, or None to disable logging.
    """

    def __init__(self, path: Path | None):
        """Initialises the logger and creates the parent directory.

        Args:
            path: Destination JSONL file, or None to disable logging.
        """
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: CallResult) -> None:
        """Appends one result as a JSON line.

        Args:
            result: The attempt to record.
        """
        if self.path:
            with open(self.path, "a") as f:
                f.write(_ORG_ID.sub("org_[MASKED]", result.to_json()) + "\n")


class LLMClient:
    """Async streaming chat client for OpenAI-compatible endpoints.

    Attributes:
        base_url: Endpoint root, e.g. ``https://api.groq.com/openai/v1``.
        source: Label stored on every result (``"real"`` or ``"stand-in"``).
        first_token_timeout_s: Abort if no visible token arrives in this time.
        stall_timeout_s: Abort if the stream pauses longer than this between chunks.
        logger: Where each attempt is recorded.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        source: str = "real",
        connect_timeout_s: float = 3.0,
        first_token_timeout_s: float = 8.0,
        stall_timeout_s: float = 5.0,
        logger: CallLogger | None = None,
        max_connections: int = 64,
    ):
        """Initialises the client and its connection pool.

        Args:
            base_url: Endpoint root without the ``/chat/completions`` suffix.
            api_key: Bearer token; empty for the stand-in.
            source: Label stored on every result.
            connect_timeout_s: TCP/TLS connect timeout.
            first_token_timeout_s: Guard against calls that "succeed slowly".
            stall_timeout_s: Guard against streams that stop mid-reply.
            logger: Attempt logger; defaults to a no-op logger.
            max_connections: Connection-pool size.
        """
        self.base_url = base_url.rstrip("/")
        self.source = source
        self.first_token_timeout_s = first_token_timeout_s
        self.stall_timeout_s = stall_timeout_s
        self.logger = logger or CallLogger(None)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(connect=connect_timeout_s, read=None, write=10, pool=None),
            limits=httpx.Limits(max_connections=max_connections),
        )

    async def aclose(self) -> None:
        """Closes the underlying connection pool."""
        await self._http.aclose()

    async def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        on_token: TokenCallback | None = None,
        extra: dict | None = None,
        tag: str = "",
        deadline_s: float | None = None,
        step: str = "",
    ) -> CallResult:
        """Streams one chat completion.

        HTTP errors, rate limits and timeouts are reported through the returned
        result's ``status`` rather than raised, so callers can branch on them.

        Args:
            messages: OpenAI-format chat messages.
            model: Model id.
            max_tokens: Completion token cap.
            temperature: Sampling temperature.
            on_token: Optional callback (sync or async) invoked per content piece.
            extra: Additional request-body fields (e.g. ``reasoning_effort``).
            tag: Label stored on the result and in the log.
            deadline_s: Remaining time budget of the caller; tightens the
                first-token timeout if shorter.
            step: Pipeline step name, sent as ``X-Relay-Step`` so the stand-in
                can shape synthetic replies; ignored by Groq.

        Returns:
            The attempt's measurements and content.

        Raises:
            asyncio.CancelledError: If the caller cancels (e.g. user gave up);
                the attempt is still logged as a timeout before re-raising.
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **(extra or {}),
        }
        res = CallResult(
            model=model,
            source=self.source,
            started_at=time.time(),
            tag=tag,
            msg_key=message_key(messages),
            prompt_chars=sum(len(m.get("content") or "") for m in messages),
        )
        t0 = time.perf_counter()

        def ms() -> float:
            return (time.perf_counter() - t0) * 1000

        first_timeout = self.first_token_timeout_s
        if deadline_s is not None:
            first_timeout = max(0.05, min(first_timeout, deadline_s))

        parts: list[str] = []
        try:
            async with self._http.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"X-Relay-Step": step} if step else None,
            ) as resp:
                res.http_status = resp.status_code
                res.headers = {h: resp.headers[h] for h in RATE_HEADERS if h in resp.headers}
                res.retry_after_s = parse_duration_s(resp.headers.get("retry-after"))
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    res.error = body[:2000]
                    if resp.status_code == 429:
                        res.status, res.limit_kind = classify_429(body)
                    elif resp.status_code >= 500:
                        res.status = SERVER_ERROR
                    else:
                        res.status = ERROR
                    return res

                lines = resp.aiter_lines()
                while True:
                    timeout = first_timeout if res.ttft_ms is None else self.stall_timeout_s
                    try:
                        line = await asyncio.wait_for(anext(lines), timeout=timeout)
                    except StopAsyncIteration:
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if res.first_byte_ms is None:
                        res.first_byte_ms = ms()
                    if "error" in chunk:
                        res.status, res.error = SERVER_ERROR, json.dumps(chunk["error"])[:2000]
                        return res
                    usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                    if usage:
                        res.tokens_in = usage.get("prompt_tokens", 0)
                        res.tokens_out = usage.get("completion_tokens", 0)
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning"):
                            res.reasoning_chars += len(delta["reasoning"])
                        piece = delta.get("content")
                        if piece:
                            if res.ttft_ms is None:
                                res.ttft_ms = ms()
                            parts.append(piece)
                            if on_token:
                                r = on_token(piece)
                                if asyncio.iscoroutine(r):
                                    await r
                        if choice.get("finish_reason"):
                            res.finish_reason = choice["finish_reason"]
                res.status = OK
        except (TimeoutError, httpx.TimeoutException) as e:
            res.status, res.error = TIMEOUT, f"{type(e).__name__}: {e}"
        except httpx.HTTPError as e:
            res.status, res.error = ERROR, f"{type(e).__name__}: {e}"
        except asyncio.CancelledError:
            res.status, res.error = TIMEOUT, "cancelled (caller gave up)"
            raise
        finally:
            res.text = "".join(parts)
            res.total_ms = ms()
            self.logger.log(res)
        return res
