"""Streaming client for any OpenAI-compatible chat endpoint (Groq or the stand-in).

One call = one HTTP attempt. Retries live in relay.retry, not here, so every
attempt is measured and logged on its own.

Timing points recorded per attempt (ms since the attempt started):
  first_byte_ms   first SSE chunk of any kind (incl. hidden reasoning)
  ttft_ms         first *visible* content token
  total_ms        stream finished / failed
"""

from __future__ import annotations

import asyncio
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


@dataclass
class CallResult:
    model: str
    source: str  # real | stand-in
    status: str = ERROR
    http_status: int | None = None
    started_at: float = 0.0  # unix seconds
    first_byte_ms: float | None = None
    ttft_ms: float | None = None
    total_ms: float = 0.0
    text: str = ""
    reasoning_chars: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reason: str | None = None
    limit_kind: str | None = None  # e.g. "tokens/minute"
    retry_after_s: float | None = None
    headers: dict = field(default_factory=dict)
    error: str | None = None
    tag: str = ""  # caller-provided label (probe name, pipeline step...)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_duration_s(value: str | None) -> float | None:
    """Parse Groq reset strings like '7.66s', '2m59.56s', '1h2m3s', '120ms'."""
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
    m = _LIMIT_RE.search(body or "")
    if not m:
        return RL_MINUTE, None
    what, per = m.group(1).lower(), m.group(2).lower()
    return (RL_DAY if per == "day" else RL_MINUTE), f"{what}/{per}"


class CallLogger:
    """Append-only JSONL of every attempt (the raw evidence reviewers ask for)."""

    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: CallResult) -> None:
        if self.path:
            with open(self.path, "a") as f:
                f.write(result.to_json() + "\n")


class LLMClient:
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
    ) -> CallResult:
        """Stream one completion. Never raises for HTTP/timeout problems."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **(extra or {}),
        }
        res = CallResult(model=model, source=self.source, started_at=time.time(), tag=tag)
        t0 = time.perf_counter()
        ms = lambda: (time.perf_counter() - t0) * 1000

        first_timeout = self.first_token_timeout_s
        if deadline_s is not None:
            first_timeout = max(0.05, min(first_timeout, deadline_s))

        parts: list[str] = []
        try:
            async with self._http.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload
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
