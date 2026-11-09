"""A controllable stand-in for the Groq chat-completions API.

The pipeline talks to this service through exactly the same client and wire
format as the real one (SSE streaming, ``x-ratelimit-*`` headers, 429 body
text, ``retry-after``, ``x_groq.usage``), so switching is a base-URL change.

Behaviours, all configurable in ``config/standin.yaml`` and at runtime via
``POST /admin/config``:

* **Slow**: lognormal time-to-first-token plus prompt-length prefill cost.
* **Slower under load**: a fixed pool of workers with a queue in front, plus
  a multiplicative slowdown once utilisation passes a knee.
* **Fails**: immediate 5xx, hangs that never produce a token, slow successes,
  and streams that drop part-way.
* **Refuses at limits**: per-model RPM / TPM / RPD / TPD buckets returning 429.
* **Streams**: replies are sent a few tokens per SSE chunk.

Typical usage example:

    uv run python -m relay.standin                 # serve on 127.0.0.1:8900
    curl -X POST localhost:8900/admin/config -d '{"limits": {"default": {"tpm": 3000}}}'
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from relay.config import CONFIG_DIR, RAW_DIR, ROOT, _merge
from relay.llm_client import message_key

CALIBRATION_FILE = CONFIG_DIR / "standin_calibration.yaml"


def load_standin_config(extra: dict | None = None) -> dict:
    """Loads stand-in config: defaults, then fitted calibration, then overrides.

    Args:
        extra: Optional overrides applied last.

    Returns:
        The merged configuration.
    """
    with open(CONFIG_DIR / "standin.yaml") as f:
        cfg = yaml.safe_load(f)
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return _merge(cfg, extra or {})


def approx_tokens(text: str) -> int:
    """Estimates tokens as characters / 4 (the stand-in has no real tokenizer).

    Args:
        text: Input text.

    Returns:
        Estimated token count, at least 1.
    """
    return max(1, len(text) // 4)


@dataclass
class Buckets:
    """Rate-limit state for one model: four continuously refilling buckets.

    Measured on Groq (2026-11-09): RPM recovers one request every 60/RPM s,
    TPM refills at TPM/60 tokens per second, and the daily request budget
    refills continuously at RPD/86400 per second (``x-ratelimit-reset-requests``
    grew by exactly 86.4 s per request at RPD 1000). TPD is assumed to behave
    the same way; it is not exposed in headers.

    Attributes:
        rpm: Requests-per-minute limit.
        tpm: Tokens-per-minute limit.
        rpd: Requests-per-day limit.
        tpd: Tokens-per-day limit.
        tokens: Tokens available in the TPM bucket.
        req_used: Requests counted against RPM (drains at ``rpm/60`` per second).
        day_requests: Requests counted against RPD (drains at ``rpd/86400`` per second).
        day_tokens: Tokens counted against TPD (drains at ``tpd/86400`` per second).
        updated: Time of the last refill.
    """

    rpm: int
    tpm: int
    rpd: int
    tpd: int
    tokens: float = 0.0
    req_used: float = 0.0
    day_requests: float = 0.0
    day_tokens: float = 0.0
    updated: float = field(default_factory=time.time)

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


class StandIn:
    """Simulation state shared by all requests.

    Attributes:
        cfg: Active configuration (mutable at runtime).
        rng: Seeded random source.
        buckets: Per-model rate-limit state.
        busy: Requests currently holding a worker.
        queued: Requests waiting for a worker.
        replay: Recorded real replies keyed by message hash.
    """

    def __init__(self, cfg: dict):
        """Initialises the simulation.

        Args:
            cfg: Stand-in configuration.
        """
        self.cfg = cfg
        self.base_cfg = json.loads(json.dumps(cfg))
        self.rng = random.Random(cfg.get("seed", 0))
        self.buckets: dict[str, Buckets] = {}
        self.busy = 0
        self.queued = 0
        self.peak_queued = 0
        self.workers = asyncio.Semaphore(cfg["latency"]["workers"])
        self.replay = self._load_replay()
        self.log_path = RAW_DIR / "standin_calls.jsonl"

    def _load_replay(self) -> dict[str, str]:
        """Loads recorded replies, if any.

        Returns:
            Mapping of message hash to reply text.
        """
        path = ROOT / self.cfg["content"]["replay_file"]
        out: dict[str, str] = {}
        if path.exists():
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    out[rec["key"]] = rec["text"]
        return out

    def reconfigure(self, patch: dict) -> None:
        """Applies a runtime config patch (limits, latency, faults).

        Changing ``latency.workers`` rebuilds the worker pool; existing
        buckets pick up new limits immediately, keeping their usage.

        Args:
            patch: Partial configuration to merge.
        """
        self.cfg = _merge(self.cfg, patch)
        if "latency" in patch and "workers" in patch["latency"]:
            self.workers = asyncio.Semaphore(self.cfg["latency"]["workers"])
        for model, b in self.buckets.items():
            lim = self._limits_for(model)
            b.rpm, b.tpm, b.rpd, b.tpd = lim["rpm"], lim["tpm"], lim["rpd"], lim["tpd"]
            b.tokens = min(b.tokens, b.tpm)
        exhaust = patch.get("exhaust_day")
        if exhaust:
            for model in exhaust if isinstance(exhaust, list) else list(self.buckets):
                b = self._bucket(model)
                b.day_requests, b.day_tokens = b.rpd, b.tpd

    def _limits_for(self, model: str) -> dict:
        lim = self.cfg["limits"]
        return {**lim["default"], **lim.get("per_model", {}).get(model, {})}

    def _bucket(self, model: str) -> Buckets:
        if model not in self.buckets:
            lim = self._limits_for(model)
            self.buckets[model] = Buckets(
                lim["rpm"], lim["tpm"], lim["rpd"], lim["tpd"], tokens=lim["tpm"]
            )
        return self.buckets[model]

    def admit(self, model: str, cost: int) -> tuple[dict | None, dict]:
        """Checks and charges rate limits for one request.

        Args:
            model: Model id.
            cost: Tokens to reserve (prompt + completion budget).

        Returns:
            ``(rejection, headers)``. ``rejection`` is None when admitted, else
            a dict with the 429 message and retry delay.
        """
        now = time.time()
        b = self._bucket(model)
        b.refill(now)
        reject = None
        if b.day_requests + 1 > b.rpd:
            wait = (b.day_requests + 1 - b.rpd) * 86400.0 / b.rpd
            reject = ("requests per day (RPD)", b.rpd, int(b.day_requests), 1, wait)
        elif b.day_tokens + cost > b.tpd:
            wait = (b.day_tokens + cost - b.tpd) * 86400.0 / b.tpd
            reject = ("tokens per day (TPD)", b.tpd, int(b.day_tokens), cost, wait)
        elif b.req_used + 1 > b.rpm:
            wait = (b.req_used + 1 - b.rpm) * 60.0 / b.rpm
            reject = ("requests per minute (RPM)", b.rpm, round(b.req_used), 1, wait)
        elif cost > b.tokens:
            wait = (cost - b.tokens) * 60.0 / b.tpm
            reject = ("tokens per minute (TPM)", b.tpm, int(b.tpm - b.tokens), cost, wait)
        if reject is None:
            b.tokens -= cost
            b.req_used += 1
            b.day_requests += 1
            b.day_tokens += cost
        headers = {
            "x-ratelimit-limit-requests": str(b.rpd),
            "x-ratelimit-limit-tokens": str(b.tpm),
            "x-ratelimit-remaining-requests": str(max(0, int(b.rpd - b.day_requests))),
            "x-ratelimit-remaining-tokens": str(max(0, int(b.tokens))),
            "x-ratelimit-reset-requests": f"{b.day_requests * 86400.0 / b.rpd:.2f}s",
            "x-ratelimit-reset-tokens": f"{max(0.0, (b.tpm - b.tokens) * 60 / b.tpm):.2f}s",
        }
        if reject is None:
            return None, headers
        what, limit, used, requested, wait = reject
        wait = max(0.1, wait)
        msg = (
            f"Rate limit reached for model `{model}` in organization `org_standin` service "
            f"tier `on_demand` on {what}: Limit {limit}, Used {used}, Requested {requested}. "
            f"Please try again in {wait:.3f}s."
        )
        headers["retry-after"] = str(math.ceil(wait))
        return {"message": msg, "wait": wait}, headers

    def refund(self, model: str, tokens: int) -> None:
        """Returns unused reserved completion tokens to the buckets.

        Args:
            model: Model id.
            tokens: Tokens to give back.
        """
        b = self._bucket(model)
        b.tokens = min(b.tpm, b.tokens + tokens)
        b.day_tokens = max(0, b.day_tokens - tokens)

    def lognormal(self, median: float, sigma: float) -> float:
        """Samples a lognormal value parameterised by its median.

        Args:
            median: Median of the distribution.
            sigma: Standard deviation of the underlying normal.

        Returns:
            A positive sample.
        """
        return median * math.exp(self.rng.gauss(0.0, sigma))

    def slowdown(self) -> float:
        """Current load multiplier applied to latency.

        Returns:
            ``1 + alpha * max(0, utilisation - knee)``, where utilisation counts
            busy and queued requests against the worker count.
        """
        lat = self.cfg["latency"]
        util = (self.busy + self.queued) / lat["workers"]
        return 1.0 + lat["load_alpha"] * max(0.0, util - lat["load_knee"])

    def reply_text(self, messages: list[dict], step: str, max_tokens: int) -> str:
        """Produces reply content: a recorded real reply, or a synthetic one.

        Synthetic replies are shaped for the pipeline step named in the
        ``X-Relay-Step`` header so downstream parsing works, but their content
        quality is meaningless; quality is only measured on real runs.

        Args:
            messages: Request messages.
            step: Pipeline step (``understand``, ``draft``, ``verify``, or other).
            max_tokens: Completion cap.

        Returns:
            Reply text.
        """
        recorded = self.replay.get(message_key(messages))
        if recorded is not None:
            return recorded
        prompt = messages[-1].get("content", "") if messages else ""
        if step == "understand":
            q = re.sub(r"\s+", " ", prompt)[-300:].replace('"', "'")
            return json.dumps({"in_scope": True, "papers": [], "query": q, "multi_part": False})
        if step == "verify":
            return json.dumps({"supported": True, "unsupported_claims": []})
        if step == "draft":
            labels = re.findall(r"\[(\d+)\]\s*\(([^)]*)\)\s*(.+)", prompt)
            if not labels:
                return "The provided passages do not say."
            n_tok = int(self.lognormal(self.cfg["content"]["synthetic_out_tokens_median"], 0.3))
            out, used = [], 0
            for num, _, text in labels:
                for sent in re.split(r"(?<=[.!?])\s+", text):
                    if used >= min(n_tok, max_tokens):
                        break
                    out.append(f"{sent.strip()} [{num}]")
                    used += approx_tokens(sent)
            return " ".join(out)
        return "pong"


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def create_app(cfg: dict | None = None) -> FastAPI:
    """Builds the FastAPI application.

    Args:
        cfg: Stand-in configuration; loaded from disk when omitted.

    Returns:
        The ASGI app.
    """
    app = FastAPI(title="relay stand-in")
    sim = StandIn(cfg or load_standin_config())
    app.state.sim = sim
    sim.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(rec: dict) -> None:
        with open(sim.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    @app.post("/openai/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        model = body.get("model", "unknown")
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 256)
        step = request.headers.get("x-relay-step", "")
        prompt_tokens = sum(approx_tokens(m.get("content") or "") for m in messages)
        reserve = prompt_tokens + (max_tokens if sim.cfg["limits"]["reserve_max_tokens"] else 0)
        arrived = time.time()
        rec = {
            "id": uuid.uuid4().hex[:12],
            "t": arrived,
            "model": model,
            "step": step,
            "prompt_tokens": prompt_tokens,
        }

        rejection, headers = sim.admit(model, reserve)
        if rejection is not None:
            log({**rec, "status": 429, "msg": rejection["message"]})
            return JSONResponse(
                status_code=429,
                headers=headers,
                content={
                    "error": {
                        "message": rejection["message"],
                        "type": "tokens",
                        "code": "rate_limit_exceeded",
                    }
                },
            )

        faults = sim.cfg["faults"]
        roll = sim.rng.random()
        if roll < faults["p_server_error"]:
            sim.refund(model, reserve)
            log({**rec, "status": 503})
            return JSONResponse(
                status_code=503,
                headers=headers,
                content={
                    "error": {"message": "Service Unavailable", "type": "internal_server_error"}
                },
            )
        roll -= faults["p_server_error"]
        hang = roll < faults["p_hang"]
        roll -= faults["p_hang"]
        slow = 0 <= roll < faults["p_slow_success"]
        roll -= faults["p_slow_success"]
        drop = 0 <= roll < faults["p_midstream_drop"]

        text = sim.reply_text(messages, step, max_tokens)
        completion_tokens = min(max_tokens, approx_tokens(text))
        lat = {**sim.cfg["latency"], **sim.cfg["latency"].get("per_model", {}).get(model, {})}

        async def stream():
            sim.queued += 1
            sim.peak_queued = max(sim.peak_queued, sim.queued)
            t_q = time.time()
            try:
                await sim.workers.acquire()
            finally:
                sim.queued -= 1
            sim.busy += 1
            status = "ok"
            queue_ms = (time.time() - t_q) * 1000
            try:
                factor = sim.slowdown()
                ttft = sim.lognormal(lat["ttft_median_ms"], lat["ttft_sigma"])
                ttft += lat["prefill_ms_per_1k"] * prompt_tokens / 1000
                ttft *= factor * (faults["slow_factor"] if slow else 1.0)
                if hang:
                    status = "hang"
                    await asyncio.sleep(3600)
                await asyncio.sleep(ttft / 1000)
                tps = sim.lognormal(lat["decode_tps_median"], lat["decode_tps_sigma"]) / factor
                cid = f"chatcmpl-{rec['id']}"
                words = re.findall(r"\S+\s*", text)
                per_chunk = max(1, lat["tokens_per_chunk"])
                cut = len(words) // 2 if drop else None
                for i in range(0, len(words), per_chunk):
                    if cut is not None and i >= cut:
                        status = "dropped"
                        return
                    piece = "".join(words[i : i + per_chunk])
                    yield _sse(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                            ],
                        }
                    )
                    await asyncio.sleep(approx_tokens(piece) / tps)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                yield _sse(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "x_groq": {"usage": usage},
                    }
                )
                yield b"data: [DONE]\n\n"
            except asyncio.CancelledError:
                status = "client_gone"
                raise
            finally:
                sim.busy -= 1
                sim.workers.release()
                sim.refund(model, max(0, reserve - prompt_tokens - completion_tokens))
                log(
                    {
                        **rec,
                        "status": status,
                        "queue_ms": round(queue_ms, 1),
                        "slow": slow,
                        "total_ms": round((time.time() - arrived) * 1000, 1),
                    }
                )

        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    @app.post("/admin/config")
    async def admin_config(request: Request) -> dict:
        sim.reconfigure(await request.json())
        return {"ok": True, "limits": sim.cfg["limits"], "workers": sim.cfg["latency"]["workers"]}

    @app.get("/admin/stats")
    async def admin_stats() -> dict:
        return {
            "busy": sim.busy,
            "queued": sim.queued,
            "peak_queued": sim.peak_queued,
            "buckets": {
                m: {
                    "tokens": round(b.tokens),
                    "rpm_used": round(b.req_used, 1),
                    "day_requests": round(b.day_requests, 1),
                    "day_tokens": round(b.day_tokens),
                }
                for m, b in sim.buckets.items()
            },
        }

    @app.post("/admin/reset")
    async def admin_reset() -> dict:
        """Restores the startup config and clears all limit state and counters."""
        sim.cfg = json.loads(json.dumps(sim.base_cfg))
        sim.workers = asyncio.Semaphore(sim.cfg["latency"]["workers"])
        sim.rng = random.Random(sim.cfg.get("seed", 0))
        sim.buckets.clear()
        sim.peak_queued = 0
        return {"ok": True}

    return app


def main() -> None:
    """Runs the stand-in with uvicorn using the configured host and port."""
    import uvicorn

    cfg = load_standin_config()
    uvicorn.run(create_app(cfg), host=cfg["host"], port=cfg["port"], log_level="warning")


if __name__ == "__main__":
    main()
