import asyncio
import time

import httpx
from conftest import admin

from relay.llm_client import OK, RL_DAY, RL_MINUTE, SERVER_ERROR, TIMEOUT, LLMClient


async def call(base_url, **kw):
    cli = LLMClient(
        f"{base_url}/openai/v1", source="stand-in", first_token_timeout_s=kw.pop("ftt", 5.0)
    )
    try:
        return await cli.chat(
            [{"role": "user", "content": "hello " * 50}],
            "m",
            max_tokens=kw.pop("max_tokens", 50),
            **kw,
        )
    finally:
        await cli.aclose()


async def test_streams_ok_with_usage(base_url):
    admin(base_url, {})
    prompt = "[1] (transformer §3.2 p.3) Attention maps queries to outputs. It is fast."
    cli = LLMClient(f"{base_url}/openai/v1", source="stand-in")
    pieces = []
    r = await cli.chat(
        [{"role": "user", "content": prompt}], "m", step="draft", on_token=pieces.append
    )
    await cli.aclose()
    assert r.status == OK and "[1]" in r.text and len(pieces) >= 2
    assert r.tokens_in > 0 and r.tokens_out > 0
    assert 0 < r.ttft_ms <= r.total_ms
    assert "x-ratelimit-remaining-tokens" in r.headers


async def test_tpm_limit_is_per_minute_429(base_url):
    admin(base_url, {"limits": {"default": {"tpm": 200}}})
    first = await call(base_url)
    second = await call(base_url)
    assert first.status == OK
    assert second.status == RL_MINUTE and second.limit_kind == "tokens/minute"
    assert second.retry_after_s and second.retry_after_s >= 1


async def test_exhausted_day_is_per_day_429(base_url):
    admin(base_url, {})
    await call(base_url)
    httpx.post(f"{base_url}/admin/config", json={"exhaust_day": True})
    r = await call(base_url)
    assert r.status == RL_DAY and r.limit_kind in ("requests/day", "tokens/day")


async def test_server_error(base_url):
    admin(base_url, {"faults": {"p_server_error": 1.0}})
    r = await call(base_url)
    assert r.status == SERVER_ERROR and r.http_status == 503


async def test_hang_hits_first_token_timeout(base_url):
    admin(base_url, {"faults": {"p_hang": 1.0}})
    t0 = time.perf_counter()
    r = await call(base_url, ftt=0.3)
    assert r.status == TIMEOUT and time.perf_counter() - t0 < 2
    await asyncio.sleep(0.1)
    stats = httpx.get(f"{base_url}/admin/stats").json()
    assert stats["busy"] == 0  # hung worker released when the client left
