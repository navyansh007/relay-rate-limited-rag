import time

from conftest import admin

from relay.limiter import Limiter
from relay.llm_client import OK, RL_DAY, RL_MINUTE, CallResult, LLMClient
from relay.retry import RetryingCaller, RetryPolicy

LIM = {"m": {"rpm": 3, "tpm": 1000, "rpd": 100, "tpd": 100000}}


async def test_acquire_charges_and_sheds_on_deadline():
    lim = Limiter(LIM)
    g = await lim.acquire("m", 900, time.time() + 5)
    assert g.ok
    # bucket now ~100 tokens; 900 more needs ~48 s of refill: shed immediately
    g2 = await lim.acquire("m", 900, time.time() + 1)
    assert not g2.ok and g2.reason == "deadline" and g2.waited_ms < 200


async def test_rpm_bucket_delays_fourth_call():
    lim = Limiter(LIM)
    for _ in range(3):
        assert (await lim.acquire("m", 1, time.time() + 1)).ok
    # continuous refill: the 4th request waits ~60/rpm = 20 s, not a full minute
    assert 15 < lim.estimated_wait_s("m", 1) <= 20


def test_observe_trusts_lower_remaining_and_new_limits():
    lim = Limiter(LIM)
    r = CallResult(
        "m",
        "real",
        status=OK,
        http_status=200,
        tokens_in=10,
        tokens_out=5,
        headers={
            "x-ratelimit-limit-tokens": "500",
            "x-ratelimit-remaining-tokens": "120",
            "x-ratelimit-remaining-requests": "40",
        },
    )
    lim.observe("m", r, reserved=15)
    b = lim.budget("m")
    assert b.tpm == 500 and b.tokens <= 120 and b.day_requests == 60


def test_day_429_marks_model_exhausted():
    lim = Limiter(LIM)
    r = CallResult(
        "m",
        "real",
        status=RL_DAY,
        http_status=429,
        headers={"x-ratelimit-reset-requests": "2h0m0s"},
    )
    lim.observe("m", r, reserved=10)
    assert lim.day_exhausted("m") and 7100 < lim.day_reset_in_s("m") <= 7200


def test_day_429_prefers_retry_after_over_full_reset():
    # Daily buckets refill continuously: the next call fits after retry-after,
    # long before the bucket is full again (reset-requests).
    lim = Limiter(LIM)
    r = CallResult(
        "m",
        "real",
        status=RL_DAY,
        http_status=429,
        retry_after_s=87.0,
        headers={"x-ratelimit-reset-requests": "23h59m0s", "retry-after": "87"},
    )
    lim.observe("m", r, reserved=10)
    assert lim.day_exhausted("m") and 80 < lim.day_reset_in_s("m") <= 87


async def test_budgeted_retry_recovers_from_server_errors(base_url):
    admin(base_url, {"faults": {"p_server_error": 0.5}})
    cli = LLMClient(f"{base_url}/openai/v1", source="stand-in")
    caller = RetryingCaller(
        cli, Limiter({}, enabled=False), RetryPolicy("budgeted", max_retries=3, budget_ratio=1.0)
    )
    oks = 0
    for _ in range(30):
        out = await caller.call(
            [{"role": "user", "content": "hi"}], "m", 20, time.time() + 10, est_prompt_tokens=5
        )
        oks += out.ok
    await cli.aclose()
    # P(success) per call = 1 - 0.5**4 = 0.94 -> expect ~28 of 30
    assert oks >= 24


async def test_no_retry_on_day_limit(base_url):
    admin(base_url, {})
    import httpx

    httpx.post(f"{base_url}/admin/config", json={"exhaust_day": ["m"]})
    cli = LLMClient(f"{base_url}/openai/v1", source="stand-in")
    caller = RetryingCaller(cli, Limiter({}, enabled=False), RetryPolicy("budgeted"))
    out = await caller.call([{"role": "user", "content": "hi"}], "m", 20, time.time() + 10, 5)
    await cli.aclose()
    assert out.final.status == RL_DAY and out.retries == 0 and out.refused == "not_retryable"


async def test_retry_honours_retry_after(base_url):
    admin(base_url, {"limits": {"default": {"rpm": 1}}})
    cli = LLMClient(f"{base_url}/openai/v1", source="stand-in")
    caller = RetryingCaller(cli, Limiter({}, enabled=False), RetryPolicy("budgeted"))
    await caller.call([{"role": "user", "content": "hi"}], "m", 5, time.time() + 5, 5)
    out = await caller.call([{"role": "user", "content": "hi"}], "m", 5, time.time() + 5, 5)
    await cli.aclose()
    # retry-after is ~60 s, deadline 5 s: the policy must give up instead of sleeping
    assert out.attempts[0].result.status == RL_MINUTE and out.refused == "deadline"


async def test_retry_budget_caps_retries_under_persistent_failure(base_url):
    admin(base_url, {"faults": {"p_server_error": 1.0}})
    cli = LLMClient(f"{base_url}/openai/v1", source="stand-in")
    caller = RetryingCaller(
        cli,
        Limiter({}, enabled=False),
        RetryPolicy("budgeted", max_retries=3, base_backoff_s=0.01, budget_ratio=0.2),
    )
    outs = [
        await caller.call([{"role": "user", "content": "hi"}], "m", 5, time.time() + 10, 5)
        for _ in range(20)
    ]
    await cli.aclose()
    retries = sum(o.retries for o in outs)
    # without a budget this would be 20 * 3 = 60 retries against a failing service
    assert retries <= 1 + 0.2 * 20 + 1
    assert any(o.refused == "budget" for o in outs)


def test_day_wait_estimated_from_local_bucket_without_a_429():
    # Exhaustion seen only through remaining-requests headers (no 429): the
    # wait comes from the continuous refill rate, 86.4 s per request at 1000 RPD.
    lim = Limiter({"m": {"rpm": 30, "tpm": 8000, "rpd": 1000, "tpd": 10**9}})
    b = lim.budget("m")
    b.day_requests = 1000
    assert lim.day_exhausted("m")
    assert 80 < lim.day_reset_in_s("m") <= 87
