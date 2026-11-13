import json
import time
from pathlib import Path

import httpx
import pytest
from conftest import admin

from relay.config import load_config
from relay.limiter import Limiter
from relay.llm_client import LLMClient
from relay.pipeline import Pipeline
from relay.retry import RetryingCaller, RetryPolicy
from relay.trace import TraceWriter

pytestmark = pytest.mark.skipif(
    not Path("data/index/chunks_300_60.jsonl").exists(), reason="run relay.ingest first"
)


@pytest.fixture(scope="module")
def retrievers():
    return Pipeline.load_retrievers(load_config())


def make(base_url, retrievers, tmp_path, **pipeline_over):
    cfg = load_config()
    cfg["pipeline"].update(pipeline_over)
    cli = LLMClient(f"{base_url}/openai/v1", source="stand-in")
    lim = Limiter(cfg["limits"])
    caller = RetryingCaller(cli, lim, RetryPolicy.from_config(cfg["retry"]))
    return Pipeline(cfg, caller, lim, retrievers, TraceWriter(tmp_path / "t.jsonl")), cli, lim


async def test_answerable_question_streams_cited_answer(base_url, retrievers, tmp_path):
    admin(base_url, {})
    pipe, cli, _ = make(base_url, retrievers, tmp_path)
    seen = []
    o = await pipe.run(
        "How many attention heads does the Transformer use?",
        "r1",
        time.time() + 20,
        on_text=seen.append,
    )
    await cli.aclose()
    assert o.status == "ok" and o.answer_kind == "answer", o
    assert o.citations and all("transformer" in c for c in o.citations)
    assert 0 < o.first_token_ms <= o.total_ms
    assert o.tokens_in > 0 and "".join(seen) == o.text
    trace = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[-1])
    assert {s["name"].split(".")[0] for s in trace["spans"]} >= {"understand", "retrieve", "draft"}


async def test_out_of_scope_refused_without_draft(base_url, retrievers, tmp_path):
    admin(base_url, {})
    pipe, cli, _ = make(base_url, retrievers, tmp_path, understand_mode="rules")
    o = await pipe.run("What's the weather in Paris tomorrow?", "r2", time.time() + 10)
    await cli.aclose()
    assert o.answer_kind == "refusal" and o.refusal_reason == "out_of_scope"
    assert o.tokens_in == 0


async def test_daily_budget_gone_gives_extractive_answer(base_url, retrievers, tmp_path):
    admin(base_url, {})
    pipe, cli, lim = make(base_url, retrievers, tmp_path, understand_mode="rules")
    cfg = load_config()
    for m in (cfg["models"]["draft"], cfg["models"]["draft_fallback"]):
        lim.budget(m).day_blocked_until = time.time() + 3 * 3600
    o = await pipe.run("What optimizer did the Transformer use?", "r3", time.time() + 10)
    await cli.aclose()
    assert o.status == "rate_limited" and o.answer_kind == "extractive"
    assert "try again in about 2 h 59 min" in o.text or "try again in about 3 h" in o.text
    assert o.tokens_in == 0 and o.citations


async def test_day_429_from_server_degrades_to_extractive(base_url, retrievers, tmp_path):
    admin(base_url, {})
    httpx.post(f"{base_url}/admin/config", json={"exhaust_day": True})
    pipe, cli, _ = make(base_url, retrievers, tmp_path, understand_mode="rules")
    # first call discovers the exhausted day via a 429, not via local state
    httpx.post(f"{base_url}/admin/config", json={"exhaust_day": [load_config()["models"]["draft"]]})
    o = await pipe.run("What optimizer did the Transformer use?", "r4", time.time() + 10)
    await cli.aclose()
    assert o.status == "rate_limited" and o.answer_kind == "extractive"
    assert o.retries == 0


async def test_named_paper_blocks_cross_paper_answer(base_url, retrievers, tmp_path):
    """H-Net never states its GPU count; the Transformer paper's "8 P100 GPUs" must not leak."""
    admin(base_url, {})
    pipe, cli, _ = make(base_url, retrievers, tmp_path)
    o = await pipe.run("How many GPUs was H-Net trained on?", "r5", time.time() + 30)
    await cli.aclose()
    assert "P100" not in o.text
    assert all(c.startswith("hnet") for c in o.citations)


async def test_give_up_while_waiting_for_limiter_is_traced_as_wait(base_url, retrievers, tmp_path):
    # Found on the real service: a user who gave up while the draft waited for
    # per-minute capacity left a trace with wait=0. The wait must be recorded.
    import asyncio

    admin(base_url, {})
    pipe, cli, lim = make(base_url, retrievers, tmp_path, understand_mode="rules")
    draft = load_config()["models"]["draft"]
    b = lim.budget(draft)
    b.tokens = -0.9 * b.tpm  # empty bucket: the next draft must wait ~50 s
    task = asyncio.create_task(
        pipe.run("What optimizer did the Transformer use?", "r9", time.time() + 120)
    )
    await asyncio.sleep(1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await cli.aclose()
    trace = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[-1])
    waits = [s for s in trace["spans"] if s["name"] == "draft.limiter_cancelled"]
    assert waits and waits[0]["kind"] == "wait" and trace["totals"]["wait"] > 500


def test_extractive_quote_picks_the_sentence_matching_the_question():
    from relay.pipeline import _best_excerpt

    text = (
        "Figure 3: Overview of the codec. "
        "The encoder projects audio to a latent representation at 12.5 frames per second. "
        "Training uses Adam."
    )
    out = _best_excerpt(text, "How many frames per second does the codec make?")
    assert out.startswith("The encoder projects audio")
