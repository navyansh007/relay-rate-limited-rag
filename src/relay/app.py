"""Wires the components together and provides a one-question CLI.

``build`` is shared by the CLI, the evaluation runner and the load generator
so every entry point uses identical wiring.

Typical usage example:

    uv run python -m relay.app "How many heads does the Transformer use?"          # stand-in
    uv run python -m relay.app "What frame rate does Mimi run at?" --real          # Groq
    uv run python -m relay.app "..." --set pipeline.verify_mode=strict --set retrieval.setting=s1_small_many
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass

import yaml

from relay.config import RAW_DIR, groq_api_key, load_config
from relay.limiter import Limiter
from relay.llm_client import CallLogger, LLMClient
from relay.pipeline import Pipeline
from relay.retrieval import Retriever
from relay.retry import RetryingCaller, RetryPolicy
from relay.trace import TraceWriter


@dataclass
class System:
    """A fully wired pipeline and the resources it owns.

    Attributes:
        cfg: Configuration in use.
        source: ``"real"`` or ``"stand-in"``.
        client: Model client.
        limiter: Shared limiter.
        pipeline: The pipeline.
    """

    cfg: dict
    source: str
    client: LLMClient
    limiter: Limiter
    pipeline: Pipeline

    async def aclose(self) -> None:
        """Releases network resources."""
        await self.client.aclose()


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """Applies ``dotted.key=value`` overrides (values parsed as YAML).

    Args:
        cfg: Configuration to modify in place.
        sets: Overrides such as ``"retry.policy=naive"``.

    Returns:
        The same configuration object.
    """
    for item in sets:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def build(
    cfg: dict,
    real: bool,
    retrievers: dict[tuple[int, int], Retriever] | None = None,
    traces_path=RAW_DIR / "traces.jsonl",
    limiter: Limiter | None = None,
) -> System:
    """Builds a pipeline against the real service or the stand-in.

    Args:
        cfg: Configuration.
        real: Use Groq (True) or the local stand-in (False).
        retrievers: Pre-loaded retrievers to share between systems.
        traces_path: Where per-request traces go.
        limiter: Limiter to share with other systems using the same API key.
            Separate limiters on one key each believe they own the whole
            budget (this caused 26 of 60 TPM 429s in the first eval run).

    Returns:
        The wired system.
    """
    p = cfg["provider"]
    source = "real" if real else "stand-in"
    client = LLMClient(
        p["real_base_url"] if real else p["standin_base_url"],
        api_key=groq_api_key() if real else "",
        source=source,
        connect_timeout_s=p["connect_timeout_s"],
        first_token_timeout_s=p["first_token_timeout_s"],
        stall_timeout_s=p["stall_timeout_s"],
        logger=CallLogger(RAW_DIR / ("groq_calls.jsonl" if real else "standin_client_calls.jsonl")),
    )
    limiter = limiter or Limiter(cfg["limits"], enabled=cfg.get("limiter", {}).get("enabled", True))
    caller = RetryingCaller(client, limiter, RetryPolicy.from_config(cfg["retry"]))
    pipe = Pipeline(
        cfg, caller, limiter, retrievers or Pipeline.load_retrievers(cfg), TraceWriter(traces_path)
    )
    return System(cfg, source, client, limiter, pipe)


async def _ask(question: str, real: bool, sets: list[str], patience_s: float) -> None:
    cfg = apply_overrides(load_config(), sets)
    system = build(cfg, real, traces_path=RAW_DIR / "traces_adhoc.jsonl")
    print(f"[{system.source}] {question}\n")
    try:
        o = await system.pipeline.run(
            question,
            f"ask-{uuid.uuid4().hex[:6]}",
            time.time() + patience_s,
            on_text=lambda t: (sys.stdout.write(t), sys.stdout.flush()),
        )
    finally:
        await system.aclose()
    print("\n\n---")
    summary = {
        k: getattr(o, k)
        for k in (
            "status",
            "answer_kind",
            "first_token_ms",
            "total_ms",
            "retries",
            "tokens_in",
            "tokens_out",
            "retrieval_setting",
            "draft_model",
            "degraded",
            "refusal_reason",
        )
    }
    print(json.dumps(summary, indent=1, default=str))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Ask the library one question.")
    ap.add_argument("question")
    ap.add_argument("--real", action="store_true", help="use Groq instead of the stand-in")
    ap.add_argument("--set", action="append", default=[], help="config override key=value")
    ap.add_argument("--patience", type=float, default=20.0, help="seconds before giving up")
    a = ap.parse_args()
    asyncio.run(_ask(a.question, a.real, a.set, a.patience))


if __name__ == "__main__":
    main()
