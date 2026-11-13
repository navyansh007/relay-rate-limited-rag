"""Web UI: watch a question go through the pipeline, live or from a recording.

Two modes share one page:

* **Replay** plays back requests that really ran against Groq, with their
  recorded spans, passages and rate-limit headers (``static/demo.json``,
  built by ``scripts/build_demo_data.py``). Needs no key and no index.
* **Live** runs the real pipeline and streams every span, released sentence
  and budget reading to the browser as server-sent events. It uses the local
  stand-in by default, or Groq with ``--real``.

Typical usage example:

    uv run python -m relay.ui                 # replay + live against the stand-in
    uv run python -m relay.ui --real          # live against Groq (needs GROQ_API_KEY)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from relay.config import DATA_DIR, RAW_DIR, load_config

STATIC = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 400


class Ask(BaseModel):
    """Body of a live question.

    Attributes:
        question: The question to answer.
    """

    question: str


def _cite(c: dict) -> str:
    sec = "abstract" if c["section"] == "abstract" else f"§{c['section']}"
    return f"{c['paper']} {sec} p.{c['page']}"


class LiveRunner:
    """Builds the pipeline on first use and streams one request at a time.

    Attributes:
        real: Whether live questions go to Groq instead of the stand-in.
    """

    def __init__(self, real: bool):
        """Initialises the runner without loading anything yet.

        Args:
            real: Send live questions to Groq instead of the stand-in.
        """
        self.real = real
        self._system = None
        self._chunks: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    def index_ready(self) -> bool:
        """Whether the paper index exists (``python -m relay.ingest``).

        Returns:
            True if live mode can retrieve passages.
        """
        return any((DATA_DIR / "index").glob("chunks_*.jsonl"))

    async def standin_up(self) -> bool:
        """Whether the local stand-in answers.

        Returns:
            True if reachable (always True in real mode).
        """
        if self.real:
            return True
        base = load_config()["provider"]["standin_base_url"].rsplit("/openai", 1)[0]
        try:
            async with httpx.AsyncClient(timeout=1.0) as c:
                return (await c.get(f"{base}/admin/stats")).status_code == 200
        except httpx.HTTPError:
            return False

    async def system(self):
        """Builds (once) and returns the wired pipeline.

        Returns:
            The ``relay.app.System``.
        """
        async with self._lock:
            if self._system is None:
                from relay.app import build  # heavy imports only when live is used

                self._system = await asyncio.to_thread(
                    build, load_config(), self.real, None, RAW_DIR / "traces_ui.jsonl"
                )
                for r in self._system.pipeline.retrievers.values():
                    self._chunks.update({c["chunk_id"]: c for c in r.chunks})
            return self._system

    def budget(self) -> dict[str, dict]:
        """Current per-minute token budget per model, as the limiter sees it.

        Returns:
            ``{model: {"remaining": tokens, "limit": tpm}}``.
        """
        if self._system is None:
            return {}
        out = {}
        for model, b in self._system.limiter.budgets.items():
            b.refill(time.time())
            out[model.split("/")[-1]] = {"remaining": max(0, int(b.tokens)), "limit": b.tpm}
        return out

    async def stream(self, question: str):
        """Runs one question and yields server-sent events.

        Args:
            question: The user's question.

        Yields:
            ``data: {...}`` lines: ``start``, ``span``, ``text``, ``budget``, ``done``.
        """
        system = await self.system()
        q: asyncio.Queue = asyncio.Queue()
        t0 = time.perf_counter()

        def now() -> float:
            return round((time.perf_counter() - t0) * 1000, 1)

        def on_span(span) -> None:
            ev = {
                "type": "span",
                "kind": span.kind,
                "name": span.name,
                "start_ms": span.start_ms,
                "end_ms": span.end_ms,
                "attrs": span.attrs,
            }
            hits = span.attrs.get("hits") if span.name.startswith("retrieve") else None
            if hits:
                ev["passages"] = [
                    {"cite": _cite(self._chunks[h]), "text": self._chunks[h]["text"][:520]}
                    for h in hits
                    if h in self._chunks
                ]
            q.put_nowait(ev)
            q.put_nowait({"type": "budget", "t": now(), "models": self.budget()})

        def on_text(text: str) -> None:
            q.put_nowait({"type": "text", "t": now(), "text": text})

        async def run():
            try:
                o = await system.pipeline.run(
                    question,
                    f"ui-{uuid.uuid4().hex[:6]}",
                    time.time() + 30,
                    on_text=on_text,
                    t0=t0,
                    on_span=on_span,
                )
                q.put_nowait(
                    {
                        "type": "done",
                        "t": now(),
                        "status": o.status,
                        "answer_kind": o.answer_kind,
                        "first_token_ms": o.first_token_ms,
                        "total_ms": o.total_ms,
                        "tokens_in": o.tokens_in,
                        "tokens_out": o.tokens_out,
                        "degraded": o.degraded,
                        "setting": o.retrieval_setting,
                        "refusal_reason": o.refusal_reason,
                    }
                )
            except Exception as e:  # noqa: BLE001 - report to the page, keep serving
                q.put_nowait({"type": "done", "t": now(), "status": "error", "error": str(e)})

        task = asyncio.create_task(run())
        yield _sse({"type": "start", "source": system.source, "budget": self.budget()})
        try:
            while True:
                ev = await q.get()
                yield _sse(ev)
                if ev["type"] == "done":
                    break
        finally:
            if not task.done():
                task.cancel()


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"


def create_app(real: bool = False) -> FastAPI:
    """Creates the UI application.

    Args:
        real: Send live questions to Groq instead of the stand-in.

    Returns:
        The FastAPI app.
    """
    app = FastAPI(title="Relay")
    runner = LiveRunner(real)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def status() -> dict:
        index_ok = runner.index_ready()
        backend_ok = await runner.standin_up()
        reason = ""
        if not index_ok:
            reason = "Build the paper index first: uv run python -m relay.ingest"
        elif not backend_ok:
            reason = "Start the stand-in first: uv run python -m relay.standin"
        return {
            "live": index_ok and backend_ok,
            "source": "real Groq" if real else "stand-in",
            "reason": reason,
        }

    @app.post("/api/ask")
    async def ask(body: Ask) -> StreamingResponse:
        question = body.question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            raise HTTPException(400, f"question must be 1-{MAX_QUESTION_CHARS} characters")
        if not runner.index_ready():
            raise HTTPException(503, "paper index missing: run python -m relay.ingest")
        return StreamingResponse(runner.stream(question), media_type="text/event-stream")

    return app


def main() -> None:
    """CLI entry point."""
    import uvicorn

    ap = argparse.ArgumentParser(description="Relay web UI.")
    ap.add_argument("--real", action="store_true", help="live questions go to Groq")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    print(f"Relay UI on http://{a.host}:{a.port}  (live: {'Groq' if a.real else 'stand-in'})")
    uvicorn.run(create_app(a.real), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
