"""The answering pipeline: understand -> retrieve -> decide -> draft -> verify.

Steps:

1. **Understand** (small model, JSON): scope check, target papers, search
   query. Skipped for keyword rules when that model's budget is tight.
2. **Retrieve** (tool call): hybrid search with optional re-rank, with its
   own timeout and a bounded re-rank concurrency (it is CPU-bound).
3. **Decide** (branch + loop): out of scope -> refuse; weak evidence -> one
   broadened re-query (drop the paper filter, use the raw question); still
   weak -> refuse without spending a draft call.
4. **Draft** (streamed): cited answer from the passages only.
5. **Verify**: every sentence is checked deterministically as it streams and
   released to the user only if it passes (``verify_mode: sentence``), or the
   whole draft is checked by a model before anything is shown (``strict``).

Degradation ladder, driven by the limiter's view of each model's budget:

* draft model's day exhausted -> fallback draft model;
* both exhausted -> extractive answer (quoted passages, zero tokens);
* draft model's per-minute headroom low -> tighter retrieval (fewer passages);
* limiter would make the request wait past its deadline -> extractive answer,
  status ``rate_limited``.

Typical usage example:

    pipe = Pipeline(cfg, caller, limiter, retrievers, TraceWriter(path))
    outcome = await pipe.run("What BLEU does the big model get?", "r1", time.time() + 10)
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from relay import prompts
from relay.limiter import Limiter
from relay.retrieval import Hit, Retriever, tokenize
from relay.retry import CallOutcome, RetryingCaller
from relay.trace import Trace, TraceWriter
from relay.verify import SentenceSplitter, check_sentence, render_citations

# Request-level status values (the ``status`` column of results.csv).
ST_OK = "ok"
ST_RATE_LIMITED = "rate_limited"
ST_TIMEOUT = "timeout"
ST_ERROR = "error"


@dataclass
class Outcome:
    """What one request produced.

    Attributes:
        request_id: Request identifier.
        question: The question asked.
        status: ``ok``, ``rate_limited``, ``timeout`` or ``error``.
        answer_kind: ``answer``, ``refusal``, ``extractive`` or ``none``.
        text: Everything shown to the user.
        first_token_ms: Time from arrival to the first user-visible answer text.
        total_ms: Time from arrival to completion.
        retries: Model-call retries across all steps.
        tokens_in: Prompt tokens across all model calls (incl. failed attempts).
        tokens_out: Completion tokens across all model calls.
        retrieval_setting: Retrieval setting actually used.
        draft_model: Model that drafted the answer, if any.
        degraded: Degradations applied (e.g. ``fallback_model``).
        refusal_reason: Why the answer is a refusal or extractive, if it is.
        citations: Citation labels used in the released answer.
    """

    request_id: str
    question: str
    status: str = ST_ERROR
    answer_kind: str = "none"
    text: str = ""
    first_token_ms: float | None = None
    total_ms: float = 0.0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retrieval_setting: str = ""
    draft_model: str | None = None
    degraded: list[str] = field(default_factory=list)
    refusal_reason: str = ""
    citations: list[str] = field(default_factory=list)


TextCallback = Callable[[str], None]


def _best_excerpt(text: str, question: str, max_chars: int = 320) -> str:
    """Returns the passage sentence(s) that best match the question.

    Sentences are scored by how many question terms (stop words removed)
    they contain; the best one is returned with its following sentence if
    that still fits. Leading sentences are often figure captions, so the
    first sentence is a poor default.

    Args:
        text: Passage text.
        question: The user's question.
        max_chars: Soft length cap.

    Returns:
        A quotable excerpt, copied verbatim from the passage.
    """
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents:
        return text[:max_chars]
    terms = set(tokenize(question))
    best = max(range(len(sents)), key=lambda i: (len(terms & set(tokenize(sents[i]))), -i))
    out = sents[best]
    if best + 1 < len(sents) and len(out) + len(sents[best + 1]) <= max_chars:
        out = f"{out} {sents[best + 1]}"
    return out[: max_chars + 80]


def _fmt_duration(seconds: float) -> str:
    h, m = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{h} h {m} min" if h else f"{max(m, 1)} min"


class Pipeline:
    """Answers questions from the library under rate limits.

    Attributes:
        cfg: Loaded configuration.
        caller: Retrying model caller (owns the client and limiter).
        limiter: Shared limiter (for degradation decisions).
        retrievers: Retriever per ``(piece_tokens, overlap)``.
        traces: Trace sink.
    """

    def __init__(
        self,
        cfg: dict,
        caller: RetryingCaller,
        limiter: Limiter,
        retrievers: dict[tuple[int, int], Retriever],
        traces: TraceWriter,
    ):
        """Initialises the pipeline.

        Args:
            cfg: Loaded configuration.
            caller: Retrying model caller.
            limiter: Shared limiter.
            retrievers: Retriever per ``(piece_tokens, overlap)``.
            traces: Trace sink.
        """
        self.cfg = cfg
        self.caller = caller
        self.limiter = limiter
        self.retrievers = retrievers
        self.traces = traces
        self.rerank_sem = asyncio.Semaphore(cfg["retrieval"].get("rerank_concurrency", 2))

    @staticmethod
    def load_retrievers(cfg: dict) -> dict[tuple[int, int], Retriever]:
        """Loads one retriever per distinct chunking in the config and warms models.

        Args:
            cfg: Loaded configuration.

        Returns:
            Retriever per ``(piece_tokens, overlap)``.
        """
        shapes = {(s["piece_tokens"], s["overlap"]) for s in cfg["retrieval"]["settings"].values()}
        out = {shape: Retriever.load(*shape) for shape in sorted(shapes)}
        next(iter(out.values())).search("warm up", rerank=True)
        return out

    # ------------------------------------------------------------------ helpers

    async def _model(
        self,
        trace: Trace,
        o: Outcome,
        step: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
        deadline_at: float,
        on_token=None,
        extra: dict | None = None,
    ) -> CallOutcome:
        """Makes one logical model call and records its spans and costs.

        Args:
            trace: Request trace.
            o: Outcome being built (tokens and retries are added).
            step: Step name.
            messages: Chat messages.
            model: Model id.
            max_tokens: Completion cap.
            deadline_at: Request deadline (unix time).
            on_token: Streaming callback.
            extra: Extra request fields.

        Returns:
            The call outcome.
        """
        start = trace.now_ms()
        # Measured: real prompt tokens ~= chars/4 plus ~110 tokens of chat-format
        # overhead (understand prompts came out at 1.46x chars/4).
        est = sum(len(m["content"]) for m in messages) // 4 + 110
        # Per-model extras (e.g. reasoning_effort) apply to every step.
        extra = {**(self.cfg.get("model_extra", {}).get(model) or {}), **(extra or {})}
        progress: dict = {}
        try:
            out = await self.caller.call(
                messages,
                model,
                max_tokens,
                deadline_at,
                est,
                on_token=on_token,
                step=step,
                tag=f"{o.request_id}:{step}",
                extra=extra,
                progress=progress,
            )
        except asyncio.CancelledError:
            # The user gave up mid-call: keep the finished attempts and say
            # which phase the request was in, so a give-up is explainable.
            self._record_attempts(trace, o, step, model, start, progress.get("outcome"))
            if progress.get("phase"):
                began = trace.now_ms() - (time.time() - progress["since"]) * 1000
                kind = {"limiter": "wait", "call": "work", "backoff": "retry"}[progress["phase"]]
                trace.add(
                    kind,
                    f"{step}.{progress['phase']}_cancelled",
                    began,
                    trace.now_ms(),
                    model=model,
                )
            raise
        self._record_attempts(trace, o, step, model, start, out)
        return out

    def _record_attempts(self, trace, o, step, model, start, out) -> None:
        """Adds an outcome's attempts to the trace and its costs to the outcome.

        Args:
            trace: Request trace.
            o: Outcome being built.
            step: Step name.
            model: Model id.
            start: Trace time (ms) when the logical call began.
            out: The call outcome so far (None if nothing happened yet).
        """
        if out is None:
            return
        t = start
        for i, a in enumerate(out.attempts):
            if a.backoff_ms:
                trace.add("retry", f"{step}.backoff", t, t + a.backoff_ms)
                t += a.backoff_ms
            if a.limiter_wait_ms > 1:
                trace.add("wait", f"{step}.limiter", t, t + a.limiter_wait_ms, model=model)
                t += a.limiter_wait_ms
            r = a.result
            kind = "work" if r.status == "ok" else "retry"
            trace.add(
                kind,
                f"{step}.attempt{i}",
                t,
                t + r.total_ms,
                model=model,
                status=r.status,
                ttft_ms=r.ttft_ms,
                first_byte_ms=r.first_byte_ms,
                tokens_in=r.tokens_in,
                tokens_out=r.tokens_out,
            )
            t += r.total_ms
        if out.refused and out.refused_wait_ms > 1:
            trace.add(
                "wait",
                f"{step}.limiter_refused",
                t,
                t + out.refused_wait_ms,
                model=model,
                reason=out.refused,
            )
        o.retries += out.retries
        o.tokens_in += out.tokens_in
        o.tokens_out += out.tokens_out

    def _emit(self, trace: Trace, o: Outcome, text: str, on_text: TextCallback | None) -> None:
        """Sends text to the user, stamping time to first token.

        Args:
            trace: Request trace.
            o: Outcome being built.
            text: Text to show.
            on_text: User-facing callback.
        """
        if o.first_token_ms is None:
            o.first_token_ms = trace.now_ms()
        o.text += text
        if on_text:
            on_text(text)

    def _refuse(self, trace, o, reason, hits, on_text) -> Outcome:
        """Finishes the request with an honest "the papers don't say" reply.

        Args:
            trace: Request trace.
            o: Outcome being built.
            reason: Machine-readable reason.
            hits: Passages that were checked (cited as "closest").
            on_text: User-facing callback.

        Returns:
            The finished outcome.
        """
        o.status, o.answer_kind, o.refusal_reason = ST_OK, "refusal", reason
        if reason == "out_of_scope":
            msg = (
                "I can only answer questions about three papers: Attention Is All You Need, "
                "Moshi, and Dynamic Chunking (H-Net). This question is outside them."
            )
        else:
            msg = "The three papers don't say this, so I won't guess."
            if hits:
                msg += (
                    " Closest passages I checked: "
                    + ", ".join(f"[{h.cite}]" for h in hits[:3])
                    + "."
                )
        self._emit(trace, o, msg, on_text)
        return o

    def _extractive(self, trace, o, hits, why, on_text, status) -> Outcome:
        """Finishes with quoted passages instead of a generated answer (0 tokens).

        Args:
            trace: Request trace.
            o: Outcome being built.
            hits: Passages to quote.
            why: ``"day"`` (daily budget gone) or ``"busy"`` (per-minute limit).
            on_text: User-facing callback.
            status: Request status to record.

        Returns:
            The finished outcome.
        """
        o.status, o.answer_kind, o.refusal_reason = status, "extractive", why
        if why == "day":
            reset = max((self.limiter.day_reset_in_s(m) or 0) for m in self._draft_models())
            when = (
                f" Capacity comes back gradually; try again in about {_fmt_duration(reset)}."
                if reset
                else ""
            )
            head = (
                "The assistant has used today's answer budget, so I can't write a new "
                f"answer right now, and I won't guess.{when}"
            )
        else:
            head = (
                "The assistant is at its request limit right now, so instead of making you "
                "wait I'm showing the source passages directly."
            )
        if hits:
            body = " Here are the passages that best match your question, quoted exactly:\n" + (
                "\n".join(
                    f'> "{_best_excerpt(h.chunk["text"], o.question)}" [{h.cite}]' for h in hits[:3]
                )
            )
            o.citations = [h.cite for h in hits[:3]]
        else:
            body = " I couldn't find a matching passage either."
        self._emit(trace, o, head + body, on_text)
        return o

    def _draft_models(self) -> list[str]:
        m = self.cfg["models"]
        return [m["draft"], m["draft_fallback"]]

    def _evidence(self, hits: list[Hit], reranked: bool) -> float:
        """Evidence strength of a retrieval result (higher is stronger).

        Args:
            hits: Retrieved passages.
            reranked: Whether scores are cross-encoder scores.

        Returns:
            Top re-rank score, or top dense cosine when not re-ranked.
        """
        if not hits:
            return float("-inf")
        return hits[0].score if reranked else max(h.dense_score for h in hits)

    # ----------------------------------------------------------------- pipeline

    async def run(
        self,
        question: str,
        request_id: str,
        deadline_at: float,
        on_text: TextCallback | None = None,
        t0: float | None = None,
        queue_wait_ms: float = 0.0,
        sink: dict | None = None,
        on_span: Callable | None = None,
    ) -> Outcome:
        """Answers one question.

        Args:
            question: The user's question.
            request_id: Identifier used in logs and traces.
            deadline_at: Unix time by which the user needs an answer.
            on_text: Receives user-visible text as it is released.
            t0: Arrival time on the ``perf_counter`` clock (defaults to now).
            queue_wait_ms: Time already spent in the admission queue (recorded
                as a ``wait`` span starting at arrival).
            sink: If given, ``sink["outcome"]`` is set to the in-progress
                outcome, so a caller that cancels the request can still read
                the tokens spent and text shown so far.
            on_span: Called with each trace span as it is recorded, so a UI can
                show the pipeline's progress live.

        Returns:
            The outcome. If the task is cancelled (the user gave up), the trace
            is still written and the cancellation propagates.
        """
        trace = Trace(request_id, t0)
        trace.on_span = on_span
        if queue_wait_ms > 1:
            trace.add("wait", "admission.queue", 0.0, queue_wait_ms)
        o = Outcome(request_id, question)
        if sink is not None:
            sink["outcome"] = o
        try:
            await self._run(question, deadline_at, on_text, trace, o)
        except asyncio.CancelledError:
            o.status = ST_TIMEOUT
            raise
        except Exception as e:  # noqa: BLE001 - one bad request must not kill the run
            o.status, o.refusal_reason = ST_ERROR, f"{type(e).__name__}: {e}"
        finally:
            o.total_ms = trace.now_ms()
            trace.attrs.update(
                question=question,
                status=o.status,
                answer_kind=o.answer_kind,
                first_token_ms=o.first_token_ms,
                total_ms=round(o.total_ms, 1),
                retries=o.retries,
                tokens_in=o.tokens_in,
                tokens_out=o.tokens_out,
                retrieval_setting=o.retrieval_setting,
                draft_model=o.draft_model,
                degraded=o.degraded,
                refusal_reason=o.refusal_reason,
            )
            self.traces.write(trace)
        return o

    async def _run(self, question, deadline_at, on_text, trace: Trace, o: Outcome) -> None:
        cfg, pcfg, models = self.cfg, self.cfg["pipeline"], self.cfg["models"]

        # -- choose draft model and retrieval setting from budget state
        draft_model = models["draft"]
        if self.limiter.day_exhausted(draft_model):
            fb = models["draft_fallback"]
            draft_model = None if self.limiter.day_exhausted(fb) else fb
            o.degraded.append("fallback_model" if draft_model else "no_draft_model")
        setting_name = cfg["retrieval"]["setting"]
        if draft_model and self.limiter.headroom(draft_model) < pcfg["tight_headroom"]:
            setting_name = pcfg["tight_setting"]
            o.degraded.append("tight_retrieval")
        setting = cfg["retrieval"]["settings"][setting_name]
        o.retrieval_setting, o.draft_model = setting_name, draft_model
        retriever = self.retrievers[(setting["piece_tokens"], setting["overlap"])]

        # -- 1. understand
        um = models["understand"]
        use_model = (
            pcfg["understand_mode"] == "model"
            and self.limiter.headroom(um) >= pcfg["understand_min_headroom"]
        )
        u = prompts.rule_understanding(question)
        if use_model:
            out = await self._model(
                trace,
                o,
                "understand",
                prompts.understand_messages(question),
                um,
                pcfg["understand_max_tokens"],
                deadline_at,
                extra={"response_format": {"type": "json_object"}},
            )
            if out.ok:
                u = prompts.parse_understanding(out.final.text, question)
            else:
                o.degraded.append("rules_understand")
        elif pcfg["understand_mode"] == "model":
            o.degraded.append("rules_understand")
        trace.attrs["understanding"] = u
        if not u["in_scope"]:
            self._refuse(trace, o, "out_of_scope", [], on_text)
            return

        # -- 2. retrieve (tool) + 3. decide, with one broadened re-query
        # A paper the user named explicitly always constrains retrieval, even if
        # the understand model returned no filter (cross-paper traps, e.g. GPUs).
        named = prompts.rule_understanding(question)["papers"]
        query, papers = u["query"], (u["papers"] or named)
        hits: list[Hit] = []
        strong = False
        rerank = setting["rerank"]
        for loop in range(pcfg["max_requery_loops"] + 1):
            t_wait = trace.now_ms()
            if rerank:
                await self.rerank_sem.acquire()
                if trace.now_ms() - t_wait > 1:
                    trace.add("wait", "retrieve.rerank_queue", t_wait, trace.now_ms())
            try:
                t_tool = trace.now_ms()
                rr = await retriever.search_tool(
                    query,
                    setting["fetch"],
                    setting["candidates"],
                    rerank,
                    cfg["retrieval"]["tool_timeout_s"],
                    papers,
                )
            finally:
                if rerank:
                    self.rerank_sem.release()
            trace.add(
                "work" if rr.status == "ok" else "retry",
                f"retrieve.{loop}",
                t_tool,
                trace.now_ms(),
                status=rr.status,
                query=query,
                papers=papers,
                hits=[h.chunk["chunk_id"] for h in rr.hits],
            )
            if rr.status == "timeout":
                o.degraded.append("retrieval_timeout")
                rerank = False  # the CPU re-ranker is the slow part; retry without it
                continue
            hits = rr.hits
            ev = self._evidence(hits, rerank)
            trace.attrs.setdefault("evidence", []).append(round(ev, 3))
            threshold = pcfg["weak_rerank"] if rerank else pcfg["weak_dense"]
            strong = ev >= threshold
            if strong:
                break
            if loop < pcfg["max_requery_loops"]:
                query, papers = question, []
                o.degraded.append("requery")
        if not strong:
            self._refuse(trace, o, "weak_evidence", hits, on_text)
            return

        # -- extractive when no model budget is left today
        if draft_model is None:
            self._extractive(trace, o, hits, "day", on_text, ST_RATE_LIMITED)
            return

        # -- 4. draft (+ 5. verify)
        await self._draft_and_verify(question, hits, draft_model, deadline_at, on_text, trace, o)

    async def _draft_and_verify(self, question, hits, model, deadline_at, on_text, trace, o):
        pcfg = self.cfg["pipeline"]
        mode = pcfg["verify_mode"]
        labels = [h.cite for h in hits]
        passages = [h.chunk["text"] for h in hits]
        passage_papers = [h.chunk["paper"] for h in hits]
        named = prompts.rule_understanding(question)["papers"]
        splitter = SentenceSplitter()
        raw: list[str] = []
        released: list[str] = []
        held: list[str] = []
        pending: list[str] = []  # strict mode: passed sentences awaiting model verify

        def handle(sentence: str) -> None:
            if mode == "off":
                released.append(sentence)
                self._emit(trace, o, render_citations(sentence, labels) + " ", on_text)
                return
            chk = check_sentence(sentence, passages)
            if (
                chk.ok
                and len(named) == 1
                and chk.cited
                and all(passage_papers[c - 1] != named[0] for c in chk.cited)
            ):
                chk.ok, chk.reason = False, "wrong_paper"
            if not chk.ok:
                held.append(f"{chk.reason}: {sentence}")
            elif mode == "strict":
                pending.append(sentence)
            else:
                released.append(sentence)
                self._emit(trace, o, render_citations(sentence, labels) + " ", on_text)

        gate = {"open": False}  # stays shut while the reply could still be NOT_IN_PAPERS

        def on_token(piece: str) -> None:
            raw.append(piece)
            if not gate["open"]:
                so_far = "".join(raw).lstrip()
                if prompts.NOT_IN_PAPERS.startswith(so_far[: len(prompts.NOT_IN_PAPERS)]):
                    return
                gate["open"] = True
                piece = "".join(raw)  # release everything buffered so far
            for s in splitter.feed(piece):
                handle(s)

        out = await self._model(
            trace,
            o,
            "draft",
            prompts.draft_messages(question, hits),
            model,
            pcfg["draft_max_tokens"],
            deadline_at,
            on_token=on_token,
        )
        text = "".join(raw).strip()
        if not out.ok and not released:
            final = out.final.status if out.final else ""
            if out.refused in ("deadline", "budget", "day") or final.startswith("rate_limited"):
                why = "day" if (out.refused == "day" or final == "rate_limited_day") else "busy"
                self._extractive(trace, o, hits, why, on_text, ST_RATE_LIMITED)
            else:
                o.status = ST_TIMEOUT if final == "timeout" else ST_ERROR
                o.refusal_reason = final or out.refused
            return
        if text.startswith(prompts.NOT_IN_PAPERS):
            self._refuse(trace, o, "model_not_in_papers", hits, on_text)
            return
        if not gate["open"]:
            for s in splitter.feed(text):
                handle(s)
        for s in splitter.flush():
            handle(s)

        if mode == "strict" and pending:
            vout = await self._model(
                trace,
                o,
                "verify",
                prompts.verify_messages(" ".join(pending), hits),
                self.cfg["models"]["verify"],
                150,
                deadline_at,
                extra={"response_format": {"type": "json_object"}},
            )
            supported = True
            if vout.ok:
                verdict = prompts.parse_json(vout.final.text)
                supported = bool(verdict.get("supported", True))
            else:
                o.degraded.append("verify_skipped")
            if supported:
                for s in pending:
                    released.append(s)
                    self._emit(trace, o, render_citations(s, labels) + " ", on_text)
            else:
                held.extend(f"model_unsupported: {s}" for s in pending)

        trace.attrs["held_sentences"] = held
        if held:
            o.degraded.append(f"dropped_sentences:{len(held)}")
        if not released:
            self._refuse(trace, o, "unverifiable_draft", hits, on_text)
            return
        o.status, o.answer_kind = ST_OK, "answer"
        o.citations = sorted(
            {
                lab
                for s in released
                for lab in re.findall(r"\[([^\]]+)\]", render_citations(s, labels))
            }
        )
        if not out.ok:
            o.degraded.append("truncated")
