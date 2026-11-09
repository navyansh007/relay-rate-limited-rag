"""Hybrid passage retrieval over the library, exposed to the pipeline as a tool.

Each query goes through three stages:

1. **Candidates**: BM25 (exact terms, numbers, names like "Mimi") and dense
   embeddings (paraphrases: "audio tokenizer" vs "neural audio codec") are
   fused with reciprocal-rank fusion (RRF).
2. **Optional paper filter**: the understand step may name target papers.
3. **Optional re-rank**: a local cross-encoder rescores the candidates.

Embeddings and re-ranking run locally on CPU, so they cost latency but no API
tokens. Under token pressure the pipeline should re-rank harder and send
*fewer* passages.

Typical usage example:

    retriever = Retriever.load(piece_tokens=300, overlap=60)
    hits = await retriever.search_tool("How many heads?", fetch=4, rerank=True,
                                       candidates=20, timeout_s=2.0)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from relay.config import DATA_DIR

INDEX_DIR = DATA_DIR / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "with",
        "as",
        "at",
        "from",
        "that",
        "this",
        "what",
        "which",
        "how",
        "does",
        "do",
        "did",
        "it",
        "its",
        "their",
        "they",
        "we",
        "our",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lower-cases and tokenises text for BM25, keeping numbers like ``41.8``.

    Args:
        text: Input text.

    Returns:
        Tokens with common stop words removed.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    """One retrieved passage.

    Attributes:
        chunk: The chunk record from the index (id, paper, section, page, text).
        score: Final score (re-rank score if re-ranked, else RRF score).
        bm25_rank: Rank in the BM25 list (0-based), or None if absent.
        dense_rank: Rank in the dense list (0-based), or None if absent.
    """

    chunk: dict
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None

    @property
    def cite(self) -> str:
        """Human-readable citation label, e.g. ``moshi §3.3 p.9``."""
        c = self.chunk
        sec = "abstract" if c["section"] == "abstract" else f"§{c['section']}"
        return f"{c['paper']} {sec} p.{c['page']}"


@dataclass
class RetrievalResult:
    """Output of one tool invocation.

    Attributes:
        hits: Passages in final order.
        status: ``"ok"`` or ``"timeout"``.
        elapsed_ms: Wall time of the tool call.
        top_rerank: Highest re-rank score (None if not re-ranked); used by the
            decide step as an evidence-strength signal.
    """

    hits: list[Hit]
    status: str
    elapsed_ms: float
    top_rerank: float | None = None


class Retriever:
    """In-memory hybrid index for one chunking configuration.

    Attributes:
        chunks: Chunk records.
        piece_tokens: Piece size the index was built with.
    """

    _embedder = None
    _reranker = None

    def __init__(self, chunks: list[dict], embeddings: np.ndarray, piece_tokens: int):
        """Initialises the index.

        Args:
            chunks: Chunk records from ``relay.ingest``.
            embeddings: L2-normalised chunk embeddings, one row per chunk.
            piece_tokens: Piece size, kept for reporting.
        """
        self.chunks = chunks
        self.piece_tokens = piece_tokens
        self._emb = embeddings
        self._bm25 = BM25Okapi([tokenize(self._doc_text(c)) for c in chunks])

    @staticmethod
    def _doc_text(chunk: dict) -> str:
        """Text that is indexed for a chunk: section title plus body.

        Args:
            chunk: Chunk record.

        Returns:
            Indexable text.
        """
        return f"{chunk['section_title']}. {chunk['text']}"

    @classmethod
    def embedder(cls):
        """Returns the shared embedding model, loading it on first use."""
        if cls._embedder is None:
            from fastembed import TextEmbedding

            cls._embedder = TextEmbedding(EMBED_MODEL)
        return cls._embedder

    @classmethod
    def reranker(cls):
        """Returns the shared cross-encoder, loading it on first use."""
        if cls._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            cls._reranker = TextCrossEncoder(RERANK_MODEL)
        return cls._reranker

    @classmethod
    def load(cls, piece_tokens: int, overlap: int) -> Retriever:
        """Loads chunks for a setting and their embeddings (computed once, cached).

        Args:
            piece_tokens: Piece size of the chunk file to load.
            overlap: Overlap of the chunk file to load.

        Returns:
            A ready retriever.

        Raises:
            FileNotFoundError: If ``relay.ingest`` has not been run.
        """
        path = INDEX_DIR / f"chunks_{piece_tokens}_{overlap}.jsonl"
        with open(path) as f:
            chunks = [json.loads(line) for line in f]
        emb_path = INDEX_DIR / f"emb_{piece_tokens}_{overlap}.npy"
        if emb_path.exists():
            emb = np.load(emb_path)
        else:
            emb = np.array(list(cls.embedder().embed([cls._doc_text(c) for c in chunks])))
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            np.save(emb_path, emb)
        return cls(chunks, emb, piece_tokens)

    def search(
        self,
        query: str,
        fetch: int = 4,
        candidates: int = 20,
        rerank: bool = False,
        papers: list[str] | None = None,
    ) -> list[Hit]:
        """Runs a synchronous hybrid search.

        Args:
            query: Search query (ideally the rewritten query from step 1).
            fetch: Number of passages to return.
            candidates: Size of the fused candidate pool before re-ranking.
            rerank: Whether to rescore candidates with the cross-encoder.
            papers: Restrict to these paper keys; None or empty means all.

        Returns:
            Up to ``fetch`` hits, best first.
        """
        allowed = np.ones(len(self.chunks), dtype=bool)
        if papers:
            allowed = np.array([c["paper"] in papers for c in self.chunks])

        bm = self._bm25.get_scores(tokenize(query))
        bm[~allowed] = -np.inf
        qv = np.array(list(self.embedder().query_embed([query])))[0]
        qv /= np.linalg.norm(qv)
        dense = self._emb @ qv
        dense[~allowed] = -np.inf

        pool = max(candidates, fetch) * 2
        bm_order = [i for i in np.argsort(-bm)[:pool] if np.isfinite(bm[i]) and bm[i] > 0]
        de_order = [i for i in np.argsort(-dense)[:pool] if np.isfinite(dense[i])]
        bm_rank = {int(i): r for r, i in enumerate(bm_order)}
        de_rank = {int(i): r for r, i in enumerate(de_order)}
        fused: dict[int, float] = {}
        for ranks in (bm_rank, de_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        top = sorted(fused, key=fused.get, reverse=True)[: max(candidates, fetch)]
        hits = [Hit(self.chunks[i], fused[i], bm_rank.get(i), de_rank.get(i)) for i in top]

        if rerank and hits:
            docs = [self._doc_text(h.chunk) for h in hits]
            scores = list(self.reranker().rerank(query, docs))
            for h, s in zip(hits, scores, strict=True):
                h.score = float(s)
            hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:fetch]

    async def search_tool(
        self,
        query: str,
        fetch: int,
        candidates: int,
        rerank: bool,
        timeout_s: float,
        papers: list[str] | None = None,
    ) -> RetrievalResult:
        """Runs the search as a pipeline tool call with a timeout.

        The search runs in a worker thread so the event loop keeps serving
        other requests. On timeout the result is empty with status
        ``"timeout"``; the pipeline then decides whether to refuse.

        Args:
            query: Search query.
            fetch: Number of passages to return.
            candidates: Candidate pool size.
            rerank: Whether to re-rank.
            timeout_s: Tool timeout.
            papers: Optional paper filter.

        Returns:
            The tool result with timing.
        """
        t0 = time.perf_counter()
        try:
            hits = await asyncio.wait_for(
                asyncio.to_thread(self.search, query, fetch, candidates, rerank, papers),
                timeout=timeout_s,
            )
            status = "ok"
        except TimeoutError:
            hits, status = [], "timeout"
        top = hits[0].score if (rerank and hits) else None
        return RetrievalResult(hits, status, (time.perf_counter() - t0) * 1000, top)


def _cli() -> None:
    """Ad-hoc retrieval check: ``python -m relay.retrieval "question" [piece] [overlap]``."""
    import sys

    q = sys.argv[1]
    piece = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    r = Retriever.load(piece, overlap)
    for rerank in (False, True):
        t0 = time.perf_counter()
        hits = r.search(q, fetch=4, candidates=20, rerank=rerank)
        print(f"\n== rerank={rerank} ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        for h in hits:
            print(
                f"  {h.score:7.3f} bm={h.bm25_rank} de={h.dense_rank} [{h.cite}] "
                f"{h.chunk['text'][:110]!r}"
            )


if __name__ == "__main__":
    _cli()
