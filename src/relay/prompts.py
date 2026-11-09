"""Prompt templates and the rule-based fallback router.

Kept in one place so prompt changes are reviewable and replay keys (hashes of
the exact messages) stay stable between real and stand-in runs.
"""

from __future__ import annotations

import json
import re

PAPER_NAMES = {
    "transformer": '"Attention Is All You Need" (Vaswani et al., the Transformer, arXiv 1706.03762v7)',
    "moshi": '"Moshi: a speech-text foundation model for real-time dialogue" (Kyutai; Mimi codec, '
    "Helium LM, Inner Monologue; arXiv 2410.00037v1)",
    "hnet": '"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (H-Net; arXiv '
    "2507.07955v2)",
}

UNDERSTAND_SYSTEM = (
    "You route questions for an assistant that answers ONLY from these three papers:\n"
    + "\n".join(f"- {k}: {v}" for k, v in PAPER_NAMES.items())
    + "\n\nReturn a JSON object with keys:\n"
    '  "in_scope": false only if the question is clearly unrelated to machine learning '
    "research of this kind (e.g. weather, cooking, personal advice); otherwise true.\n"
    '  "papers": list of paper keys the question is about; [] if unsure or general.\n'
    '  "query": a search query for the papers, using the papers\' own terminology.\n'
    '  "multi_part": true if the question asks for several distinct facts.\n'
    "Return only the JSON object."
)

DRAFT_SYSTEM = (
    "You answer engineers' questions using ONLY the numbered passages, which come from "
    "three research papers. Rules:\n"
    "1. End every sentence with citations like [1] or [2][3] naming the passages that "
    "support it.\n"
    "2. Use only facts stated in the passages. Never use outside knowledge, even if you "
    "know the paper.\n"
    "3. Copy numbers exactly as written in the passages.\n"
    "4. If the passages do not contain the answer, reply with exactly: NOT_IN_PAPERS\n"
    "5. If they answer only part of the question, answer that part and add one sentence "
    "saying which part the papers do not cover (no citation needed for that sentence).\n"
    "6. At most 4 sentences. No preamble."
)

VERIFY_SYSTEM = (
    "You check an answer against the passages it cites. For each sentence, decide whether "
    "every fact in it is stated in the cited passages. Return a JSON object: "
    '{"supported": true|false, "unsupported_claims": ["..."]}. Return only the JSON.'
)

NOT_IN_PAPERS = "NOT_IN_PAPERS"

# Lexical routing used when the understand model is skipped (budget pressure)
# or fails. Terms are distinctive to one paper.
_ROUTES = {
    "transformer": r"\b(transformer (base|big)|attention is all you need|vaswani|wmt|bleu|"
    r"en-de|en-fr|english-(german|french)|constituency pars|positional encoding|"
    r"label smoothing|scaled dot-product)\b",
    "moshi": r"\b(moshi|mimi|helium|inner monologue|kyutai|full-duplex|speech|audio|"
    r"codec|rvq|depth transformer|watermark)",
    "hnet": r"\b(h-net|hnet|dynamic chunking|chunking|dechunk|routing module|smoothing module|"
    r"ratio loss|byte-level|tokenizer-free|bpb|bits-per-byte)",
}
_OFF_TOPIC = re.compile(r"\b(weather|recipe|stock price|football|horoscope)\b", re.IGNORECASE)


def understand_messages(question: str) -> list[dict]:
    """Builds messages for the understand step.

    Args:
        question: The user's question.

    Returns:
        Chat messages.
    """
    return [
        {"role": "system", "content": UNDERSTAND_SYSTEM},
        {"role": "user", "content": question},
    ]


def parse_json(text: str) -> dict:
    """Extracts the first JSON object from model output.

    Args:
        text: Model output, possibly with surrounding prose.

    Returns:
        The parsed object, or an empty dict if none could be parsed.
    """
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_understanding(text: str, question: str) -> dict:
    """Parses the understand step's JSON, falling back to rules on bad output.

    Args:
        text: Model output.
        question: Original question (for fallback).

    Returns:
        Dict with ``in_scope``, ``papers``, ``query``, ``multi_part``, ``source``.
    """
    data = parse_json(text)
    if not data:
        return rule_understanding(question)
    try:
        papers = [p for p in data.get("papers", []) if p in PAPER_NAMES]
        query = str(data.get("query") or question).strip() or question
        return {
            "in_scope": bool(data.get("in_scope", True)),
            "papers": papers,
            "query": query,
            "multi_part": bool(data.get("multi_part", False)),
            "source": "model",
        }
    except (ValueError, AttributeError, TypeError):
        return rule_understanding(question)


def rule_understanding(question: str) -> dict:
    """Routes a question with keyword rules (no model call).

    Args:
        question: The user's question.

    Returns:
        Same shape as ``parse_understanding``.
    """
    q = question.lower()
    papers = [k for k, pat in _ROUTES.items() if re.search(pat, q)]
    return {
        "in_scope": not _OFF_TOPIC.search(q),
        "papers": papers if len(papers) == 1 else [],
        "query": question,
        "multi_part": " and " in q and "?" in q,
        "source": "rules",
    }


def format_passages(hits: list) -> str:
    """Renders retrieved hits as numbered, labelled passages.

    Args:
        hits: ``relay.retrieval.Hit`` objects.

    Returns:
        Passage block for the draft prompt.
    """
    return "\n\n".join(f"[{i}] ({h.cite}) {h.chunk['text']}" for i, h in enumerate(hits, 1))


def draft_messages(question: str, hits: list) -> list[dict]:
    """Builds messages for the draft step.

    Args:
        question: The user's question.
        hits: Retrieved passages.

    Returns:
        Chat messages.
    """
    user = f"Passages:\n{format_passages(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": user}]


def verify_messages(answer: str, hits: list) -> list[dict]:
    """Builds messages for the model-based verify step.

    Args:
        answer: Draft answer with ``[n]`` citations.
        hits: Passages the answer may cite.

    Returns:
        Chat messages.
    """
    user = f"Passages:\n{format_passages(hits)}\n\nAnswer:\n{answer}"
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]
