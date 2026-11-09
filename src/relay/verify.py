"""Deterministic checks on drafted sentences, cheap enough to run per sentence.

A sentence passes when:

* it cites at least one passage, and every cited number is a real passage; or
  it is a "the papers do not say/cover ..." disclaimer (no citation needed);
* every number it states (other than citation markers and section numbers)
  appears in at least one passage it cites.

The number check targets the most damaging failure for this library: a
plausible but wrong figure (e.g. a BLEU score or a frame rate from memory of
another paper version).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITE = re.compile(r"\[(\d+)\]")
# Numbers may be glued to units ("12.5Hz", "1.1kbps", "7B"); the lookbehind
# keeps digits inside names ("H100", "P100") out.
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*")
_SECTION_REF = re.compile(
    r"(§|section|table|figure|fig\.|eq\.|equation|appendix)\s*[\w.]+", re.IGNORECASE
)
_DISCLAIMER = re.compile(
    r"\b(papers?|passages?)\b.*\b(do(es)? not|don't|doesn't|not)\b.*"
    r"\b(say|state|mention|cover|specify|report|provide|include|address|discuss)",
    re.IGNORECASE,
)
# A sentence ends at . ! or ? optionally followed by citations, then whitespace.
_SENT_END = re.compile(r"[.!?](?:\s*\[\d+\])*(?=\s+\S)")
# gpt-oss cites in its native style, e.g. "【3】" or "【3†L1-L4】"; others write
# "[3†...]" or "[1, 2]". All are normalised to "[3]" / "[1][2]".
_NATIVE_CITE = re.compile(r"[【\[]\s*(\d+(?:\s*,\s*\d+)*)\s*(?:†[^】\]]*)?[】\]]")
_OPEN_CITE = re.compile(r"\s*(【[^】]*|\[[\d,\s†]*)$")


def normalise_citations(text: str) -> str:
    """Rewrites model-native citation markers into ``[n]`` form.

    Args:
        text: Model output.

    Returns:
        Text with every citation as one or more ``[n]`` markers, and
        non-breaking spaces replaced by plain spaces.
    """

    def sub(m: re.Match) -> str:
        return "".join(f"[{n.strip()}]" for n in m.group(1).split(","))

    return _NATIVE_CITE.sub(sub, text).replace("\u202f", " ").replace("\u00a0", " ")


@dataclass
class SentenceCheck:
    """Verdict for one sentence.

    Attributes:
        text: The sentence.
        ok: Whether it passed.
        reason: Why it failed (``"no_citation"``, ``"bad_citation"``,
            ``"number_not_in_cited"``), or ``""``.
        cited: Passage numbers cited.
    """

    text: str
    ok: bool
    reason: str
    cited: list[int]


def _norm_num(s: str) -> str:
    return s.replace(",", "")


def check_sentence(sentence: str, passages: list[str]) -> SentenceCheck:
    """Checks one drafted sentence against the passages.

    Args:
        sentence: Sentence text including its ``[n]`` citations.
        passages: Passage texts, where passage ``n`` is ``passages[n-1]``.

    Returns:
        The verdict.
    """
    cited = [int(c) for c in _CITE.findall(sentence)]
    if not cited:
        ok = bool(_DISCLAIMER.search(sentence))
        return SentenceCheck(sentence, ok, "" if ok else "no_citation", [])
    if any(c < 1 or c > len(passages) for c in cited):
        return SentenceCheck(sentence, False, "bad_citation", cited)
    body = _SECTION_REF.sub(" ", _CITE.sub(" ", sentence))
    cited_text = " ".join(passages[c - 1] for c in cited)
    cited_nums = {_norm_num(n) for n in _NUM.findall(cited_text)}
    for n in _NUM.findall(body):
        if _norm_num(n) not in cited_nums:
            return SentenceCheck(sentence, False, "number_not_in_cited", cited)
    return SentenceCheck(sentence, True, "", cited)


class SentenceSplitter:
    """Incrementally splits streamed text into complete sentences.

    Citations that follow the full stop (``"... BLEU. [1]"``) stay attached to
    their sentence.
    """

    def __init__(self) -> None:
        """Initialises an empty buffer."""
        self.buf = ""

    def feed(self, piece: str) -> list[str]:
        """Adds streamed text and returns any sentences now complete.

        Args:
            piece: Newly streamed text.

        Returns:
            Complete sentences, in order.
        """
        self.buf = normalise_citations(self.buf + piece)
        out = []
        while True:
            m = _SENT_END.search(self.buf)
            if not m:
                return out
            if _OPEN_CITE.match(self.buf[m.end() :]):
                return out  # a citation for this sentence is still streaming in
            out.append(self.buf[: m.end()].strip())
            self.buf = self.buf[m.end() :]

    def flush(self) -> list[str]:
        """Returns whatever remains at end of stream.

        Returns:
            The final partial sentence, if any.
        """
        rest, self.buf = normalise_citations(self.buf).strip(), ""
        return [rest] if rest else []


def render_citations(sentence: str, labels: list[str]) -> str:
    """Replaces ``[n]`` markers with human-readable locations.

    Args:
        sentence: Sentence with numeric citations.
        labels: Citation label per passage (``labels[n-1]``).

    Returns:
        Sentence with e.g. ``[moshi §3.3 p.9]``.
    """

    def sub(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[{labels[n - 1]}]" if 1 <= n <= len(labels) else m.group(0)

    out = re.sub(r"\]\s*\[", "][", _CITE.sub(sub, sentence))
    return re.sub(r"(?<=[^\s\[\]])\[", " [", out)
