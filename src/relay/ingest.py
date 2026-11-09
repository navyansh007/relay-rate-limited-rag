"""Fetches the three library papers at pinned versions and splits them into chunks.

Pipeline:

1. Download ``https://arxiv.org/pdf/<id>v<N>`` and verify the arXiv stamp on
   page 1 matches the pinned version.
2. Read the PDF outline (all three papers carry a hyperref TOC) to get section
   titles, nesting and start pages, and locate each heading on its page.
3. Walk the text in reading order, assigning every line to a section; drop
   running headers, page numbers and the margin arXiv stamp; tag the
   bibliography as ``References``.
4. Split each section into overlapping, sentence-aligned pieces for every
   retrieval setting in the config.

Outputs ``data/index/sections.jsonl`` and ``data/index/chunks_<piece>_<overlap>.jsonl``.

Typical usage example:

    uv run python -m relay.ingest
"""

from __future__ import annotations

import json
import re
import string
import urllib.request
from dataclasses import asdict, dataclass

import pymupdf

from relay.config import DATA_DIR, load_config

PAPERS = {
    "transformer": {
        "arxiv": "1706.03762v7",
        "title": "Attention Is All You Need",
    },
    "moshi": {
        "arxiv": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
    },
    "hnet": {
        "arxiv": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
    },
}

PAPER_DIR = DATA_DIR / "papers"
INDEX_DIR = DATA_DIR / "index"
REFERENCES = "References"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_NOISE = re.compile(r"^(\d{1,3}|arXiv:\S+.*)$")


@dataclass
class Section:
    """A contiguous section of one paper.

    Attributes:
        paper: Paper key, e.g. ``"moshi"``.
        number: Section number, e.g. ``"3.2"`` or ``"A.1"``.
        title: Section title from the PDF outline.
        page_start: First page (1-based) containing section text.
        page_end: Last page containing section text.
        text: Section body text with hyphenation repaired.
    """

    paper: str
    number: str
    title: str
    page_start: int
    page_end: int
    text: str


@dataclass
class Chunk:
    """A retrievable piece of a section.

    Attributes:
        chunk_id: Stable id, e.g. ``"moshi:3.3:p9:c2"``.
        paper: Paper key.
        arxiv: Pinned arXiv id with version.
        section: Section number.
        section_title: Section title.
        page: Page (1-based) where the piece starts.
        text: Piece text.
        n_tokens: Approximate token count (words / 0.75).
    """

    chunk_id: str
    paper: str
    arxiv: str
    section: str
    section_title: str
    page: int
    text: str
    n_tokens: int


def approx_tokens(text: str) -> int:
    """Estimates the token count of English text.

    Llama/gpt-oss tokenizers average roughly 0.75 words per token on prose;
    exact counts come back from the API in ``usage`` and are what gets reported.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    return max(1, round(len(text.split()) / 0.75))


def download(key: str) -> pymupdf.Document:
    """Downloads (if missing) and opens a pinned paper, verifying its version.

    Args:
        key: Paper key in ``PAPERS``.

    Returns:
        The opened PDF document.

    Raises:
        ValueError: If page 1 does not carry the expected arXiv version stamp.
    """
    arxiv = PAPERS[key]["arxiv"]
    path = PAPER_DIR / f"{arxiv}.pdf"
    if not path.exists():
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            f"https://arxiv.org/pdf/{arxiv}", headers={"User-Agent": "relay/0.1"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    doc = pymupdf.open(path)
    if f"arXiv:{arxiv}" not in doc[0].get_text():
        raise ValueError(f"{path.name}: version stamp arXiv:{arxiv} not found on page 1")
    return doc


def _number_toc(toc: list[list]) -> list[tuple[str, str, int]]:
    """Assigns section numbers to outline entries.

    Top-level entries after "Conclusion" are appendices and are lettered
    (A, B, ...); everything else is numbered 1, 1.1, 1.1.1 as in the papers.

    Args:
        toc: PyMuPDF outline entries ``[level, title, page]``.

    Returns:
        ``(number, title, page)`` tuples in document order.
    """
    counters = [0, 0, 0, 0]
    appendix = False
    letters = iter(string.ascii_uppercase)
    top = ""
    out = []
    for level, title, page in toc:
        level = min(level, 4)
        if level == 1:
            if appendix:
                top = next(letters)
            else:
                counters[0] += 1
                top = str(counters[0])
            counters[1:] = [0, 0, 0]
            number = top
            if title.strip().lower() == "conclusion":
                appendix = True
        else:
            counters[level - 1] += 1
            counters[level:] = [0] * (4 - level)
            number = ".".join([top] + [str(c) for c in counters[1:level]])
        out.append((number, title.strip(), page))
    return out


def _page_lines(page: pymupdf.Page) -> list[tuple[float, str]]:
    """Extracts text lines of a page in reading order, minus margin noise.

    Args:
        page: PDF page.

    Returns:
        ``(y, text)`` pairs ordered top to bottom.
    """
    width = page.rect.width
    lines = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        for line in block.get("lines", []):
            if line.get("dir", (1, 0))[1] != 0:  # rotated margin text (arXiv stamp)
                continue
            x0, y0, _, _ = line["bbox"]
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text or x0 < width * 0.04 or _NOISE.match(text):
                continue
            lines.append((y0, text))
    return lines


def _join(lines: list[str]) -> str:
    """Joins PDF lines into prose, repairing end-of-line hyphenation.

    Args:
        lines: Raw lines.

    Returns:
        Joined text with normalised whitespace.
    """
    text = ""
    for ln in lines:
        if text.endswith("-") and ln[:1].islower():
            text = text[:-1] + ln
        else:
            text = f"{text} {ln}" if text else ln
    return re.sub(r"\s+", " ", text).strip()


def _norm(text: str) -> str:
    """Normalises text for heading comparison.

    Args:
        text: Raw text.

    Returns:
        Lower-cased text with everything but letters and digits removed.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_heading(line: str, number: str, title: str) -> bool:
    """Checks whether a text line is the heading for a given section.

    Handles ``"3.2 Title"``, ``"Appendix A. Title"``, a bare ``"Title"`` (when
    the number sits on its own line) and long titles truncated by a line break.

    Args:
        line: Candidate line.
        number: Section number.
        title: Section title.

    Returns:
        True if the line is that section's heading.
    """
    ln, t = _norm(line), _norm(title)
    if not t or not ln:
        return False
    stripped = re.sub(r"^(appendix\s+)?(([a-z]|\d+)(\.\d+)*\.?\s+)?", "", line.lower().strip())
    cand = _norm(stripped)
    return cand == t or (len(cand) >= 15 and t.startswith(cand)) or ln == _norm(number) + t


def _running_headers(pages: list[list[tuple[float, str]]]) -> set[str]:
    """Finds lines that repeat on many pages (running headers / footers).

    Args:
        pages: Per-page ``(y, text)`` lines.

    Returns:
        Normalised texts that appear on at least 20% of pages (min 3).
    """
    counts: dict[str, int] = {}
    for lines in pages:
        for t in {_norm(t) for _, t in lines}:
            counts[t] = counts.get(t, 0) + 1
    threshold = max(3, len(pages) // 5)
    return {t for t, c in counts.items() if c >= threshold and len(t) > 3}


def extract_sections(key: str) -> list[Section]:
    """Splits one paper into sections using its PDF outline.

    Headings are matched against extracted lines strictly in document order,
    each search starting just after the previous heading, so a title word that
    also appears earlier on the page (e.g. "Attention") cannot be mistaken
    for the heading.

    Args:
        key: Paper key in ``PAPERS``.

    Returns:
        Sections in document order, including a leading ``abstract``
        section and a ``References`` section for the bibliography.
    """
    doc = download(key)
    heads = _number_toc(doc.get_toc())
    pages = [_page_lines(doc[i]) for i in range(len(doc))]
    headers = _running_headers(pages)
    flat = [
        (p + 1, t) for p, lines in enumerate(pages) for _, t in lines if _norm(t) not in headers
    ]

    # Position (index into flat) where each section starts.
    marks: dict[int, tuple[str, str]] = {}
    pos = 0
    for number, title, page in heads:
        # Strict pass: the section number must be on the line itself or on the
        # line just before it (figure labels often repeat a title verbatim).
        # Loose pass: title text alone.
        found = None
        for strict in (True, False):
            for i in range(pos, len(flat)):
                if flat[i][0] > page + 1:
                    break
                if flat[i][0] < page or not _is_heading(flat[i][1], number, title):
                    continue
                prev = flat[i - 1][1] if i > 0 else ""
                numbered = _norm(flat[i][1]) != _norm(title) or _norm(prev) in (
                    _norm(number),
                    _norm(f"appendix {number}"),
                )
                if numbered or not strict:
                    found = i
                    break
            if found is not None:
                break
        if found is None:  # fall back to the first line of the outline page
            print(f"  warning: {key} heading {number} {title!r} not found on p{page}")
            found = next((i for i in range(pos, len(flat)) if flat[i][0] >= page), pos)
        marks[found] = (number, title)
        pos = found + 1

    current = ("abstract", "Title and abstract")
    buckets: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for i, (page_no, text) in enumerate(flat):
        if i in marks:
            current = marks[i]
            if _is_heading(text, *current):
                continue
        if _norm(text) == "references" and len(text) < 15:
            current = (REFERENCES, REFERENCES)
            continue
        if re.fullmatch(r"(Appendix )?[A-Z]?\d*(\.\d+)*\.?", text):
            continue  # a section number detached from its title
        b = buckets.get(current)
        if b is None:
            b = buckets[current] = {"lines": [], "p0": page_no, "p1": page_no}
            order.append(current)
        b["lines"].append(text)
        b["p1"] = page_no
    return [
        Section(
            key,
            num,
            title,
            buckets[(num, title)]["p0"],
            buckets[(num, title)]["p1"],
            _join(buckets[(num, title)]["lines"]),
        )
        for num, title in order
    ]


def chunk_section(sec: Section, piece_tokens: int, overlap: int) -> list[Chunk]:
    """Splits a section into sentence-aligned pieces of roughly ``piece_tokens``.

    Pieces never cross section boundaries, so every chunk has one citable
    location. The start page of each piece is estimated by its position within
    the section's page span.

    Args:
        sec: The section to split.
        piece_tokens: Target piece size in approximate tokens.
        overlap: Approximate tokens of trailing context repeated in the next piece.

    Returns:
        The section's chunks.
    """
    sentences = []
    for sent in _SENT_SPLIT.split(sec.text):
        words = sent.split()
        # Tables and equations have no sentence punctuation; hard-split them.
        step = max(1, int(piece_tokens * 0.75))
        sentences += [" ".join(words[i : i + step]) for i in range(0, len(words), step)]
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tok = 0
    consumed = 0
    total = max(1, len(sec.text))
    span = sec.page_end - sec.page_start

    def emit() -> None:
        text = " ".join(buf)
        page = sec.page_start + round(span * consumed / total)
        cid = f"{sec.paper}:{sec.number}:p{page}:c{len(chunks)}"
        chunks.append(
            Chunk(
                cid,
                sec.paper,
                PAPERS[sec.paper]["arxiv"],
                sec.number,
                sec.title,
                page,
                text,
                approx_tokens(text),
            )
        )

    for s in sentences:
        t = approx_tokens(s)
        if buf and buf_tok + t > piece_tokens:
            emit()
            consumed += sum(len(x) + 1 for x in buf)
            keep, kept = [], 0
            for x in reversed(buf):
                kept += approx_tokens(x)
                if kept > overlap:
                    break
                keep.insert(0, x)
            consumed -= sum(len(x) + 1 for x in keep)
            buf, buf_tok = keep, sum(approx_tokens(x) for x in keep)
        buf.append(s)
        buf_tok += t
    if buf:
        emit()
    return chunks


def build() -> None:
    """Extracts all papers and writes sections and per-setting chunk files."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    sections = [s for key in PAPERS for s in extract_sections(key)]
    with open(INDEX_DIR / "sections.jsonl", "w") as f:
        f.writelines(json.dumps(asdict(s)) + "\n" for s in sections)
    indexed = [s for s in sections if s.number != REFERENCES]
    cfg = load_config()
    shapes = {(v["piece_tokens"], v["overlap"]) for v in cfg["retrieval"]["settings"].values()}
    for piece, overlap in sorted(shapes):
        chunks = [c for s in indexed for c in chunk_section(s, piece, overlap)]
        with open(INDEX_DIR / f"chunks_{piece}_{overlap}.jsonl", "w") as f:
            f.writelines(json.dumps(asdict(c)) + "\n" for c in chunks)
        print(f"piece={piece} overlap={overlap}: {len(chunks)} chunks")
    for key in PAPERS:
        n = sum(1 for s in sections if s.paper == key)
        toks = sum(approx_tokens(s.text) for s in indexed if s.paper == key)
        print(f"{key}: {n} sections, ~{toks} indexed tokens")


if __name__ == "__main__":
    build()
