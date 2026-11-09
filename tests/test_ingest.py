from relay.ingest import Section, _is_heading, _number_toc, approx_tokens, chunk_section


def test_number_toc_letters_appendix_after_conclusion():
    toc = [
        [1, "Intro", 1],
        [2, "Sub", 1],
        [3, "SubSub", 2],
        [1, "Conclusion", 3],
        [1, "Extra", 4],
        [2, "More", 4],
    ]
    nums = [n for n, _, _ in _number_toc(toc)]
    assert nums == ["1", "1.1", "1.1.1", "2", "A", "A.1"]


def test_is_heading_variants():
    assert _is_heading("Introduction", "1", "Introduction")
    assert _is_heading("3.2.1 Architecture", "3.2.1", "Architecture")
    assert _is_heading(
        "Appendix A. Additional ablation on Mimi codec", "A", "Additional ablation on Mimi codec"
    )
    assert not _is_heading("Recurrent neural networks, long short-term", "2", "Background")


def test_chunk_section_respects_size_and_ids():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    sec = Section("transformer", "3.2", "Attention", 3, 4, text)
    chunks = chunk_section(sec, piece_tokens=100, overlap=20)
    assert len(chunks) > 5
    assert all(c.n_tokens <= 130 for c in chunks)
    assert chunks[0].chunk_id == "transformer:3.2:p3:c0"
    assert chunks[-1].page == 4


def test_chunk_section_splits_unpunctuated_tables():
    sec = Section("hnet", "3.1", "LM", 12, 12, " ".join(["1.23"] * 1000))
    chunks = chunk_section(sec, piece_tokens=200, overlap=0)
    assert max(c.n_tokens for c in chunks) <= approx_tokens(" ".join(["x"] * 150)) + 1
