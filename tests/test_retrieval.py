from relay.retrieval import tokenize


def test_tokenize_keeps_numbers_and_hyphenated_terms():
    toks = tokenize("The H-Net reaches 41.8 BLEU, and the Mimi codec runs at 12.5 Hz.")
    assert "h-net" in toks and "41.8" in toks and "12.5" in toks and "mimi" in toks
    assert "the" not in toks
