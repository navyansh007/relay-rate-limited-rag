from relay.verify import SentenceSplitter, check_sentence, render_citations

P = ["The big model achieves 41.8 BLEU on EN-FR after 3.5 days.", "We use h = 8 heads."]


def test_number_must_appear_in_cited_passage():
    assert check_sentence("It reaches 41.8 BLEU [1].", P).ok
    bad = check_sentence("It reaches 41.0 BLEU [1].", P)
    assert not bad.ok and bad.reason == "number_not_in_cited"
    assert not check_sentence("It uses 8 heads [1].", P).ok  # 8 is in passage 2, not 1
    assert check_sentence("It uses 8 heads [2].", P).ok


def test_citation_rules():
    assert check_sentence("No citation here.", P).reason == "no_citation"
    assert check_sentence("Bad ref [3].", P).reason == "bad_citation"
    assert check_sentence("The papers do not say how long inference takes.", P).ok
    assert check_sentence("See Table 2 and Section 6.1 [1].", P).ok  # refs aren't claims


def test_splitter_keeps_trailing_citations():
    sp = SentenceSplitter()
    out = sp.feed("It reaches 41.8 BLEU. [1] It uses 8 heads [2]. Tra")
    assert out == ["It reaches 41.8 BLEU. [1]", "It uses 8 heads [2]."]
    assert sp.feed("ining took 3.5 days [1].") == []
    assert sp.flush() == ["Training took 3.5 days [1]."]


def test_render_citations():
    assert render_citations("X [1][2].", ["a §1 p.1", "b §2 p.2"]) == "X [a §1 p.1][b §2 p.2]."
