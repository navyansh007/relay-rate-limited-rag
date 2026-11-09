from relay.analysis import pct, peak_in_system, scenario_rows
from relay.grade import propose_grade

Q_ANS = {"answerable": True, "must_contain": ["41.8"], "location": "transformer abstract p.1"}
Q_UNANS = {"answerable": False, "must_contain": [], "location": "none"}


def test_propose_grade_rules():
    good = {
        "answer_kind": "answer",
        "text": "It is 41.8 BLEU.",
        "citations": ["transformer abstract p.1"],
    }
    assert propose_grade(Q_ANS, good) == "correct"
    assert propose_grade(Q_ANS, {**good, "text": "It is 41.0 BLEU."}) == "wrong"
    assert propose_grade(Q_ANS, {**good, "citations": ["hnet §3 p.12"]}) == "wrong"
    assert propose_grade(Q_ANS, {**good, "answer_kind": "refusal"}) == "refused_wrongly"
    assert propose_grade(Q_UNANS, {**good, "answer_kind": "refusal"}) == "refused_correctly"
    assert propose_grade(Q_UNANS, good) == "wrong"


def row(start, total, status="ok", ft="100", tin=100, tout=50, retries=0):
    return {
        "request_id": "x",
        "scenario": "s",
        "source": "stand-in",
        "start_time": str(start),
        "first_token_ms": ft,
        "total_ms": str(total),
        "status": status,
        "retries": str(retries),
        "tokens_in": str(tin),
        "tokens_out": str(tout),
        "retrieval_setting": "s3",
        "answer_grade": "",
    }


def test_peak_and_percentiles():
    rows = [row(0, 2000), row(1, 2000), row(1.5, 100), row(5, 100)]
    assert peak_in_system(rows) == 3
    assert pct([5, 1, 3, 2, 4], 50) == 3 and pct([], 50) is None


def test_scenario_rows_counts():
    rows = [
        row(0, 1000),
        row(30, 1000, status="timeout", ft=""),
        row(60, 1000, status="rate_limited", retries=2),
    ]
    a, b = scenario_rows(rows, (1.0, 2.0))
    assert "| s | stand-in | 3 |" in a[0] and "(n=1)" in a[0]
    # 1/3 ok, 1/3 rl, 1/3 timeout; retries 2/3; cost = 1000*(100*1+50*2)/1e6 = 0.20
    assert "| 33.3 | 33.3 | 33.3 | 0.67 |" in b[0] and "$0.20" in b[0]
