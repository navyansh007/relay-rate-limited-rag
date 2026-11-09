from relay.llm_client import RL_DAY, RL_MINUTE, classify_429, parse_duration_s


def test_parse_duration():
    assert parse_duration_s("7.66s") == 7.66
    assert abs(parse_duration_s("2m59.56s") - 179.56) < 1e-9
    assert parse_duration_s("1h2m3s") == 3723
    assert abs(parse_duration_s("120ms") - 0.12) < 1e-9
    assert parse_duration_s("3") == 3.0
    assert parse_duration_s(None) is None


def test_classify_429():
    tpm = "Rate limit reached for model `x` on tokens per minute (TPM): Limit 6000"
    rpd = "Rate limit reached for model `x` on requests per day (RPD): Limit 1000"
    assert classify_429(tpm) == (RL_MINUTE, "tokens/minute")
    assert classify_429(rpd) == (RL_DAY, "requests/day")
    assert classify_429("something else") == (RL_MINUTE, None)
