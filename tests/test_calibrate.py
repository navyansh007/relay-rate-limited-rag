import math
import random

from relay.calibrate import _pct, fit_model


def test_fit_recovers_known_parameters():
    rng = random.Random(0)
    calls = []
    for _ in range(400):
        tin = rng.randint(50, 3000)
        ttft = 200 * math.exp(rng.gauss(0, 0.4)) + 0.05 * tin  # 50 ms per 1k tokens
        tout = rng.randint(20, 300)
        calls.append(
            {
                "ttft_ms": ttft,
                "tokens_in": tin,
                "tokens_out": tout,
                "total_ms": ttft + tout / 500 * 1000,
            }
        )
    fit = fit_model(calls)
    assert 170 < fit["ttft_median_ms"] < 230
    assert 0.3 < fit["ttft_sigma"] < 0.5
    assert 30 < fit["prefill_ms_per_1k"] < 70
    assert 480 < fit["decode_tps_median"] < 520


def test_nearest_rank_percentile():
    xs = list(range(1, 101))
    assert _pct(xs, 50) == 50 and _pct(xs, 95) == 95 and _pct(xs, 99) == 99 and _pct(xs, 100) == 100
