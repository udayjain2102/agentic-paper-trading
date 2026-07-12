import numpy as np
import pandas as pd

from rhagent.search.score import ConfigScore, score_config


def _panel(n_syms=12, n_bars=60, ascending=True, seed=3):
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="D", name="date")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.002, size=(n_bars, n_syms))
    bars, closes = {}, {}
    for i in range(n_syms):
        k = i if ascending else (n_syms - 1 - i)
        rate = 1 + k / 200
        c = [100.0 * (rate ** t) * (1 + noise[t][i]) for t in range(n_bars)]
        idxc = idx
        bars[f"S{i}"] = pd.DataFrame(
            {"open": c, "high": c, "low": c, "close": c, "volume": [1e6] * n_bars},
            index=idxc,
        )
        closes[f"S{i}"] = c
    return bars, pd.DataFrame(closes, index=idx)


def test_score_config_positive_when_signal_predicts():
    bars, close = _panel(ascending=True)
    sc = score_config("momentum", {"lookback": 40}, bars, close, horizon=1, min_names=5)
    assert isinstance(sc, ConfigScore)
    assert sc.strategy == "momentum" and sc.params == {"lookback": 40}
    assert sc.icir > 0
    assert sc.subperiod_ic_signs == (1, 1, 1)
    assert sc.n_obs > 0


def test_score_config_negative_when_signal_anti_predicts():
    # On a monotonic uptrend, mean_reversion's signal (-z) is most negative for
    # the strongest trenders, which keep rising (highest forward return), so the
    # signal genuinely anti-predicts -> negative IC each day.
    bars, close = _panel(ascending=True)
    sc = score_config("mean_reversion", {"lookback": 20}, bars, close,
                      horizon=1, min_names=5)
    assert sc.icir < 0
    assert all(s == -1 for s in sc.subperiod_ic_signs)


def test_score_config_subperiod_count():
    bars, close = _panel()
    sc = score_config("momentum", {"lookback": 40}, bars, close, horizon=1,
                      min_names=5, n_subperiods=4)
    assert len(sc.subperiod_ic_signs) == 4
