import pandas as pd
import pytest
from rhagent.features import entry_features


_KEYS = {
    "vol20", "gap", "trend5", "dow", "dist_high20", "dist_low20", "ret1",
    "ret5", "ret20", "zscore20", "rsi14", "vol_ratio",
}


def _bars(closes, opens=None):
    opens = opens or closes
    return pd.DataFrame({"open": opens, "close": closes})


def test_entry_features_shape_and_lookahead_free():
    bars = _bars([10, 11, 12, 13, 14, 15, 16])
    f = entry_features(bars)
    assert set(f) == _KEYS
    # trend5 = sign(close[-1] - close[-6]) = sign(16 - 11) = +1
    assert f["trend5"] == 1.0
    # dropping the last bar changes the features (only past data used, no peeking ahead)
    assert entry_features(bars.iloc[:-1])["trend5"] == 1.0  # sign(15-10)
    # dist_high20/dist_low20/ret1 over the 7-bar run (all-time high is the last close)
    assert f["dist_high20"] == 0.0
    assert f["dist_low20"] == pytest.approx(16 / 10 - 1.0)
    assert f["ret1"] == pytest.approx(16 / 15 - 1.0)
    # no DatetimeIndex on this fixture -> dow defaults to 0.0
    assert f["dow"] == 0.0


def test_entry_features_dow_from_datetime_index():
    bars = _bars([10, 11])
    bars.index = pd.to_datetime(["2026-07-13", "2026-07-14"])  # Mon, Tue
    assert entry_features(bars)["dow"] == 1.0


def test_entry_features_short_history_defaults_zero():
    f = entry_features(_bars([10]))
    assert f == {
        "vol20": 0.0, "gap": 0.0, "trend5": 0.0,
        "dow": 0.0, "dist_high20": 0.0, "dist_low20": 0.0, "ret1": 0.0,
        "ret5": 0.0, "ret20": 0.0, "zscore20": 0.0,
        "rsi14": 50.0, "vol_ratio": 1.0,  # neutral defaults on short history
    }


def test_entry_features_new_factors_values():
    # Strictly rising series: RSI saturates at 100 (no down moves), momentum
    # positive, z-score of the last (highest) close is the max of its window.
    up = entry_features(_bars(list(range(1, 30))))
    assert up["rsi14"] == 100.0
    assert up["ret5"] > 0 and up["ret20"] > 0
    assert up["zscore20"] > 0

    # Strictly falling series: RSI floors at 0, momentum negative.
    down = entry_features(_bars(list(range(30, 1, -1))))
    assert down["rsi14"] == 0.0
    assert down["ret5"] < 0 and down["ret20"] < 0

    # vol_ratio: a volume spike on the last bar reads > 1.
    bars = _bars([10, 11, 12, 13, 14])
    bars["volume"] = [100, 100, 100, 100, 500]
    assert entry_features(bars)["vol_ratio"] > 1.0


def test_entry_features_lookahead_free_on_new_factors():
    # Dropping the final bar must change momentum/RSI — proves no forward peeking.
    bars = _bars(list(range(1, 30)))
    full = entry_features(bars)
    trimmed = entry_features(bars.iloc[:-1])
    assert full["ret5"] != trimmed["ret5"]
