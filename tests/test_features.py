import pandas as pd
import pytest
from rhagent.features import entry_features


def _bars(closes, opens=None):
    opens = opens or closes
    return pd.DataFrame({"open": opens, "close": closes})


def test_entry_features_shape_and_lookahead_free():
    bars = _bars([10, 11, 12, 13, 14, 15, 16])
    f = entry_features(bars)
    assert set(f) == {"vol20", "gap", "trend5", "dow", "dist_high20", "dist_low20", "ret1"}
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
    }
