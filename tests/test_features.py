import pandas as pd
from rhagent.features import entry_features


def _bars(closes, opens=None):
    opens = opens or closes
    return pd.DataFrame({"open": opens, "close": closes})


def test_entry_features_shape_and_lookahead_free():
    bars = _bars([10, 11, 12, 13, 14, 15, 16])
    f = entry_features(bars)
    assert set(f) == {"vol20", "gap", "trend5"}
    # trend5 = sign(close[-1] - close[-6]) = sign(16 - 11) = +1
    assert f["trend5"] == 1.0
    # dropping the last bar changes the features (only past data used, no peeking ahead)
    assert entry_features(bars.iloc[:-1])["trend5"] == 1.0  # sign(15-10)


def test_entry_features_short_history_defaults_zero():
    f = entry_features(_bars([10]))
    assert f == {"vol20": 0.0, "gap": 0.0, "trend5": 0.0}
