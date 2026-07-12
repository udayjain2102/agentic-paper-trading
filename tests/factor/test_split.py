import pandas as pd

from rhagent.factor.split import in_sample_mask, oos_cutoff


def _idx(n):
    return pd.date_range("2026-01-01", periods=n, freq="D", name="date")


def test_oos_cutoff_at_quantile():
    idx = _idx(100)
    cut = oos_cutoff(idx, oos_frac=0.25)
    # 75% in-sample -> cutoff is the 76th day (index 75)
    assert cut == idx[75]


def test_in_sample_mask_excludes_forward_window_crossing_boundary():
    idx = _idx(100)
    cut = oos_cutoff(idx, 0.25)  # idx[75]
    mask = in_sample_mask(idx, cut, horizon=5)
    # day at index 70: 70+5=75 -> idx[75] == cutoff, NOT < cutoff -> excluded
    assert mask.iloc[70] == False
    # day at index 69: 69+5=74 -> idx[74] < cutoff -> included
    assert mask.iloc[69] == True


def test_in_sample_mask_all_oos_days_false():
    idx = _idx(100)
    cut = oos_cutoff(idx, 0.25)
    mask = in_sample_mask(idx, cut, horizon=5)
    assert not mask[idx >= cut].any()


def test_in_sample_mask_horizon_one():
    idx = _idx(10)
    cut = idx[8]  # last day is OOS
    mask = in_sample_mask(idx, cut, horizon=1)
    # index 7: 7+1=8 == cutoff -> excluded; index 6: 6+1=7 < cutoff -> included
    assert mask.iloc[6] == True and mask.iloc[7] == False
