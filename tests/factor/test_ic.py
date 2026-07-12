import numpy as np
import pandas as pd

from rhagent.factor.ic import (
    forward_returns, ic_decay, ic_series, icir, half_life, rank_ic_one,
)


def _panel(rows: dict, cols: int):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", name="date")
    return pd.DataFrame(list(rows.values()), index=idx,
                        columns=[f"S{i}" for i in range(cols)])


def test_forward_returns():
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]})
    fwd = forward_returns(close, 1)
    assert abs(fwd["A"].iloc[0] - 0.1) < 1e-12
    assert pd.isna(fwd["A"].iloc[-1])  # last row has no forward value


def test_rank_ic_perfect_and_reversed():
    n = 12
    sig = pd.Series(range(n), index=[f"S{i}" for i in range(n)], dtype=float)
    ret = pd.Series(range(n), index=[f"S{i}" for i in range(n)], dtype=float)
    assert abs(rank_ic_one(sig, ret) - 1.0) < 1e-9
    assert abs(rank_ic_one(sig, ret[::-1].reset_index(drop=True)
                           .set_axis(sig.index)) - (-1.0)) < 1e-9


def test_rank_ic_too_few_names_is_nan():
    sig = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    ret = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    assert np.isnan(rank_ic_one(sig, ret, min_names=10))


def test_rank_ic_invariant_to_common_return_shift():
    # market-relative property: adding a constant to every name's return
    # that day does not change rank-IC (rank is shift-invariant)
    n = 12
    idx = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(1)
    sig = pd.Series(rng.normal(size=n), index=idx)
    ret = pd.Series(rng.normal(size=n), index=idx)
    base = rank_ic_one(sig, ret)
    shifted = rank_ic_one(sig, ret + 0.05)
    assert abs(base - shifted) < 1e-12


def test_ic_series_and_icir():
    # 3 days, 12 names; day-by-day signal perfectly ranks next-day return
    n = 12
    cols = [f"S{i}" for i in range(n)]
    idx = pd.date_range("2026-01-01", periods=4, freq="D", name="date")
    # close chosen so 1-day forward return ordering matches the signal ordering
    sig = pd.DataFrame([list(range(n))] * 4, index=idx, columns=cols, dtype=float)
    # each name grows at a distinct rate -> forward-return rank == name index.
    # A tiny seeded perturbation is added so the daily IC isn't a bit-for-bit
    # constant 1.0 (which would make the series zero-variance and icir() == 0.0
    # by definition); the perturbation is small enough that ranks stay intact
    # on all but rare adjacent ties, keeping IC > 0.99 every day.
    rates = [1.0 + i / 100 for i in range(n)]
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=0.003, size=(4, n))
    close = pd.DataFrame(
        [[(r ** t) * (1 + noise[t][i]) for i, r in enumerate(rates)] for t in range(4)],
        index=idx, columns=cols,
    )
    ic = ic_series(sig, close, h=1, min_names=10)
    assert len(ic) >= 2
    assert (ic > 0.99).all()
    assert icir(ic) > 0  # positive and finite


def test_icir_empty_and_zero_variance():
    assert icir(pd.Series(dtype=float)) == 0.0
    assert icir(pd.Series([0.3, 0.3, 0.3])) == 0.0


def test_ic_decay_and_half_life():
    decay = {1: 0.10, 5: 0.08, 10: 0.04, 20: 0.02, 50: 0.01}
    assert half_life(decay) == 10  # first horizon where |IC| <= 0.05
    assert half_life({1: 0.10, 5: 0.09, 10: 0.09, 20: 0.09, 50: 0.09}) == ">50"
    assert half_life({1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}) is None
