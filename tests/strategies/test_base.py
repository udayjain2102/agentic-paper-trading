import pandas as pd

from rhagent.strategies.base import Strategy, clamp_short


def test_clamp_short_zeros_negatives_when_long_only():
    pos = pd.Series([-1, 0, 1])
    out = clamp_short(pos, allow_short=False)
    assert list(out) == [0, 0, 1]


def test_clamp_short_keeps_negatives_when_allowed():
    pos = pd.Series([-1, 0, 1])
    out = clamp_short(pos, allow_short=True)
    assert list(out) == [-1, 0, 1]


def test_base_strategy_positions_not_implemented():
    s = Strategy()
    try:
        s.positions(pd.DataFrame({"close": [1.0]}))
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
