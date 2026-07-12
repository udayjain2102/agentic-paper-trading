from rhagent.search.loop import Gates, apply_gates, config_key, first_failing_gate
from rhagent.search.score import ConfigScore


def _cs(params, icir=0.4, hl=10, signs=(1, 1, 1)):
    return ConfigScore("momentum", params, icir, hl, signs, 100)


def test_config_key_is_order_independent_hashable():
    assert config_key({"a": 1, "b": 2}) == config_key({"b": 2, "a": 1})


def test_icir_floor_gate():
    grids = {"lookback": [40]}
    low = _cs({"lookback": 40}, icir=0.1)
    assert first_failing_gate(low, {config_key(low.params): low}, grids,
                              Gates(use_robustness=False, use_sign=False,
                                    use_half_life=False)) == "icir_floor"


def test_half_life_gate():
    g = Gates(use_robustness=False, use_sign=False, use_icir=False)
    grids = {"lookback": [40]}
    none_hl = _cs({"lookback": 40}, hl=None)
    short_hl = _cs({"lookback": 40}, hl=3)
    ok_int = _cs({"lookback": 40}, hl=20)
    ok_str = _cs({"lookback": 40}, hl=">50")
    assert first_failing_gate(none_hl, {}, grids, g) == "half_life_floor"
    assert first_failing_gate(short_hl, {}, grids, g) == "half_life_floor"
    assert first_failing_gate(ok_int, {}, grids, g) is None
    assert first_failing_gate(ok_str, {}, grids, g) is None


def test_sign_stability_gate():
    g = Gates(use_robustness=False, use_icir=False, use_half_life=False)
    flip = _cs({"lookback": 40}, signs=(1, -1, 1))
    zero = _cs({"lookback": 40}, signs=(1, 0, 1))
    good = _cs({"lookback": 40}, signs=(1, 1, 1))
    assert first_failing_gate(flip, {}, {"lookback": [40]}, g) == "sign_stability"
    assert first_failing_gate(zero, {}, {"lookback": [40]}, g) == "sign_stability"
    assert first_failing_gate(good, {}, {"lookback": [40]}, g) is None


def test_robustness_gate():
    g = Gates(use_icir=False, use_half_life=False, use_sign=False)  # robustness on
    grids = {"lookback": [20, 40, 60]}
    mid = _cs({"lookback": 40}, icir=0.5)
    lo = _cs({"lookback": 20}, icir=0.5)
    hi = _cs({"lookback": 60}, icir=0.1)   # weak neighbor
    by_key = {config_key(c.params): c for c in (mid, lo, hi)}
    # 40's neighbors are 20 (0.5, ok) and 60 (0.1, below floor) -> fails robustness
    assert first_failing_gate(mid, by_key, grids, g) == "robustness"
    # make both neighbors strong
    hi2 = _cs({"lookback": 60}, icir=0.5)
    by_key2 = {config_key(c.params): c for c in (mid, lo, hi2)}
    assert first_failing_gate(mid, by_key2, grids, g) is None


def test_robustness_no_scored_neighbors_fails():
    g = Gates(use_icir=False, use_half_life=False, use_sign=False)
    grids = {"lookback": [20, 40, 60]}
    lonely = _cs({"lookback": 40}, icir=0.9)
    assert first_failing_gate(lonely, {config_key(lonely.params): lonely}, grids, g) == "robustness"


def test_apply_gates_splits_and_ranks():
    grids = {"lookback": [20, 40, 60]}
    a = _cs({"lookback": 20}, icir=0.6)
    b = _cs({"lookback": 40}, icir=0.8)
    c = _cs({"lookback": 60}, icir=0.1)  # below floor
    survivors, rejected = apply_gates([a, b, c], grids,
                                      Gates(use_robustness=False, use_sign=False,
                                            use_half_life=False))
    assert [s.params["lookback"] for s in survivors] == [40, 20]  # ranked by icir desc
    assert ({"lookback": 60}, "icir_floor") in rejected
