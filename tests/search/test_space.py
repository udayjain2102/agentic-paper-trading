from rhagent.search.space import (
    SEARCH_SPACES, coarse_grids, configs_from_grids, neighbors, refine_grids,
)
import pytest


def test_coarse_grids_are_copies():
    g = coarse_grids("mean_reversion")
    g["lookback"].append(999)
    assert 999 not in SEARCH_SPACES["mean_reversion"]["lookback"]


def test_coarse_grids_unknown_strategy_raises():
    with pytest.raises(KeyError):
        coarse_grids("nope")


def test_configs_from_grids_cartesian_deterministic():
    cfgs = configs_from_grids({"lookback": [10, 20], "entry": [1.0, 2.0]})
    # keys iterated in sorted order ["entry","lookback"]; product varies the
    # last key (lookback) fastest
    assert cfgs == [
        {"entry": 1.0, "lookback": 10},
        {"entry": 1.0, "lookback": 20},
        {"entry": 2.0, "lookback": 10},
        {"entry": 2.0, "lookback": 20},
    ]


def test_neighbors_interior_and_edge():
    grids = {"lookback": [10, 20, 40, 60]}
    # interior value 20 -> neighbors 10 and 40
    nb = neighbors({"lookback": 20}, grids)
    assert {"lookback": 10} in nb and {"lookback": 40} in nb and len(nb) == 2
    # edge value 10 -> only neighbor 20
    assert neighbors({"lookback": 10}, grids) == [{"lookback": 20}]


def test_neighbors_multi_param():
    grids = {"a": [1, 2, 3], "b": [10, 20]}
    nb = neighbors({"a": 2, "b": 10}, grids)
    assert {"a": 1, "b": 10} in nb
    assert {"a": 3, "b": 10} in nb
    assert {"a": 2, "b": 20} in nb
    assert len(nb) == 3


def test_refine_grids_inserts_midpoints_int():
    refined = refine_grids({"lookback": [10, 20, 40, 60]}, [{"lookback": 20}])
    # around 20: midpoints to 10 (->15) and 40 (->30)
    assert refined["lookback"] == [15, 20, 30]


def test_refine_grids_float_param():
    refined = refine_grids({"entry": [0.5, 1.0, 1.5, 2.0]}, [{"entry": 1.0}])
    assert refined["entry"] == [0.75, 1.0, 1.25]


def test_refine_grids_caps_values():
    # many survivors across a wide grid -> capped to max_values nearest the survivor mean
    grids = {"x": [0, 10, 20, 30, 40, 50, 60]}
    survivors = [{"x": v} for v in [10, 20, 30, 40, 50]]
    refined = refine_grids(grids, survivors, max_values=4)
    assert len(refined["x"]) == 4
