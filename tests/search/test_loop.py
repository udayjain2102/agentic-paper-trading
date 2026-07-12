from rhagent.search.loop import Gates, RoundLog, SearchResult, run_search
from rhagent.search.score import ConfigScore


def _fake_scorer(good_lookbacks):
    def scorer(params):
        icir = 0.5 if params["lookback"] in good_lookbacks else 0.1
        return ConfigScore("momentum", dict(params), icir, 10, (1, 1, 1), 100)
    return scorer


def test_round0_scores_coarse_grid():
    res = run_search("momentum", None, None, max_rounds=1,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer({40, 60}))
    assert isinstance(res, SearchResult)
    # momentum coarse grid has 5 configs
    assert res.rounds[0].n_scored == 5
    assert res.n_tested == 5
    assert {s.params["lookback"] for s in res.rounds[0].survivors} == {40, 60}


def test_refined_round_scores_new_configs_and_counts_distinct():
    res = run_search("momentum", None, None, max_rounds=2,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer({40, 60}))
    assert len(res.rounds) == 2
    # round1 refines around {40,60}: midpoints 30,50,75 (20/40 -> 30, 40/60 ->50, 60/90 ->75)
    round1_tested = {tuple(sorted(p.items())) for p in
                     [s.params for s in res.rounds[1].survivors]}
    assert res.n_tested == 8  # 5 coarse + 3 new refined
    # those refined configs are not "good" -> round1 has no survivors
    assert res.rounds[1].survivors == []


def test_best_is_top_icir_survivor():
    res = run_search("momentum", None, None, max_rounds=2,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer({40, 60}))
    assert res.best is not None
    assert res.best.icir == 0.5
    assert res.best.params["lookback"] in {40, 60}


def test_stops_when_no_survivors():
    res = run_search("momentum", None, None, max_rounds=4,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer(set()))
    # nothing passes -> round 0 has no survivors -> loop stops after round 0
    assert len(res.rounds) == 1
    assert res.best is None
    assert res.survivors == []


def test_rejected_records_failing_gate():
    res = run_search("momentum", None, None, max_rounds=1,
                     gates=Gates(use_robustness=False), scorer=_fake_scorer({40}))
    gates_hit = {gate for _, gate in res.rounds[0].rejected}
    assert "icir_floor" in gates_hit


def test_refinement_survivors_with_robustness_on():
    # Every config scores well. With robustness ON, a refined-round midpoint must
    # be able to survive by looking up its neighbors, which were scored in a prior
    # round (the coarse survivor values). Before the cumulative-map fix, those
    # neighbors were absent from the round's score set and every midpoint failed
    # robustness, silently disabling refinement for single-parameter strategies.
    def scorer(params):
        return ConfigScore("momentum", dict(params), 0.5, 10, (1, 1, 1), 100)
    res = run_search("momentum", None, None, max_rounds=2, gates=Gates(), scorer=scorer)
    assert len(res.rounds) == 2
    assert len(res.rounds[1].survivors) > 0          # refined midpoints survived
    coarse = {20, 40, 60, 90, 120}
    assert any(s.params["lookback"] not in coarse for s in res.survivors)  # explored new values
