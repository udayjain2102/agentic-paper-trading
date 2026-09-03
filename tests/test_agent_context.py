"""Market-context block for the agent_ctx record: pure block math, prompt
insertion, and the regression pin that market_context=False changes nothing."""

from __future__ import annotations

import json

import pandas as pd

from rhagent.engine import AgentEngine, market_block, render_context


def _hist(closes):
    closes = list(closes)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


def _universe():
    # SPY up 1% on the last bar and ~+2% over 5; AAA strongest, CCC weakest.
    return {
        "SPY": _hist([100, 100.5, 101, 101.5, 101.0, 102.0]),
        "AAA": _hist([10, 10, 10, 10, 10, 12]),      # +20% 5d
        "BBB": _hist([10, 10, 10, 10, 10, 10.5]),    # +5% 5d
        "CCC": _hist([10, 10, 10, 10, 10, 9]),       # -10% 5d
    }


def test_market_block_values():
    b = market_block(_universe())
    assert abs(b["spy_1d"] - (102.0 / 101.0 - 1)) < 1e-12
    assert abs(b["spy_5d"] - (102.0 / 100.0 - 1)) < 1e-12
    assert b["n"] == 4
    # SPY, AAA, BBB positive; CCC negative -> 3/4
    assert abs(b["breadth"] - 0.75) < 1e-12
    assert b["rank"]["AAA"] == 1 and b["rank"]["CCC"] == 4


def test_market_block_without_spy_is_na_not_error():
    u = _universe()
    del u["SPY"]
    b = market_block(u)
    assert b["spy_1d"] is None and b["spy_5d"] is None
    assert b["n"] == 3
    txt = render_context(b, "AAA")
    assert "SPY_1d=n/a SPY_5d=n/a" in txt
    assert "rank 1/3" in txt


def test_render_context_shape():
    txt = render_context(market_block(_universe()), "BBB")
    lines = txt.split("\n")
    assert lines[0].startswith("Market today: SPY_1d=+0.99% SPY_5d=+2.00% breadth_5d=0.75 ")
    assert lines[0].endswith("(share of 4 names with positive 5d momentum)")
    assert lines[1] == "This name: momentum_5d rank 2/4 (1 = strongest)"
    assert txt.endswith("\n")


def test_market_context_false_prompt_is_unchanged():
    """Regression pin: the control record's prompt must not move."""
    h = _hist([10, 11, 12, 13, 14, 15, 16])
    eng = AgentEngine(complete=lambda p: "{}")
    before = eng._prompt("NVDA", h, 0.0)
    after = AgentEngine(complete=lambda p: "{}", market_context=False)._prompt("NVDA", h, 0.0)
    assert before == after
    assert "Market today" not in before
    # and the exact text a production prompt has today
    assert before.startswith("You are a trading agent deciding today's position in NVDA.\n"
                             "NVDA: last_close=16.00 momentum_5d=+0.4545 vol_20d=")
    assert "Respond with ONLY this JSON object" in before


def test_decide_all_passes_block_into_every_prompt():
    u = _universe()
    seen = {}

    def fake(prompt):
        sym = prompt.split("position in ")[1].split(".")[0]
        seen[sym] = prompt
        return json.dumps({"target": 1, "reason": "up"})

    eng = AgentEngine(complete=fake, market_context=True)
    # decide only two names, but give the whole universe as context
    out = eng.decide_all(["AAA", "CCC"], {s: u[s] for s in ("AAA", "CCC")},
                         {"AAA": 0.0, "CCC": 0.0}, context_histories=u)
    assert set(seen) == {"AAA", "CCC"}
    for sym, p in seen.items():
        assert "Market today: SPY_1d=+0.99% SPY_5d=+2.00% breadth_5d=0.75" in p
        assert "Respond with ONLY this JSON object" in p
        # block sits after the feature line and before the JSON instruction
        assert p.index(f"{sym}: last_close=") < p.index("Market today") < p.index("Respond with ONLY")
    assert "rank 1/4" in seen["AAA"] and "rank 4/4" in seen["CCC"]
    assert out["AAA"].status == "ok" and out["CCC"].status == "ok"


def test_decide_all_without_context_histories_uses_histories():
    u = _universe()
    seen = []
    eng = AgentEngine(complete=lambda p: (seen.append(p), '{"target": 0, "reason": "x"}')[1],
                      market_context=True)
    eng.decide_all(list(u), u, {s: 0.0 for s in u})
    assert all("breadth_5d=0.75" in p for p in seen)


def test_market_context_false_decide_all_sends_no_block():
    u = _universe()
    seen = []
    eng = AgentEngine(complete=lambda p: (seen.append(p), '{"target": 0, "reason": "x"}')[1])
    eng.decide_all(list(u), u, {s: 0.0 for s in u}, context_histories=u)
    assert seen and not any("Market today" in p for p in seen)
