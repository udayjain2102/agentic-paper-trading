"""--market-context must reach the AgentEngine a production tick constructs,
the per-bar decide_all call must carry the full universe as context, run.json
must say which kind of record this is, and the ctx record must not reflect."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rhagent import forward


def _bars(symbols, n=30):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    out = {}
    for i, s in enumerate(symbols):
        close = pd.Series(np.linspace(100, 100 + i, n), index=idx)
        out[s] = pd.DataFrame({"open": close, "high": close, "low": close,
                               "close": close, "volume": 1e6}, index=idx)
    return out


def _cfg(universe):
    return SimpleNamespace(
        strategy=SimpleNamespace(name="agent", params={}, universe=list(universe),
                                 overlay="none", cost_bps=7.0, fill_mode="next_open"),
        agent=SimpleNamespace(model="", max_tokens=None, allow_short=True,
                              use_lessons=False),
    )


def test_build_agent_passes_flag():
    cfg = _cfg(["AAA"])
    assert forward._build_agent(cfg).market_context is False
    assert forward._build_agent(cfg, market_context=True).market_context is True
    assert forward._build_agent(cfg, market_context=True).allow_short is True


def test_agent_positions_gives_decide_all_the_full_universe(monkeypatch, tmp_path):
    bars = _bars(["AAA", "BBB", "SPY"])
    calls = []

    class Fake:
        market_context = True

        def decide_all(self, symbols, histories, current_pos, context_histories=None):
            calls.append((tuple(symbols), None if context_histories is None
                          else tuple(sorted(context_histories))))
            from rhagent.engine import Decision
            return {s: Decision(target=0.0, reason="x", status="ok") for s in symbols}

    forward._agent_positions(tmp_path, bars, Fake())
    assert calls, "decide_all never called"
    for symbols, ctx in calls:
        assert ctx == ("AAA", "BBB", "SPY")   # every bar sees the whole universe


def test_positions_uses_market_context_flag(monkeypatch, tmp_path):
    bars = _bars(["AAA", "SPY"])
    seen = []

    def fake_complete(prompt):
        seen.append(prompt)
        return '{"target": 1, "reason": "long"}'

    monkeypatch.setattr("rhagent.engine.AgentEngine._default_complete",
                        lambda self: fake_complete)
    cfg = _cfg(["AAA", "SPY"])
    forward._positions(cfg, "agent", bars, tmp_path,
                       agent=forward._build_agent(cfg, market_context=True))
    assert seen and all("Market today: SPY_1d=" in p for p in seen)


def test_tick_and_reflect_records_flag_and_skips_reflection(monkeypatch, tmp_path):
    bars = _bars(["AAA", "SPY"], n=40)
    cfg = _cfg(["AAA", "SPY"])

    def fake_complete(prompt):
        return '{"target": 1, "reason": "long"}'

    monkeypatch.setattr("rhagent.engine.AgentEngine._default_complete",
                        lambda self: fake_complete)
    reflected = []
    monkeypatch.setattr("rhagent.memory.reflect",
                        lambda *a, **k: (reflected.append(1), "x")[1])

    def fake_fetch(*a, **k):
        raise AssertionError("must not fetch")

    cache = tmp_path / "data"
    cache.mkdir()
    # Seed-then-extend, same pattern as test_memory.py: a fresh eval_dir's
    # first tick only anchors (seeds every bar but the newest as never-decided
    # legacy, and decides the newest -- but that newest bar can't be "fully
    # realized" yet since no bar follows it, so nothing is appended). Write
    # one more bar afterward so the previously-decided bar gets a successor
    # and becomes realized on the second call.
    for s, f in bars.items():
        f.iloc[:-1].to_csv(cache / f"{s}.csv", index_label="date")

    today = date(2025, 2, 25)  # at/after the 40th (last) bar's date
    ed = tmp_path / "agent_ctx"
    agent = forward._build_agent(cfg, market_context=True)
    forward.tick_and_reflect(cfg, ed, 7.0, engine="agent", fill="next_open",
                             today=today, cache_dir=cache, agent=agent,
                             memory_path=str(tmp_path / "mem.md"))

    ed2 = tmp_path / "agent"
    forward.tick_and_reflect(cfg, ed2, 7.0, engine="agent", fill="next_open",
                             today=today, cache_dir=cache,
                             agent=forward._build_agent(cfg),
                             memory_path=str(tmp_path / "mem.md"))

    for s, f in bars.items():
        f.to_csv(cache / f"{s}.csv", index_label="date")

    res = forward.tick_and_reflect(cfg, ed, 7.0, engine="agent", fill="next_open",
                                   today=today, cache_dir=cache, agent=agent,
                                   memory_path=str(tmp_path / "mem.md"))
    assert res["appended"] >= 1
    meta = json.loads((ed / "run.json").read_text())
    assert meta["market_context"] is True
    assert meta["reflected"] is False
    assert reflected == []          # ctx record never touches the shared memory

    res2 = forward.tick_and_reflect(cfg, ed2, 7.0, engine="agent", fill="next_open",
                                    today=today, cache_dir=cache,
                                    agent=forward._build_agent(cfg),
                                    memory_path=str(tmp_path / "mem.md"))
    assert res2["appended"] >= 1
    meta2 = json.loads((ed2 / "run.json").read_text())
    assert meta2["market_context"] is False
    assert meta2["reflected"] is True
    assert reflected == [1]         # control record reflects (spy returns "x")


def test_cli_flag_reaches_engine(monkeypatch, tmp_path):
    captured = {}

    def fake_tick_and_reflect(cfg, eval_dir, cost_bps, **kw):
        captured["agent"] = kw.get("agent")
        captured["eval_dir"] = eval_dir
        return {"appended": 0, "total_days": 0, "meta": {}}

    monkeypatch.setattr(forward, "tick_and_reflect", fake_tick_and_reflect)
    monkeypatch.setattr(forward, "_report", lambda *a, **k: None)
    forward.main(["--engine", "agent", "--eval-id", "agent_ctx", "--market-context",
                  "--out-dir", str(tmp_path)])
    assert captured["agent"] is not None
    assert captured["agent"].market_context is True
    assert captured["eval_dir"] == Path(tmp_path) / "agent_ctx"

    forward.main(["--engine", "agent", "--out-dir", str(tmp_path)])
    assert captured["agent"] is None   # control path unchanged: engine built inside
