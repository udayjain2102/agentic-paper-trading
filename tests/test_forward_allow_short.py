"""T1: config.yaml's agent.allow_short must actually reach the AgentEngine
that a production tick constructs -- both call sites in forward.py used to
omit the kwarg entirely, silently running allow_short=False forever.

T2: a tick where every genuine decision is flat must print a loud warning
(the cheap check that would have caught the allow_short regression)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rhagent import forward
from rhagent.config import load


def test_config_allow_short_defaults_true():
    cfg = load()
    assert cfg.agent.allow_short is True


def test_positions_wires_allow_short_into_agent_engine(monkeypatch, tmp_path):
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    close = pd.Series(np.linspace(100, 90, 30), index=idx)  # steady downtrend
    bars = {"AAA": pd.DataFrame({"open": close, "high": close, "low": close,
                                 "close": close, "volume": 1e6}, index=idx)}
    cfg = SimpleNamespace(
        strategy=SimpleNamespace(name="agent", params={}, universe=["AAA"], overlay="none"),
        agent=SimpleNamespace(model="", max_tokens=None, allow_short=True,
                              use_lessons=False),
    )

    def fake_complete(_prompt):
        return '{"target": -1, "reason": "short"}'  # decide()'s single-symbol shape

    # decide_all's lazy default client is _default_complete (decide_all fans
    # out per-symbol calls through decide() -- see engine.py).
    monkeypatch.setattr(
        "rhagent.engine.AgentEngine._default_complete",
        lambda self: fake_complete,
    )

    pos, _excluded = forward._positions(cfg, "agent", bars, tmp_path)
    # allow_short=True reached the engine iff a -1 target actually survives
    # (engine.py zeroes -1 targets when allow_short=False).
    assert (pos["AAA"] == -1.0).any()


def test_all_flat_tick_warns(capsys):
    rows = [
        {"date": "2026-01-01", "symbol": "AAA", "target": 0.0, "reason": "x", "status": "ok"},
        {"date": "2026-01-01", "symbol": "BBB", "target": 0.0, "reason": "y", "status": "ok"},
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        forward._append_decisions(Path(d), rows)
    err = capsys.readouterr().err
    assert "all" in err and "flat" in err


def test_mixed_tick_does_not_warn(capsys):
    rows = [
        {"date": "2026-01-01", "symbol": "AAA", "target": 1.0, "reason": "x", "status": "ok"},
        {"date": "2026-01-01", "symbol": "BBB", "target": 0.0, "reason": "y", "status": "ok"},
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        forward._append_decisions(Path(d), rows)
    err = capsys.readouterr().err
    assert err == ""


def test_use_lessons_false_sends_no_lessons_block(monkeypatch, tmp_path):
    """production default: the stale caution-heavy lessons text (which produced
    259 flat decisions of 260) must not reach the prompt when the knob is off."""
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    close = pd.Series(np.linspace(100, 110, 30), index=idx)
    bars = {"AAA": pd.DataFrame({"open": close, "high": close, "low": close,
                                 "close": close, "volume": 1e6}, index=idx)}
    cfg = SimpleNamespace(
        strategy=SimpleNamespace(name="agent", params={}, universe=["AAA"], overlay="none"),
        agent=SimpleNamespace(model="", max_tokens=None, allow_short=True,
                              use_lessons=False),
    )
    # if the knob leaks, these would be pulled into the prompt
    monkeypatch.setattr("rhagent.memory.read_memory", lambda *a, **k: "MEMORY_LEAK")
    monkeypatch.setattr("rhagent.learn.lessons_from_runs", lambda *a, **k: "LESSONS_LEAK")

    seen = []

    def fake_complete(prompt):
        seen.append(prompt)
        return '{"target": 1, "reason": "long"}'

    monkeypatch.setattr("rhagent.engine.AgentEngine._default_complete",
                        lambda self: fake_complete)
    eval_dir = tmp_path / "ed"
    eval_dir.mkdir()
    forward._positions(cfg, "agent", bars, eval_dir)

    assert seen, "no prompt was built"
    assert "MEMORY_LEAK" not in seen[0]
    assert "LESSONS_LEAK" not in seen[0]
