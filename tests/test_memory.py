"""Memory loop: read/append/cap, recent_outcomes, reflect, and forward wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from rhagent import forward
from rhagent.memory import append_reflection, read_memory, recent_outcomes, reflect


def test_append_read_roundtrip(tmp_path):
    p = tmp_path / "mem.md"
    assert read_memory(p) == ""
    append_reflection(p, "2026-07-01", "- lesson one")
    append_reflection(p, "2026-07-02", "- lesson two")
    text = read_memory(p)
    assert "## 2026-07-01" in text
    assert "## 2026-07-02" in text
    assert "lesson one" in text and "lesson two" in text


def test_cap_drops_oldest(tmp_path):
    p = tmp_path / "mem.md"
    for i in range(45):
        append_reflection(p, f"2026-01-{i:02d}", f"entry {i}")
    text = read_memory(p)
    entries = text.split("\n## ")[1:]
    assert len(entries) == 40
    assert "entry 0" not in text          # oldest dropped
    assert "entry 44" in text             # newest kept


def _bars(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes}, index=idx)


def test_recent_outcomes_mentions_symbol_pos_and_return_sign(tmp_path):
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    close = [10, 11, 12, 13, 14]  # steadily rising -> long position wins
    bars = {"AAA": _bars(close)}
    pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    pos.rename_axis("date").rename("pos").to_csv(tmp_path / "pos_AAA.csv")

    out = recent_outcomes(tmp_path, bars, n_days=5)
    assert "AAA" in out
    assert "pos=+1" in out
    assert "next_day_ret=+" in out  # long + rising price -> positive next-day ret


def test_reflect_appends_on_success(tmp_path):
    p = tmp_path / "mem.md"
    text = reflect(lambda prompt: "- worked: momentum longs", p, "AAA pos=+1", "2026-07-18")
    assert text == "- worked: momentum longs"
    assert "## 2026-07-18" in read_memory(p)
    assert "worked: momentum longs" in read_memory(p)


def test_reflect_noop_on_failure(tmp_path):
    p = tmp_path / "mem.md"

    def boom(prompt):
        raise RuntimeError("model down")

    assert reflect(boom, p, "AAA pos=+1", "2026-07-18") == ""
    assert read_memory(p) == ""

    assert reflect(lambda p_: "", p, "AAA pos=+1", "2026-07-18") == ""
    assert read_memory(p) == ""


def _cfg(universe):
    return SimpleNamespace(strategy=SimpleNamespace(
        name="mean_reversion", params={}, universe=universe, overlay="none"))


def test_forward_tick_and_reflect_writes_memory_and_meta(tmp_path, monkeypatch):
    import numpy as np

    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    bars = {"AAA": pd.DataFrame({"open": close, "close": close})}
    cache = tmp_path / "cache"
    cache.mkdir()
    bars["AAA"].to_csv(cache / "AAA.csv", index_label="date")

    monkeypatch.chdir(tmp_path)
    from datetime import date

    from rhagent.engine import AgentEngine

    calls = {"agent": 0, "reflect": 0}

    def agent_complete(_prompt):
        calls["agent"] += 1
        return '{"target": 1, "reason": "test"}'

    def reflect_complete(_prompt):
        calls["reflect"] += 1
        return "- lesson from today"

    eval_dir = tmp_path / "journal" / "forward" / "agent"
    cfg = _cfg(["AAA"])
    res = forward.tick_and_reflect(
        cfg, eval_dir, today=date(2026, 3, 20), cache_dir=cache, engine="agent",
        agent=AgentEngine(complete=agent_complete), reflect_complete=reflect_complete,
        memory_path=str(tmp_path / "journal" / "agent_memory.md"),
    )
    assert res["appended"] == 1
    assert calls["reflect"] == 1
    mem_text = read_memory(tmp_path / "journal" / "agent_memory.md")
    assert "lesson from today" in mem_text

    import json
    meta = json.loads((eval_dir / "run.json").read_text())
    assert meta["reflected"] is True
    assert "memory_chars" in meta

    # second same-day tick appends 0 -> must not reflect again
    res2 = forward.tick_and_reflect(
        cfg, eval_dir, today=date(2026, 3, 20), cache_dir=cache, engine="agent",
        agent=AgentEngine(complete=agent_complete), reflect_complete=reflect_complete,
        memory_path=str(tmp_path / "journal" / "agent_memory.md"),
    )
    assert res2["appended"] == 0
    assert calls["reflect"] == 1  # unchanged


def test_positions_lessons_include_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mem_path = tmp_path / "journal" / "agent_memory.md"
    append_reflection(mem_path, "2026-07-01", "- avoid overtrading small caps")

    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    bars = {"AAA": _bars([10, 11, 12, 13, 14])}
    cfg = _cfg(["AAA"])

    captured = {}

    class FakeAgent:
        def __init__(self, lessons=""):
            captured["lessons"] = lessons

        def decide(self, symbol, history, current_pos):
            from rhagent.engine import Decision
            return Decision(target=0.0, reason="noop")

    monkeypatch.setattr("rhagent.engine.AgentEngine", FakeAgent)
    ed = tmp_path / "ed"
    ed.mkdir()
    forward._positions(cfg, "agent", bars, ed)
    assert "avoid overtrading small caps" in captured["lessons"]


def test_reflect_failure_is_loud(tmp_path, capsys):
    def boom(prompt):
        raise RuntimeError("model down")

    reflect(boom, tmp_path / "mem.md", "AAA pos=+1", "2026-07-18")
    assert "model call failed" in capsys.readouterr().err

    reflect(lambda p_: "", tmp_path / "mem.md", "AAA pos=+1", "2026-07-18")
    assert "empty text" in capsys.readouterr().err


def test_agent_positions_log_decisions_with_reason(tmp_path):
    import json

    from rhagent.engine import Decision

    class FakeAgent:
        def decide(self, symbol, history, current_pos):
            return Decision(target=1.0, reason="agent: dip buy")

    bars = _bars([10, 11, 12, 13, 14])
    forward._agent_positions(tmp_path, "AAA", bars, FakeAgent())
    lines = [json.loads(l) for l in
             (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert lines and lines[-1]["symbol"] == "AAA"
    assert lines[-1]["target"] == 1.0
    assert lines[-1]["reason"] == "agent: dip buy"
    # second call: all bars cached, nothing new appended
    forward._agent_positions(tmp_path, "AAA", bars, FakeAgent())
    n2 = len((tmp_path / "decisions.jsonl").read_text().splitlines())
    assert n2 == len(lines)
