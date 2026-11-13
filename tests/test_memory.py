"""Memory loop: one current block + archive, recent_outcomes, reflect, forward wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from rhagent import forward
from rhagent.memory import (archive_path, read_memory, recent_outcomes,
                            reflect, write_lessons)


def test_write_replaces_the_block_and_archives_every_one(tmp_path):
    p = tmp_path / "mem.md"
    assert read_memory(p) == ""
    write_lessons(p, "2026-07-01", "- lesson one")
    write_lessons(p, "2026-07-02", "- lesson two")
    text = read_memory(p)
    assert "## 2026-07-02" in text and "lesson two" in text
    assert "lesson one" not in text            # the prompt sees one block only
    arch = archive_path(p).read_text()
    assert "lesson one" in arch and "lesson two" in arch


def test_first_write_moves_the_old_log_into_the_archive(tmp_path):
    p = tmp_path / "mem.md"
    p.write_text("\n## 2026-01-01\nold a\n\n## 2026-01-02\nold b\n")
    assert "old b" in read_memory(p) and "old a" not in read_memory(p)
    # the old free-form log must not seed the new block
    seen = {}
    reflect(lambda pr: seen.setdefault("p", pr) and "- new rule", p, "x", "2026-02-01")
    assert "old b" not in seen["p"] and "(none yet)" in seen["p"]
    assert "old" not in p.read_text()
    arch = archive_path(p).read_text()
    assert "old a" in arch and "old b" in arch and "new rule" in arch


def test_reflect_keeps_only_capped_bullets(tmp_path):
    p = tmp_path / "mem.md"
    reply = "Here are my lessons:\n" + "\n".join(
        f"* **rule {i}**   with   spaces\n  a sub-paragraph" for i in range(8))
    out = reflect(lambda pr: reply, p, "x", "2026-02-01")
    assert out.splitlines() == [f"- **rule {i}** with spaces" for i in range(5)]
    # prose with no bullets is rejected and the existing block survives
    assert reflect(lambda pr: "no bullets here", p, "x", "2026-02-02") == ""
    assert "## 2026-02-01" in read_memory(p)


def test_recent_outcomes_skips_failed_decisions(tmp_path):
    idx = pd.date_range("2026-01-01", periods=4, freq="B")
    pd.DataFrame({"pos": [1.0, 1.0, 1.0, 1.0],
                  "status": ["ok", "failed", "failed", "ok"]},
                 index=idx.rename("date")).to_csv(tmp_path / "pos_AAA.csv")
    out = recent_outcomes(tmp_path, {"AAA": _bars([10, 11, 12, 13])})
    assert str(idx[0].date()) in out and str(idx[3].date()) in out
    assert str(idx[1].date()) not in out and str(idx[2].date()) not in out


def _bars(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes}, index=idx)


def test_recent_outcomes_splits_a_big_day_by_side_and_setup(tmp_path):
    idx = pd.date_range("2026-01-01", periods=3, freq="B")
    bars = {}
    for i in range(8):
        sym = f"S{i}"
        bars[sym] = _bars([10, 11, 12])                      # rising: longs win
        pd.DataFrame({"pos": [1.0 if i < 5 else -1.0] * 3, "status": "ok"},
                     index=idx.rename("date")).to_csv(tmp_path / f"pos_{sym}.csv")
    out = recent_outcomes(tmp_path, bars)
    assert "long 5 names avg next_day_ret=+" in out
    assert "short 3 names avg next_day_ret=-" in out
    assert "long with momentum_5d up: n=5" in out and "win_rate 100%" in out
    assert "short with momentum_5d up: n=3" in out and "win_rate 0%" in out


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
    return SimpleNamespace(
        strategy=SimpleNamespace(
            name="mean_reversion", params={}, universe=universe, overlay="none"),
        # use_lessons=True: this test is the guard that the memory/lessons block
        # still reaches the prompt when the knob is on (production default is
        # off -- see config.yaml).
        agent=SimpleNamespace(model="", max_tokens=None, allow_short=True,
                              use_lessons=True),
    )


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

    calls = {"agent": [], "reflect": 0}

    def agent_complete(_prompt):
        calls["agent"].append(_prompt)
        return '{"target": 1, "reason": "x"}'  # decide()'s single-symbol shape

    def reflect_complete(_prompt):
        calls["reflect"] += 1
        return "- lesson from today"

    eval_dir = tmp_path / "journal" / "forward" / "agent"
    cfg = _cfg(["AAA"])
    # First tick only anchors: every prior bar is seeded (never decided), so no
    # realized day is a genuine decision yet and nothing is appended.
    bars["AAA"].iloc[:-1].to_csv(cache / "AAA.csv", index_label="date")
    res0 = forward.tick_and_reflect(
        cfg, eval_dir, today=date(2026, 3, 20), cache_dir=cache, engine="agent",
        agent=AgentEngine(complete=agent_complete), reflect_complete=reflect_complete,
        memory_path=str(tmp_path / "journal" / "agent_memory.md"),
    )
    assert res0["appended"] == 0
    assert calls["reflect"] == 0

    # a new bar prints: the previously-decided day is now realized
    bars["AAA"].to_csv(cache / "AAA.csv", index_label="date")
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
    write_lessons(mem_path, "2026-07-01", "- avoid overtrading small caps")

    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    bars = {"AAA": _bars([10, 11, 12, 13, 14])}
    cfg = _cfg(["AAA"])

    captured = {}

    class FakeAgent:
        def __init__(self, lessons="", allow_short=None, market_context=False):
            captured["lessons"] = lessons

        def decide_all(self, symbols, histories, current_pos, **_):
            from rhagent.engine import Decision
            return {s: Decision(target=0.0, reason="noop") for s in symbols}

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
        def __init__(self):
            self.calls = 0

        def decide_all(self, symbols, histories, current_pos, **_):
            self.calls += 1
            return {s: Decision(target=1.0, reason="agent: dip buy") for s in symbols}

    bars = {"AAA": _bars([10, 11, 12, 13, 14]), "BBB": _bars([20, 21, 22, 23, 24])}
    agent = FakeAgent()
    pos, excluded = forward._agent_positions(tmp_path, bars, agent)
    assert agent.calls == 1  # one call for the bar, not one per symbol
    lines = [json.loads(l) for l in
             (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert lines and lines[-1]["symbol"] in ("AAA", "BBB")
    assert lines[-1]["target"] == 1.0
    assert lines[-1]["reason"] == "agent: dip buy"
    assert lines[-1]["status"] == "ok"
    # the seeded anchor bars were never decided -> excluded, the decided one isn't
    assert excluded == set(bars["AAA"].index[:-1])
    # second call: all bars cached, nothing new appended and no model call
    forward._agent_positions(tmp_path, bars, agent)
    assert agent.calls == 1
    n2 = len((tmp_path / "decisions.jsonl").read_text().splitlines())
    assert n2 == len(lines)


def test_memory_chars_reports_what_the_engine_held_not_the_file(tmp_path):
    """run.json is an audit trail of the education the agent actually got.
    With use_lessons off (the production default -- see config.yaml) the
    lessons string is empty, so a fat agent_memory.md on disk must not be
    reported as memory the agent received."""
    import json
    from datetime import date

    from rhagent.engine import AgentEngine

    mem = tmp_path / "agent_memory.md"
    write_lessons(mem, "2026-07-20", "a long prior lesson " * 20)
    assert len(read_memory(mem)) > 100, "fixture must have real content on disk"

    cache, eval_dir = tmp_path / "cache", tmp_path / "fwd"
    cache.mkdir()
    idx = pd.date_range("2026-01-01", periods=12, freq="B")
    close = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)
    pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                  "volume": 1e6}, index=idx).to_csv(cache / "AAA.csv",
                                                    index_label="date")

    cfg = _cfg(["AAA"])
    cfg.agent.use_lessons = False
    forward.tick_and_reflect(
        cfg, eval_dir, today=date(2026, 3, 20), cache_dir=cache, engine="agent",
        agent=AgentEngine(complete=lambda _p: '{"target": 1, "reason": "x"}', lessons=""),
        reflect_complete=lambda _p: "", memory_path=str(mem))

    meta = json.loads((eval_dir / "run.json").read_text())
    assert meta["memory_chars"] == 0, (
        f"reported {meta['memory_chars']} chars of memory the agent never saw")
