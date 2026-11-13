import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

_SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "make_dashboard.py"


def _dashboard_module():
    spec = importlib.util.spec_from_file_location("make_dashboard", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(run_dir: Path, *, engine="mean_reversion", net=(0.01,)):
    run_dir.mkdir(parents=True)
    rid = run_dir.name
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": rid,
        "engine": engine,
        "symbols": ["A"],
        "start": "2026-07-01",
        "end": "2026-07-02",
        "cost_bps": 1.0,
        "notional": 10_000.0,
    }))
    (run_dir / "trades.jsonl").write_text("")
    idx = pd.date_range("2026-07-01", periods=len(net), freq="D")
    pd.DataFrame({"date": idx, "net": list(net)}).to_csv(
        run_dir / "returns.csv", index=False
    )


def test_default_dashboard_writes_only_one_dashboard_html(tmp_path):
    mod = _dashboard_module()
    paper = tmp_path / "journal" / "papertrade"
    _write_run(paper / "2026-07-12T00-00-00Z-aaaaaaaa")
    _write_run(tmp_path / "journal" / "forward" / "mean_reversion")

    assert mod.main(["--base-dir", str(paper)]) == 0

    dashboard = tmp_path / "journal" / "dashboard.html"
    assert dashboard.exists()
    html = dashboard.read_text()
    assert "Trading Control Room" in html
    assert "Runbook" in html
    # the run appears in the embedded control-room data blob
    assert "2026-07-12T00-00-00Z-aaaaaaaa" in html

    for legacy_dashboard in [
        tmp_path / "journal" / "papertrade" / "dashboard.html",
        tmp_path / "journal" / "forward" / "dashboard.html",
    ]:
        assert not legacy_dashboard.exists()


def test_every_js_render_target_exists_in_the_static_shell():
    """The JS writes into elements by id, so a renamed id silently blanks a whole
    panel. Every $('...') target must be present in the HTML skeleton."""
    src = _SOURCE.read_text()
    targets = set(re.findall(r"\$\('([\w-]+)'\)", src))
    declared = set(re.findall(r'id="([\w-]+)"', src))
    assert targets, "no $('id') calls found -- this check has drifted from the source"
    assert targets <= declared, f"JS targets with no element: {sorted(targets - declared)}"


def test_no_fabricated_breach_literal():
    """The dashboard must never print a hardcoded '0 breaches' -- guardrail
    rejections are read from journal/paper_orders.jsonl, real count or zero."""
    src = _SOURCE.read_text()
    assert "breaches" not in src


def test_rejected_order_count_reads_real_journal(tmp_path):
    mod = _dashboard_module()
    journal = tmp_path / "paper_orders.jsonl"
    journal.write_text(
        json.dumps({"event": "order_rejected"}) + "\n"
        + json.dumps({"event": "order_placed"}) + "\n"
        + json.dumps({"event": "order_rejected"}) + "\n"
    )
    assert mod._rejected_order_count(journal) == 2
    assert mod._rejected_order_count(tmp_path / "missing.jsonl") == 0


def test_bakeoff_survives_all_none_deflated(tmp_path, monkeypatch):
    """A record too short for the bootstrap/fold stats must not crash
    idxmax() on an all-NaN 'deflated' column."""
    mod = _dashboard_module()
    monkeypatch.setattr(mod, "robust_table", lambda base_dir: pd.DataFrame({
        "engine": ["agent"], "overlay": ["none"], "point_sharpe": [0.1],
        "deflated": [None], "fold_mean": [None], "fold_std": [None],
        "ci_lo": [None], "ci_hi": [None], "beats_baseline": [False],
        "n_return_days": [3],
    }))
    rows = mod._bakeoff_data(tmp_path, "agent")
    assert rows[0]["deflated"] is None
    assert "n/a" in rows[0]["fold"]


def test_locked_buckets_rank_by_dollars_relative_to_size(tmp_path):
    """The loss panel must surface the bucket that loses more than its share of
    trades, in dollars -- not the biggest bucket, which is what a raw share of
    losing-trade counts ranks first (and ranks first in the win panel too)."""
    run = tmp_path / "r1"
    _write_run(run)
    # 6 quick trades: five +10 wins, one -1 loss. 4 slow trades: one +5 win,
    # three -20 losses. Slow = 40% of trades but 60/61 of the loss dollars.
    spec = [(1, 10.0)] * 5 + [(1, -1.0)] + [(8, 5.0)] + [(8, -20.0)] * 3
    (run / "trades.jsonl").write_text("".join(
        json.dumps({"symbol": f"S{i}", "side": "long", "holding_bars": bars,
                    "pnl_abs": pnl, "outcome": "win" if pnl > 0 else "loss",
                    # vol scattered across the list so it does not track holding time
                    "entry_features": {"vol20": 0.01 + (i * 7 % 10) / 1000, "gap": 0.0}}) + "\n"
        for i, (bars, pnl) in enumerate(spec)))

    loss, win = _dashboard_module()._locked_buckets(run)

    top = loss[0]
    assert (top["dim"], top["bucket"]) == ("holding", "5+ bars")
    assert abs(top["share"] - 60 / 61) < 1e-9 and abs(top["tradeShare"] - 0.4) < 1e-9
    assert top["lift"] > 2
    assert (win[0]["dim"], win[0]["bucket"]) != ("holding", "5+ bars")
    for r in loss + win:
        assert r["lift"] > 1                      # proportional buckets are not findings
        assert r["dim"] not in ("symbol", "side")  # side=long is 100% of trades: lift 1
        if r["dim"] == "holding":                 # never the bare "short"/"long" of a side
            assert r["bucket"] in ("under 5 bars", "5+ bars")

    empty = tmp_path / "r0"
    _write_run(empty)
    assert _dashboard_module()._locked_buckets(empty) == ([], [])


# ---- single-page data layer: records, today, memory, experiments, health ----

import shutil  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

NOW = datetime(2026, 7, 6, 12, tzinfo=timezone.utc)
SPEC = ("agent", "Agent", "agent", "7 bp", "n")


def _jl(path: Path, *rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _spy(data: Path, n=6):
    data.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2026-07-01", periods=n, freq="D")
    pd.DataFrame({"date": idx, "close": [100.0 + i for i in range(n)]}).to_csv(data / "SPY.csv", index=False)


def _returns(fwd: Path, rid: str, net, start="2026-07-01"):
    (fwd / rid).mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(start, periods=len(net), freq="D")
    pd.DataFrame({"date": idx, "net": list(net)}).to_csv(fwd / rid / "returns.csv", index=False)


def _pos(fwd: Path, rid: str, sym: str, rows):
    (fwd / rid).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["date", "pos", "status"]).to_csv(fwd / rid / f"pos_{sym}.csv", index=False)


def test_record_numbers_match_hand_computation(tmp_path):
    mod = _dashboard_module()
    fwd, data = tmp_path / "forward", tmp_path / "data"
    _spy(data)  # SPY bars 07-01..07-06, closes 100..105
    _returns(fwd, "agent", [0.10, -0.05, 0.02])
    _jl(fwd / "agent" / "decisions.jsonl",
        {"date": "2026-07-01", "symbol": "A", "status": "failed"},
        {"date": "2026-07-01", "symbol": "A", "status": "ok"},   # retry resolves it
        {"date": "2026-07-01", "symbol": "B", "status": "failed"})
    r = mod._record(SPEC, fwd, mod._spy_dates(data), data)
    assert not r["missing"] and r["days"] == 3
    assert (r["start"], r["end"]) == ("2026-07-01", "2026-07-03")
    assert r["totalReturn"] == pytest.approx(1.1 * 0.95 * 1.02 - 1)       # 0.0659
    assert r["maxDrawdown"] == pytest.approx(-0.05)                        # 1.1 -> 1.045
    assert r["lastNet"] == 0.02
    assert r["curve"]["dates"] == ["2026-07-01", "2026-07-02", "2026-07-03"]
    assert r["curve"]["equity"] == pytest.approx([1.1, 1.045, 1.0659])
    assert r["behind"] == 2          # bars 07-04, 07-05, 07-06 after end, minus the unrealized newest
    assert r["spyReturn"] == pytest.approx(102 / 100 - 1)
    assert r["sharpe"] is None and r["sharpeNeeds"] == 20
    assert r["decisions"] == {"ok": 1, "failed": 1}


def test_record_with_enough_days_has_sharpe_and_rule_has_no_decisions(tmp_path):
    mod = _dashboard_module()
    fwd = tmp_path / "forward"
    _returns(fwd, "mean_reversion", [0.001, -0.002] * 15)
    r = mod._record(("mean_reversion", "Rule", "rule", "b", "n"), fwd, [], tmp_path / "data")
    assert r["sharpe"] is not None and r["sharpeNeeds"] is None
    assert r["decisions"] is None
    assert r["behind"] is None and r["spyReturn"] is None   # no SPY file


def test_missing_record_shape(tmp_path):
    mod = _dashboard_module()
    fwd = tmp_path / "forward"
    expect = mod._empty_record(SPEC)
    assert mod._record(SPEC, fwd, [], tmp_path) == expect     # no directory
    _returns(fwd, "agent", [])                                  # header-only file
    assert mod._record(SPEC, fwd, [], tmp_path) == expect
    assert expect["missing"] and expect["curve"] == {"dates": [], "equity": []}
    assert all(v is None for k, v in expect.items()
               if k not in ("id", "label", "kind", "basis", "note", "missing", "curve"))


def test_today_agents_counts_and_changes(tmp_path):
    mod = _dashboard_module()
    fwd = tmp_path / "forward"
    d1, d2 = "2026-07-01", "2026-07-02"
    _pos(fwd, "agent", "A", [(d1, 0.0, "ok"), (d2, 1.0, "ok")])      # long, 0 -> 1
    _pos(fwd, "agent", "B", [(d1, 1.0, "ok"), (d2, -1.0, "ok")])     # short, 1 -> -1
    _pos(fwd, "agent", "C", [(d1, 1.0, "ok"), (d2, 1.0, "failed")])  # failed: held, not a change
    _pos(fwd, "agent", "D", [(d1, 0.0, "ok"), (d2, 0.0, "ok")])      # flat, unchanged
    _jl(fwd / "agent" / "decisions.jsonl",
        {"date": d2, "symbol": "A", "reason": "ra", "status": "ok"},
        {"date": d2, "symbol": "B", "reason": "rb", "status": "ok"})
    [a] = mod._today_agents(fwd)
    assert (a["id"], a["date"]) == ("agent", d2)
    assert (a["long"], a["short"], a["flat"], a["failed"]) == (1, 1, 1, 1)
    assert a["changes"] == [{"symbol": "B", "from": 1.0, "to": -1.0, "reason": "rb"},
                            {"symbol": "A", "from": 0.0, "to": 1.0, "reason": "ra"}]
    assert mod._today_agents(tmp_path / "nothing") == []


def test_today_open_newest_line_or_none(tmp_path):
    mod = _dashboard_module()
    fwd = tmp_path / "forward"
    assert mod._today_open(fwd) is None
    _jl(fwd / "agent_open" / "decisions.jsonl",
        {"date": "2026-07-01", "status": "failed", "fail": "old"},
        {"date": "2026-07-02", "status": "ok", "scan_date": "2026-07-01", "n_floor": 900,
         "screens": [{"feature": "z20", "order": "lowest", "n": 5}], "screen_reason": "dips",
         "candidates": 7, "positions": [{"symbol": "ZZ", "target": 1, "reason": "cheap"}],
         "reason": "one name"})
    o = mod._today_open(fwd)
    assert (o["date"], o["status"], o["scanDate"], o["nFloor"], o["candidates"]) == ("2026-07-02", "ok", "2026-07-01", 900, 7)
    assert o["positions"] == [{"symbol": "ZZ", "target": 1, "reason": "cheap"}]
    assert o["screenReason"] == "dips" and o["reason"] == "one name" and o["fail"] is None


def test_today_rule_enter_exit_held(tmp_path, monkeypatch):
    mod = _dashboard_module()
    idx = pd.to_datetime(["2026-07-01", "2026-07-02"])
    series = {"A": [0, 1], "B": [1, 0], "C": [1, 1], "D": [0, 0]}
    monkeypatch.setattr(mod, "_positions", lambda *a: ({s: pd.Series(v, index=idx, dtype=float) for s, v in series.items()}, set()))
    cfg = SimpleNamespace(strategy=SimpleNamespace(universe=list(series), name="mean_reversion"))
    assert mod._today_rule(cfg, tmp_path / "data") == {
        "date": "2026-07-02", "held": ["A", "C"], "entered": ["A"], "exited": ["B"]}


def test_today_paper_newest_tick_and_halt(tmp_path):
    mod = _dashboard_module()
    assert mod._today_paper(tmp_path) is None
    _jl(tmp_path / "paper_orders.jsonl",
        {"event": "order_rejected", "reason": "old", "ts": "2026-07-01T10:00:00+00:00"},
        {"event": "paper_run_end", "n_orders": 1, "n_accepted": 0, "ts": "2026-07-01T10:00:01+00:00"},
        {"event": "order_placed", "ts": "2026-07-02T10:00:00+00:00"},
        {"event": "order_rejected", "reason": "cap", "ts": "2026-07-02T10:00:00+00:00"},
        {"event": "order_rejected", "reason": "cap", "ts": "2026-07-02T10:00:00+00:00"},
        {"event": "paper_run_end", "n_orders": 3, "n_accepted": 1, "ts": "2026-07-02T10:00:01+00:00"})
    (tmp_path / "paper_account.json").write_text(json.dumps(
        {"cash_usd": 4000.0, "date": "2026-07-02", "realized_pnl_today_usd": 5.0,
         "positions": {"X": {"shares": 1, "cost_basis_usd": 2}}}))
    p = mod._today_paper(tmp_path)
    assert (p["date"], p["orders"], p["accepted"], p["rejected"]) == ("2026-07-02", 3, 1, 2)
    assert p["rejectReasons"] == {"cap": 2} and not p["halted"] and p["haltReason"] is None
    assert p["account"] == {"date": "2026-07-02", "cashUsd": 4000.0, "realizedPnlTodayUsd": 5.0, "positions": ["X"]}
    with (tmp_path / "paper_orders.jsonl").open("a") as f:
        f.write(json.dumps({"event": "paper_run_halted", "reason": "HALT file", "ts": "2026-07-03T10:00:00+00:00"}) + "\n")
    p = mod._today_paper(tmp_path)
    assert p["halted"] and p["haltReason"] == "HALT file" and p["orders"] == 0 and p["rejected"] == 0


def test_memory_one_block_and_legacy(tmp_path):
    mod = _dashboard_module()
    assert mod._memory(tmp_path) == {"date": None, "bullets": [], "legacy": True, "archiveEntries": 0}
    (tmp_path / "agent_memory.md").write_text("\n## 2026-07-01\nold free text\n\n## 2026-07-02\n- a\n- b\n")
    m = mod._memory(tmp_path)                       # old log: only its last entry, no archive yet
    assert (m["date"], m["bullets"], m["legacy"], m["archiveEntries"]) == ("2026-07-02", ["a", "b"], True, 0)
    (tmp_path / "agent_memory_archive.md").write_text("\n## 2026-07-01\nx\n\n## 2026-07-02\n- a\n- b\n")
    (tmp_path / "agent_memory.md").write_text("\n## 2026-07-03\n- only\n")
    m = mod._memory(tmp_path)
    assert (m["date"], m["bullets"], m["legacy"], m["archiveEntries"]) == ("2026-07-03", ["only"], False, 2)


def test_experiments_newest_first_and_missing(tmp_path):
    mod = _dashboard_module()
    assert mod._experiments(tmp_path / "none.jsonl") == []
    _jl(tmp_path / "r.jsonl",
        {"created": "2026-07-01T00:00:00+00:00", "experiment": "e1", "variant": "old", "sharpe": 0.1},
        {"created": "2026-07-03T00:00:00+00:00", "experiment": "e2", "variant": "a", "sharpe": 0.5,
         "sharpe_ci_lo": -0.1, "sharpe_ci_hi": 0.9, "ann_return": 0.04, "max_drawdown": -0.2,
         "start": "2003-01-02", "end": "2025-05-30", "note": "n"},
        {"created": "2026-07-03T00:00:00+00:00", "experiment": "e2", "variant": "b", "sharpe": 0.2})
    rows = mod._experiments(tmp_path / "r.jsonl")
    assert [r["variant"] for r in rows] == ["b", "a", "old"]          # same-second ties: later line first
    assert rows[1] == {"created": "2026-07-03T00:00:00+00:00", "experiment": "e2", "variant": "a", "sharpe": 0.5,
                       "ciLo": -0.1, "ciHi": 0.9, "annReturn": 0.04, "maxDrawdown": -0.2,
                       "start": "2003-01-02", "end": "2025-05-30", "note": "n"}


def _rec(rid, behind=0, missing=False):
    r = {**_dashboard_module()._empty_record((rid, rid, "agent", "", "")), "missing": missing}
    if not missing:
        r.update(behind=behind, end="2026-07-03")
    return r


def _levels(items):
    return {i["id"]: i["level"] for i in items}


def test_health_record_freshness_levels():
    mod = _dashboard_module()
    lv = _levels(mod._chk_records([_rec("a", 0), _rec("b", 1), _rec("c", 3), _rec("d", 4),
                                  _rec("e", missing=True), _rec("agent_open", missing=True),
                                  _rec("f", behind=None)]))
    assert lv == {"record:a": "ok", "record:b": "warn", "record:c": "warn", "record:d": "bad",
                  "record:e": "bad", "record:agent_open": "warn", "record:f": "warn"}


def test_health_outage_both_ways(tmp_path):
    mod = _dashboard_module()
    _jl(tmp_path / "agent" / "decisions.jsonl",
        {"date": "2026-07-01", "symbol": "A", "status": "ok"},
        {"date": "2026-07-02", "symbol": "A", "status": "failed"})
    _jl(tmp_path / "agent_ctx" / "decisions.jsonl",
        {"date": "2026-07-02", "symbol": "A", "status": "failed"},
        {"date": "2026-07-02", "symbol": "B", "status": "ok"})
    assert _levels(mod._chk_outage(tmp_path)) == {"outage:agent": "bad", "outage:agent_ctx": "ok"}


def test_health_agent_open_failure_and_stuck(tmp_path):
    mod = _dashboard_module()
    eval_dir, data = tmp_path / "forward" / "agent_open", tmp_path / "data"
    _jl(eval_dir / "decisions.jsonl", {"date": "2026-06-20", "status": "ok"})
    (eval_dir / "positions.csv").write_text("date,symbol,pos,status\n2026-06-20,ZZ,1,ok\n")
    _returns(tmp_path / "forward", "agent_open", [])
    ok_open = {"date": "2026-07-05", "status": "ok", "fail": None}
    assert mod._chk_open(tmp_path / "x", None, data, NOW) == []
    items = mod._chk_open(tmp_path / "forward", ok_open, data, NOW)
    assert _levels(items) == {"open:decision": "ok", "open:stuck": "warn"}
    assert "2026-06-20" in items[1]["detail"] and "ZZ" in items[1]["detail"]
    _jl(tmp_path / "y" / "agent_open" / "decisions.jsonl", {"date": "2026-07-05", "status": "ok"})
    bad = mod._chk_open(tmp_path / "y", {"date": "2026-07-05", "status": "failed", "fail": "HTTP 500"}, data, NOW)
    assert bad[0]["level"] == "bad" and "HTTP 500" in bad[0]["detail"]


def test_health_memory_and_data_rules():
    mod = _dashboard_module()
    agent = {"end": "2026-07-10"}
    legacy = {"legacy": True, "date": "2026-07-09"}
    assert _levels(mod._chk_memory(legacy, agent)) == {"memory:legacy": "warn", "memory:fresh": "ok"}
    stale = {"legacy": False, "date": "2026-07-06"}                     # 4 days older
    assert _levels(mod._chk_memory(stale, agent)) == {"memory:fresh": "warn"}
    assert _levels(mod._chk_memory({"legacy": False, "date": "2026-07-07"}, agent)) == {"memory:fresh": "ok"}  # 3 days
    assert mod._chk_memory({"legacy": False, "date": None}, agent) == []
    assert _levels(mod._chk_data("2026-07-02", NOW)) == {"data:spy": "ok"}      # 4 days
    assert _levels(mod._chk_data("2026-07-01", NOW)) == {"data:spy": "warn"}    # 5 days
    assert _levels(mod._chk_data(None, NOW)) == {"data:spy": "warn"}


def _stub_cfg():
    return SimpleNamespace(strategy=SimpleNamespace(universe=["A"], name="mean_reversion", params={}, overlay="none"))


def test_new_sections_on_partial_state_and_level_is_worst(tmp_path):
    mod = _dashboard_module()
    journal = tmp_path / "journal"
    _spy(tmp_path / "data")
    _returns(journal / "forward", "mean_reversion_real", [0.01, 0.0, 0.0, 0.0, 0.0])
    out = mod._new_sections(journal / "papertrade", _stub_cfg(), tmp_path / "data", NOW)
    assert out["asOf"] == "2026-07-06"
    assert [r["id"] for r in out["records"]] == ["mean_reversion_real", "mean_reversion", "agent", "agent_ctx", "agent_open"]
    assert out["healthLevel"] == "bad"      # four records missing
    assert _levels(out["health"])["record:mean_reversion_real"] == "ok"
    assert out["today"]["date"] == "2026-07-06" and out["today"]["open"] is None


def test_corrupt_inputs_give_empty_values_and_health_not_exceptions(tmp_path):
    mod = _dashboard_module()
    journal, data = tmp_path / "journal", tmp_path / "data"
    fwd = journal / "forward"
    data.mkdir()
    (data / "SPY.csv").write_text("not,a\ncsv at all\n\x00\x01")
    for rid in ("mean_reversion_real", "agent", "agent_ctx", "agent_open"):
        (fwd / rid).mkdir(parents=True)
        (fwd / rid / "returns.csv").write_text("garbage\n\x00")
        (fwd / rid / "decisions.jsonl").write_text('{"date": "2026-07-01", "sy')   # truncated line
        (fwd / rid / "pos_A.csv").write_text("\x00\x00")
    (journal / "paper_orders.jsonl").write_text('{"event": "paper_ru')
    (journal / "paper_account.json").write_text("{")
    (journal / "agent_memory.md").write_bytes(b"\xff\xfe\x00")
    out = mod._new_sections(journal / "papertrade", _stub_cfg(), data, NOW)
    assert out["asOf"] is None
    assert all(r["missing"] for r in out["records"])
    assert out["today"]["open"] is None and out["today"]["agents"] == []
    assert out["today"]["rule"] is None and out["today"]["paper"] is None
    assert out["memory"] == {"date": None, "bullets": [], "legacy": False, "archiveEntries": 0}
    built = {h["id"] for h in out["health"] if h["id"].startswith("build:")}
    assert {"build:asOf", "build:record:agent", "build:today.open", "build:today.agents",
            "build:today.rule", "build:today.paper", "build:memory"} <= built
    assert out["healthLevel"] == "bad"


def test_corrupt_experiments_file_gives_empty_list_and_health(tmp_path, monkeypatch):
    mod = _dashboard_module()
    bad = tmp_path / "results.jsonl"
    bad.write_text('{"created": "2026-07-01", "experiment": "e", "var')
    monkeypatch.setattr(mod, "_EXPERIMENTS", bad)
    out = mod._new_sections(tmp_path / "journal" / "papertrade", _stub_cfg(), tmp_path / "data", NOW)
    assert out["experiments"] == []
    assert "build:experiments" in {h["id"] for h in out["health"]}


def test_build_data_keeps_every_old_key_and_adds_the_new_ones(tmp_path, monkeypatch):
    mod = _dashboard_module()
    shutil.copy(Path(__file__).resolve().parents[1] / "config.yaml", tmp_path / "config.yaml")
    monkeypatch.chdir(tmp_path)
    _write_run(Path("journal/papertrade/2026-07-12T00-00-00Z-aaaaaaaa"))
    _write_run(Path("journal/forward/mean_reversion"))
    data = mod._build_data(Path("journal/papertrade"))
    old = {"updated", "lockedEngine", "lockedOverlay", "guardrails", "thresholds", "forward", "bakeoff",
           "curveDaily", "curveDates", "runs", "winScore", "winTrades", "buckets", "winBuckets",
           "runbook", "reflections", "actionsUrl"}
    new = {"asOf", "records", "today", "memory", "experiments", "health", "healthLevel"}
    assert old <= set(data) and new <= set(data)
    json.dumps(data)   # the page embeds it with json.dumps
