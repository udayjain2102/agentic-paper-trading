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
