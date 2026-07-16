import importlib.util
import json
import re
from pathlib import Path

import pandas as pd


def _dashboard_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_dashboard.py"
    spec = importlib.util.spec_from_file_location("make_dashboard", path)
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
    assert "Trading Dashboard" in html
    assert "Now · forward track record" in html
    assert "Research pulse" in html
    assert "href='#run-2026-07-12T00-00-00Z-aaaaaaaa'" in html
    assert "<details class=\"rundetail\" id=\"run-2026-07-12T00-00-00Z-aaaaaaaa\"" in html
    assert "Runbook" in html
    assert not re.search(r"\d\.\d{3,}×", html)  # equity labels stay at 2 decimals

    for legacy_dashboard in [
        tmp_path / "journal" / "papertrade" / "dashboard.html",
        tmp_path / "journal" / "forward" / "dashboard.html",
    ]:
        assert not legacy_dashboard.exists()
