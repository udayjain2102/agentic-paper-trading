import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

_DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_dashboard.py"


def _dashboard_module():
    spec = importlib.util.spec_from_file_location("make_dashboard", _DASHBOARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_js_function(name: str) -> str:
    """Pull one top-level `function <name>(...) { ... }` out of the embedded
    JS template by brace-matching, so the verdict logic can be exercised
    directly (in node) instead of only by inspecting Python-side data."""
    src = _DASHBOARD_PATH.read_text()
    m = re.search(rf"function {re.escape(name)}\(", src)
    assert m, f"{name} not found in {_DASHBOARD_PATH}"
    start = m.start()
    brace_start = src.index("{", m.end())
    depth = 0
    for i in range(brace_start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise AssertionError("unbalanced braces")


def _run_verdict_info(agent: dict, base: dict) -> dict:
    if not shutil.which("node"):
        pytest.skip("node not available")
    js = _extract_js_function("verdictInfo")
    script = f"""
{js}
console.log(JSON.stringify(verdictInfo({json.dumps(agent)}, {json.dumps(base)})));
"""
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


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


# ── A1/A2/A3: verdict correctness, empty state, frozen state ───────────────

def test_forward_leg_absent_renders_present_false_not_a_flat_number(tmp_path):
    """A2: a leg that never ran must be distinguishable from one that ran
    flat. `present: False` is the signal; the JS layer renders a dash for it,
    never money(0)."""
    mod = _dashboard_module()
    missing_dir = tmp_path / "journal" / "forward" / "agent"
    today = pd.Timestamp("2026-07-25")

    leg = mod._forward_leg(missing_dir, today)

    assert leg["present"] is False
    assert leg["days"] == 0
    assert leg["staleSessions"] is None


def test_missing_agent_leg_does_not_render_baseline_leads():
    """A1: the historical bug — an absent agent leg borrowed the baseline's
    day count via `agent.days || base.days`, satisfied the day gate, and
    rendered 'BASELINE LEADS' for a pipeline failure. It must not."""
    agent = {"present": False, "days": 0, "pnl": 0.0, "staleSessions": None, "end": ""}
    base = {"present": True, "days": 30, "pnl": 500.0}

    result = _run_verdict_info(agent, base)

    assert result["badge"] == "NO AGENT RECORD"
    assert result["badge"] != "BASELINE LEADS"


def test_frozen_agent_record_gets_frozen_badge_not_a_competitive_result():
    """A3: an agent leg that stopped ticking 5+ sessions ago must not be
    silently compared as if its last number were current."""
    agent = {"present": True, "days": 40, "pnl": 900.0, "staleSessions": 6, "end": "2026-07-01"}
    base = {"present": True, "days": 40, "pnl": 100.0}

    result = _run_verdict_info(agent, base)

    assert result["badge"] == "RECORD FROZEN"
    assert result["warn"] is True


def test_incomplete_record_and_too_early_gates_still_fire():
    agent_incomplete = {"present": True, "days": 5, "pnl": 0.0, "staleSessions": 0, "end": "2026-07-25"}
    base = {"present": True, "days": 30, "pnl": 100.0}
    assert _run_verdict_info(agent_incomplete, base)["badge"] == "RECORD INCOMPLETE"

    agent_early = {"present": True, "days": 18, "pnl": 0.0, "staleSessions": 0, "end": "2026-07-25"}
    base_early = {"present": True, "days": 18, "pnl": 100.0}
    assert _run_verdict_info(agent_early, base_early)["badge"] == "TOO EARLY TO CALL"


def test_stale_sessions_zero_on_the_tick_day_and_positive_after():
    mod = _dashboard_module()
    assert mod._stale_sessions("2026-07-24", pd.Timestamp("2026-07-24")) == 0
    # 2026-07-24 is a Friday; the next business day is Monday 2026-07-27
    assert mod._stale_sessions("2026-07-24", pd.Timestamp("2026-07-27")) == 1


def test_agent_leg_dir_prefers_agent_v2(tmp_path):
    mod = _dashboard_module()
    forward_dir = tmp_path / "journal" / "forward"
    (forward_dir / "agent").mkdir(parents=True)
    assert mod._agent_leg_dir(forward_dir) == forward_dir / "agent"

    (forward_dir / "agent-v2").mkdir(parents=True)
    assert mod._agent_leg_dir(forward_dir) == forward_dir / "agent-v2"
