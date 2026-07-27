"""forward._report_record_consistency cross-checks decisions.jsonl (the
append-only log) against pos_<SYM>.csv (the cache the code actually reads).
This is the check that would have caught #41's silent divergence -- see
forward._report_record_consistency's docstring.
"""

import json

import pandas as pd

from rhagent import forward


def _write_decisions(eval_dir, rows):
    with (eval_dir / "decisions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_pos(eval_dir, sym, rows):
    cols = ["date", "pos", "status"]
    pd.DataFrame(rows, columns=cols).to_csv(eval_dir / f"pos_{sym}.csv", index=False)


def _run(eval_dir, capsys):
    forward._report_record_consistency(eval_dir)
    return capsys.readouterr().out


def test_clean_record_reports_nothing(tmp_path, capsys):
    _write_decisions(tmp_path, [
        {"date": "2026-07-24", "symbol": "AAA", "target": 1.0, "status": "ok"},
    ])
    _write_pos(tmp_path, "AAA", [{"date": "2026-07-24", "pos": 1.0, "status": "ok"}])
    assert _run(tmp_path, capsys) == ""


def test_status_divergence_fires(tmp_path, capsys):
    # exactly the #41 bug: decisions.jsonl says ok, pos_*.csv says legacy
    _write_decisions(tmp_path, [
        {"date": "2026-07-22", "symbol": "LLY", "target": 1.0, "status": "ok"},
    ])
    _write_pos(tmp_path, "LLY", [{"date": "2026-07-22", "pos": 1.0, "status": "legacy"}])
    out = _run(tmp_path, capsys)
    assert "status differs" in out
    assert "LLY" in out


def test_missing_on_one_side_fires(tmp_path, capsys):
    _write_decisions(tmp_path, [
        {"date": "2026-07-24", "symbol": "AAA", "target": 1.0, "status": "ok"},
    ])
    _write_pos(tmp_path, "AAA", [])  # cache has no row at all for this date
    out = _run(tmp_path, capsys)
    assert "has no row" in out or "has a row but pos" in out


def test_target_pos_mismatch_fires(tmp_path, capsys):
    _write_decisions(tmp_path, [
        {"date": "2026-07-24", "symbol": "AAA", "target": 1.0, "status": "ok"},
    ])
    _write_pos(tmp_path, "AAA", [{"date": "2026-07-24", "pos": -1.0, "status": "ok"}])
    out = _run(tmp_path, capsys)
    assert "target/pos differ" in out


def test_retry_supersession_stays_quiet(tmp_path, capsys):
    # a failed row followed by a later ok row for the same (date, symbol) --
    # only the LATEST row is truth, and it matches the cache.
    _write_decisions(tmp_path, [
        {"date": "2026-07-24", "symbol": "AAA", "target": 0.0, "status": "failed"},
        {"date": "2026-07-24", "symbol": "AAA", "target": 1.0, "status": "ok"},
    ])
    _write_pos(tmp_path, "AAA", [{"date": "2026-07-24", "pos": 1.0, "status": "ok"}])
    assert _run(tmp_path, capsys) == ""


def test_seeded_legacy_bar_stays_quiet(tmp_path, capsys):
    # seeded anchor bar: no decisions.jsonl row at all, status=legacy in the
    # cache -- nobody decided it, that's correct, not a divergence.
    _write_decisions(tmp_path, [])
    _write_pos(tmp_path, "AAA", [{"date": "2025-06-02", "pos": 0.0, "status": "legacy"}])
    assert _run(tmp_path, capsys) == ""
