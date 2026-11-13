"""The experiment ledger must keep full return series and compare them fairly."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import ledger  # noqa: E402


def test_record_roundtrip_and_paired_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "results.jsonl")
    monkeypatch.setattr(ledger, "RUNS", tmp_path / "runs")
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    noise = pd.Series(np.random.default_rng(0).normal(0, 0.01, len(idx)), index=idx)

    row = ledger.record("demo", "base", noise, {"k": 1})
    assert ledger.load(row["run_id"]).round(12).equals(noise.round(12))
    assert list(ledger.table()["variant"]) == ["base"]

    # identical series: no difference. Same noise plus a steady edge: clearly better,
    # and the paired test sees it even though each Sharpe CI alone is wide.
    assert ledger.compare(noise, noise)["diff"] == 0
    better = ledger.compare(noise + 0.0005, noise)
    assert better["diff"] > 0 and better["ci_lo"] > 0
