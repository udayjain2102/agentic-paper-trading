import pandas as pd
import pytest

from rhagent.gate.__main__ import main


def _seed(cache_dir, symbol, closes):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_cli_runs_gate(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    for k in range(6):
        closes = [100.0 + k + (0.5 * k + 1) * i for i in range(160)]
        _seed(cache, f"S{k}", closes)
    rc = main([
        "--strategy", "momentum", "--symbols", "S0,S1,S2,S3,S4,S5",
        "--horizon", "1", "--min-names", "3", "--rounds", "1", "--days", "200",
        "--cache-dir", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "configs tested" in out.lower()
    assert "viable" in out.lower()


def test_cli_unknown_strategy_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--strategy", "nope", "--symbols", "S0", "--cache-dir", str(tmp_path)])
