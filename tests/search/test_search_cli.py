import pandas as pd
import pytest

from rhagent.search.__main__ import main


def _seed(cache_dir, symbol, closes):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_cli_runs_search(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    for k in range(6):
        closes = [100.0 + k + (0.5 * k + 1) * i for i in range(140)]
        _seed(cache, f"S{k}", closes)
    rc = main([
        "--strategy", "momentum", "--symbols", "S0,S1,S2,S3,S4,S5",
        "--rounds", "2", "--min-names", "3", "--days", "200",
        "--cache-dir", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "round 0" in out.lower()
    assert "configs tested" in out.lower()


def test_cli_unknown_strategy_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--strategy", "nope", "--symbols", "S0", "--cache-dir", str(tmp_path)])
