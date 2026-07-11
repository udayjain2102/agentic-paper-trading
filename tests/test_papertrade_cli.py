import pandas as pd
import pytest

from rhagent.papertrade import main


def _seed_cache(cache_dir, symbol, closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_cli_runs_and_writes_ledger(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    # a wave so mean_reversion actually trades
    closes = [100 + 10 * ((i % 10) - 5) for i in range(80)]
    _seed_cache(cache, "AAPL", [float(c) for c in closes])

    rc = main([
        "--engine", "mean_reversion", "--symbols", "AAPL", "--days", "80",
        "--out-dir", str(tmp_path / "runs"), "--cache-dir", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_id" in out
    assert "win_rate" in out
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "trades.jsonl").exists()


def test_cli_compare_lists_runs(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    _seed_cache(cache, "AAPL", [100.0 + (i % 7) for i in range(60)])
    for _ in range(2):
        main(["--engine", "momentum", "--symbols", "AAPL", "--days", "60",
              "--out-dir", str(tmp_path / "runs"), "--cache-dir", str(cache)])
    capsys.readouterr()

    rc = main(["compare", "--out-dir", str(tmp_path / "runs")])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("momentum") == 2


def test_cli_unknown_engine_exits_with_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["--engine", "nope", "--symbols", "AAPL",
              "--out-dir", str(tmp_path / "runs")])
