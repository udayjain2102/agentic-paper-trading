import pandas as pd
import pytest

from rhagent.factor.__main__ import main


def _seed(cache_dir, symbol, closes):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_cli_reports_icir(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    # 5 symbols, 120 bars each, distinct trends so signals vary across names
    for k in range(5):
        closes = [100.0 + k + (0.5 * k + 1) * i for i in range(120)]
        _seed(cache, f"S{k}", closes)

    rc = main([
        "--strategy", "momentum",
        "--symbols", "S0,S1,S2,S3,S4",
        "--horizon", "5", "--min-names", "3", "--days", "200",
        "--cache-dir", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ICIR" in out
    assert "decay" in out.lower()


def test_cli_unknown_strategy_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--strategy", "nope", "--symbols", "S0", "--cache-dir", str(tmp_path)])


def test_cli_empty_ic_series_reports_insufficient_data(tmp_path, capsys):
    cache = tmp_path / "data"
    cache.mkdir()
    # Only 2 symbols; default min_names=10 means every day has < 10 valid
    # names, so ic_series() is empty and the CLI must say so plainly instead
    # of printing a misleading "ICIR ... [likely noise]" band.
    for k in range(2):
        closes = [100.0 + k + (0.5 * k + 1) * i for i in range(120)]
        _seed(cache, f"S{k}", closes)

    rc = main([
        "--strategy", "momentum",
        "--symbols", "S0,S1",
        "--horizon", "5", "--days", "200",
        "--cache-dir", str(cache),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "insufficient data" in out.lower()
