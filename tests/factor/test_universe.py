import pandas as pd
import pytest

from rhagent.factor.universe import UNIVERSE, load_universe


def _seed(cache_dir, symbol, closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    ).to_csv(cache_dir / f"{symbol}.csv")


def test_universe_is_deduped_nonempty_no_etfs():
    assert len(UNIVERSE) >= 50
    assert len(UNIVERSE) == len(set(UNIVERSE))
    assert "SPY" not in UNIVERSE  # ETFs excluded from the cross-section


def test_load_universe_builds_aligned_close_panel(tmp_path):
    _seed(tmp_path, "AAA", [float(i) for i in range(1, 11)])
    _seed(tmp_path, "BBB", [float(i) * 2 for i in range(1, 11)])
    bars, close = load_universe(["AAA", "BBB"], "2026-01-01", "2026-01-10",
                                cache_dir=tmp_path, min_bars=5)
    assert set(bars) == {"AAA", "BBB"}
    assert list(close.columns) == ["AAA", "BBB"]
    assert len(close) == 10
    assert close["BBB"].iloc[-1] == 20.0


def test_load_universe_drops_short_history(tmp_path):
    _seed(tmp_path, "AAA", [float(i) for i in range(1, 11)])  # 10 bars
    _seed(tmp_path, "SHORT", [1.0, 2.0, 3.0])                 # 3 bars
    bars, close = load_universe(["AAA", "SHORT"], "2026-01-01", "2026-01-10",
                                cache_dir=tmp_path, min_bars=5)
    assert set(bars) == {"AAA"}
    assert list(close.columns) == ["AAA"]


def test_load_universe_inner_joins_on_common_dates(tmp_path):
    _seed(tmp_path, "AAA", [float(i) for i in range(1, 11)])   # days 1..10
    # BBB shifted to start later so only some dates overlap
    idx = pd.date_range("2026-01-04", periods=10, freq="D", name="date")
    pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [float(i) for i in range(1, 11)],
         "volume": 1e6},
        index=idx,
    ).to_csv(tmp_path / "BBB.csv")
    _, close = load_universe(["AAA", "BBB"], "2026-01-01", "2026-01-14",
                             cache_dir=tmp_path, min_bars=5)
    # inner join keeps only the overlapping dates (2026-01-04 .. 2026-01-10)
    assert close.index.min() == pd.Timestamp("2026-01-04")
    assert close.index.max() == pd.Timestamp("2026-01-10")
    assert not close.isna().any().any()


def test_load_universe_empty_raises(tmp_path):
    _seed(tmp_path, "SHORT", [1.0, 2.0])
    with pytest.raises(ValueError, match="min_bars"):
        load_universe(["SHORT"], "2026-01-01", "2026-01-02",
                      cache_dir=tmp_path, min_bars=5)
