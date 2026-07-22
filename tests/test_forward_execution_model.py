"""forward.tick must not hardcode cost_bps/fill: it takes them from
cfg.strategy (config.yaml's strategy.cost_bps/fill_mode), with an explicit
argument (--cost-bps/--fill-mode on the CLI) as an override. And the fill
mode threaded through must actually reach backtest.net_returns -- close vs
next_open must produce different return series through the forward path,
not just in backtest.py's own tests.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rhagent import forward


def _cfg(cost_bps=None, fill_mode=None):
    kw = {}
    if cost_bps is not None:
        kw["cost_bps"] = cost_bps
    if fill_mode is not None:
        kw["fill_mode"] = fill_mode
    return SimpleNamespace(strategy=SimpleNamespace(
        name="mean_reversion", params={}, universe=["AAA"], overlay="none", **kw))


def _write_cache(cache_dir: Path, closes, opens=None):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    df = pd.DataFrame({
        "open": opens if opens is not None else closes,
        "high": closes, "low": closes, "close": closes, "volume": 1e6,
    }, index=idx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_dir / "AAA.csv", index_label="date")


def test_tick_takes_cost_bps_and_fill_mode_from_cfg(tmp_path):
    from datetime import date

    cache = tmp_path / "cache"
    _write_cache(cache, [100.0] * 60)
    cfg = _cfg(cost_bps=7.0, fill_mode="next_open")

    res = forward.tick(cfg, tmp_path / "mr", today=date(2026, 3, 20), cache_dir=cache)

    assert res["meta"]["cost_bps"] == 7.0
    assert res["meta"]["fill_mode"] == "next_open"


def test_explicit_args_override_cfg(tmp_path):
    from datetime import date

    cache = tmp_path / "cache"
    _write_cache(cache, [100.0] * 60)
    cfg = _cfg(cost_bps=7.0, fill_mode="next_open")

    res = forward.tick(cfg, tmp_path / "mr", 1.0, fill="close",
                       today=date(2026, 3, 20), cache_dir=cache)

    assert res["meta"]["cost_bps"] == 1.0
    assert res["meta"]["fill_mode"] == "close"


def test_missing_cfg_fields_fall_back_to_historical_default(tmp_path):
    """A lightweight test cfg (no cost_bps/fill_mode attrs, like the other
    forward tests use) must still tick -- same 1.0/close default as before
    this fill was wired in."""
    from datetime import date

    cache = tmp_path / "cache"
    _write_cache(cache, [100.0] * 60)
    cfg = _cfg()  # no cost_bps/fill_mode set at all

    res = forward.tick(cfg, tmp_path / "mr", today=date(2026, 3, 20), cache_dir=cache)

    assert res["meta"]["cost_bps"] == 1.0
    assert res["meta"]["fill_mode"] == "close"


def test_close_vs_next_open_give_different_forward_returns():
    # Random-walk close, open offset from close by a constant gap so any day
    # the mean-reversion strategy changes position, close-fill and
    # next-open-fill price that change differently.
    idx = pd.date_range("2025-01-01", periods=150, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(
        np.random.default_rng(7).normal(0, 0.015, len(idx)))), index=idx)
    open_ = close - 1.0
    bars = {"AAA": pd.DataFrame({"open": open_, "close": close})}
    cfg = SimpleNamespace(strategy=SimpleNamespace(
        name="mean_reversion", params={}, universe=["AAA"], overlay="none"))

    net_close = forward._net_series(cfg, "mean_reversion", bars, 0.0, Path("/tmp"), fill="close")
    net_next_open = forward._net_series(cfg, "mean_reversion", bars, 0.0, Path("/tmp"), fill="next_open")

    assert not net_close.equals(net_next_open)
