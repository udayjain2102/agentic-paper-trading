import pandas as pd
import pytest

from rhagent.broker import MockBroker
from rhagent.config import StrategyConfig
from rhagent.executor import OrderExecutor
from rhagent.guardrails import Limits, RunState
from rhagent.journal import Journal
from rhagent import runner


def _limits():
    return Limits(
        per_trade_max_usd=250,
        total_deployed_max_usd=2000,
        max_new_positions_per_run=2,
        max_orders_per_run=5,
        daily_loss_limit_usd=200,
    )


class _Cfg:
    def __init__(self, strategy):
        self.strategy = strategy
        self.dry_run = True


def test_strategy_mode_dry_run_places_nothing(tmp_path):
    broker = MockBroker(quotes={"AAPL": 100.0})
    journal = Journal(tmp_path / "runs.jsonl")
    ex = OrderExecutor(
        broker=broker,
        account=broker.get_account(),
        limits=_limits(),
        run_state=RunState(),
        journal=journal,
        dry_run=True,
    )
    cfg = _Cfg(StrategyConfig(name="momentum", params={"lookback": 40}, universe=["AAPL"]))

    def fake_fetch(symbols, start, end):
        return {
            "AAPL": [
                {"date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                 "open": 0, "high": 0, "low": 0, "close": 100 + i, "volume": 0}
                for i in range(50)
            ]
        }

    summary = runner.run_strategy_mode(
        cfg, broker, ex, journal, fetch=fake_fetch
    )
    assert broker.placed == []  # dry-run: nothing reaches the broker
    assert "AAPL" in summary  # the buy was proposed and logged


def test_strategy_mode_pairs_dry_run_places_nothing(tmp_path):
    broker = MockBroker(quotes={"AAPL": 100.0, "MSFT": 100.0})
    journal = Journal(tmp_path / "runs.jsonl")
    ex = OrderExecutor(
        broker=broker,
        account=broker.get_account(),
        limits=_limits(),
        run_state=RunState(),
        journal=journal,
        dry_run=True,
    )
    cfg = _Cfg(
        StrategyConfig(name="pairs", params={}, universe=["AAPL", "MSFT"])
    )
    cfg.limits = _limits()

    def fake_fetch(symbols, start, end):
        out = {}
        for i, sym in enumerate(symbols):
            base = 100 - (10 if i == 0 else 0)
            out[sym] = [
                {"date": f"2025-{(j // 28) + 1:02d}-{(j % 28) + 1:02d}",
                 "open": 0, "high": 0, "low": 0,
                 "close": base if j == 49 else 100, "volume": 0}
                for j in range(50)
            ]
        return out

    summary = runner.run_strategy_mode(
        cfg, broker, ex, journal, fetch=fake_fetch
    )
    assert broker.placed == []  # dry-run: nothing reaches the broker
    assert isinstance(summary, str) and summary


def test_strategy_mode_pairs_wrong_universe_size_raises_systemexit(tmp_path):
    broker = MockBroker(quotes={"NVDA": 100.0})
    journal = Journal(tmp_path / "runs.jsonl")
    ex = OrderExecutor(
        broker=broker,
        account=broker.get_account(),
        limits=_limits(),
        run_state=RunState(),
        journal=journal,
        dry_run=True,
    )
    cfg = _Cfg(StrategyConfig(name="pairs", params={}, universe=["NVDA"]))
    cfg.limits = _limits()

    def fake_fetch(symbols, start, end):
        # The guard should fire before any fetch happens; data need not be valid.
        return {}

    with pytest.raises(SystemExit):
        runner.run_strategy_mode(cfg, broker, ex, journal, fetch=fake_fetch)
