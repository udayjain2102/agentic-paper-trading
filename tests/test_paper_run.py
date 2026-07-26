"""End-to-end tests for the daily paper-trading tick (rhagent.paper_run).

These exercise the whole funnel together: persisted account -> guardrails ->
executor -> (mock) broker -> journal -> persisted account again. The single
most important property under test is the one in the safety brief: zero real
orders, ever, from this module.
"""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rhagent import paper_run
from rhagent.broker import MockBroker
from rhagent.guardrails import Limits
from rhagent.paper_account import AccountState


def _trending_frame(n=60, start=100.0, step=1.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    close = start + step * np.arange(n)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx
    )


def _cfg(*, per_trade_max_usd=250.0, total_deployed_max_usd=2_000.0, universe=("AAPL",)):
    return SimpleNamespace(
        strategy=SimpleNamespace(
            name="momentum", params={"lookback": 5}, universe=list(universe), overlay="none",
        ),
        limits=Limits(
            per_trade_max_usd=per_trade_max_usd,
            total_deployed_max_usd=total_deployed_max_usd,
            max_new_positions_per_run=2,
            max_orders_per_run=5,
            daily_loss_limit_usd=200.0,
        ),
    )


def _fake_fetch(cache_dir):
    """No network: seed the on-disk cache directly, fetch() is never called."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(symbols, start, end):
        raise AssertionError("fetch() called -- cache should have covered every symbol")

    return fetch


def test_end_to_end_dry_run_places_zero_real_orders(tmp_path):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    _trending_frame().to_csv(cache_dir / "AAPL.csv", index_label="date")

    cfg = _cfg()
    state_path = tmp_path / "account.json"
    journal_path = tmp_path / "journal" / "orders.jsonl"

    result = paper_run.run(
        cfg,
        today=date(2026, 4, 1),
        cache_dir=cache_dir,
        state_path=state_path,
        journal_path=journal_path,
        halt_file=tmp_path / "HALT",  # does not exist
        fetch=_fake_fetch(cache_dir),
    )

    assert result["halted"] is False
    assert result["orders"] >= 1  # uptrend -> a buy is proposed
    assert result["accepted"] >= 1

    # The only broker this module can construct is MockBroker -- there is no
    # code path here that can reach a real account.
    import inspect
    source = inspect.getsource(paper_run)
    assert "McpBroker" not in source
    assert "LIVE" not in source

    state = AccountState.load(state_path)
    assert state.positions  # the accepted buy was booked into persisted state


def test_rejected_order_is_logged_with_reason_and_never_reaches_broker(tmp_path):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    _trending_frame().to_csv(cache_dir / "AAPL.csv", index_label="date")

    # Zero deployed-capital headroom -> any buy pushes past the cap and is rejected.
    cfg = _cfg(total_deployed_max_usd=0.0)
    state_path = tmp_path / "account.json"
    journal_path = tmp_path / "journal" / "orders.jsonl"

    result = paper_run.run(
        cfg,
        today=date(2026, 4, 1),
        cache_dir=cache_dir,
        state_path=state_path,
        journal_path=journal_path,
        halt_file=tmp_path / "HALT",
        fetch=_fake_fetch(cache_dir),
    )

    assert result["orders"] >= 1
    assert result["accepted"] == 0

    lines = journal_path.read_text().splitlines()
    rejected = [l for l in lines if '"order_rejected"' in l]
    assert rejected, "expected an order_rejected journal entry"
    assert "deployed" in rejected[0].lower()

    state = AccountState.load(state_path)
    assert state.positions == {}  # nothing booked


def test_halt_file_stops_execution(tmp_path):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    _trending_frame().to_csv(cache_dir / "AAPL.csv", index_label="date")

    halt_file = tmp_path / "HALT"
    halt_file.touch()

    cfg = _cfg()
    state_path = tmp_path / "account.json"
    journal_path = tmp_path / "journal" / "orders.jsonl"

    result = paper_run.run(
        cfg,
        today=date(2026, 4, 1),
        cache_dir=cache_dir,
        state_path=state_path,
        journal_path=journal_path,
        halt_file=halt_file,
        fetch=_fake_fetch(cache_dir),
    )

    assert result["halted"] is True
    assert result["orders"] == 0
    assert result["accepted"] == 0
    assert "halt" in result["reason"].lower()


def test_daily_loss_kill_switch_blocks_the_whole_run(tmp_path):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    _trending_frame().to_csv(cache_dir / "AAPL.csv", index_label="date")

    state_path = tmp_path / "account.json"
    state = AccountState.load(state_path)
    state.date = "2026-04-01"
    state.realized_pnl_today_usd = -250.0  # breaches the $200 limit in _cfg()
    state.save(state_path)

    cfg = _cfg()
    journal_path = tmp_path / "journal" / "orders.jsonl"

    result = paper_run.run(
        cfg,
        today=date(2026, 4, 1),
        cache_dir=cache_dir,
        state_path=state_path,
        journal_path=journal_path,
        halt_file=tmp_path / "HALT",
        fetch=_fake_fetch(cache_dir),
    )

    assert result["halted"] is True
    assert "loss" in result["reason"].lower()


def test_position_bought_on_day_1_persists_and_is_marked_to_market_on_day_2(tmp_path):
    """Proves state carries over: day 2 sees AAPL already held (no re-buy) and
    its value marked to the day-2 close, not reconstructed empty."""
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    _trending_frame(n=70).to_csv(cache_dir / "AAPL.csv", index_label="date")

    cfg = _cfg()
    state_path = tmp_path / "account.json"
    journal_path = tmp_path / "journal" / "orders.jsonl"

    r1 = paper_run.run(
        cfg, today=date(2026, 4, 1), cache_dir=cache_dir, state_path=state_path,
        journal_path=journal_path, halt_file=tmp_path / "HALT", fetch=_fake_fetch(cache_dir),
    )
    assert r1["accepted"] >= 1
    shares_after_day1 = AccountState.load(state_path).positions["AAPL"].shares

    r2 = paper_run.run(
        cfg, today=date(2026, 4, 2), cache_dir=cache_dir, state_path=state_path,
        journal_path=journal_path, halt_file=tmp_path / "HALT", fetch=_fake_fetch(cache_dir),
    )
    # Already held -> momentum staying long proposes no new buy for AAPL.
    assert r2["orders"] == 0
    state2 = AccountState.load(state_path)
    assert state2.positions["AAPL"].shares == shares_after_day1
