"""Tests for the order execution funnel — the dry-run guarantee especially."""

from rhagent.broker import MockBroker
from rhagent.executor import OrderExecutor
from rhagent.guardrails import Limits, RunState
from rhagent.journal import Journal


def limits():
    return Limits(
        per_trade_max_usd=250,
        total_deployed_max_usd=2000,
        max_new_positions_per_run=2,
        max_orders_per_run=5,
        daily_loss_limit_usd=200,
    )


def make_executor(broker, journal, dry_run):
    return OrderExecutor(
        broker=broker,
        account=broker.get_account(),
        limits=limits(),
        run_state=RunState(),
        journal=journal,
        dry_run=dry_run,
    )


def test_dry_run_places_nothing(tmp_path):
    broker = MockBroker()
    journal = Journal(tmp_path / "runs.jsonl")
    ex = make_executor(broker, journal, dry_run=True)

    result = ex.execute("AAPL", "buy", 100)

    assert result.accepted is True
    assert result.placed is False
    assert broker.placed == []  # the critical assertion: no order reached the broker


def test_live_places_order(tmp_path):
    broker = MockBroker()
    journal = Journal(tmp_path / "runs.jsonl")
    ex = make_executor(broker, journal, dry_run=False)

    result = ex.execute("AAPL", "buy", 100)

    assert result.placed is True
    assert len(broker.placed) == 1
    assert broker.placed[0].symbol == "AAPL"


def test_rejected_order_never_reaches_broker_even_when_live(tmp_path):
    broker = MockBroker()
    journal = Journal(tmp_path / "runs.jsonl")
    ex = make_executor(broker, journal, dry_run=False)

    result = ex.execute("AAPL", "buy", 999)  # over per-trade cap

    assert result.accepted is False
    assert broker.placed == []


def test_counters_advance_and_enforce_run_limits(tmp_path):
    broker = MockBroker()
    journal = Journal(tmp_path / "runs.jsonl")
    run = RunState()
    ex = OrderExecutor(
        broker=broker,
        account=broker.get_account(),
        limits=Limits(
            per_trade_max_usd=250,
            total_deployed_max_usd=10_000,
            max_new_positions_per_run=2,
            max_orders_per_run=5,
            daily_loss_limit_usd=200,
        ),
        run_state=run,
        journal=journal,
        dry_run=True,
    )

    assert ex.execute("AAPL", "buy", 100).accepted
    assert ex.execute("MSFT", "buy", 100).accepted
    # Third distinct new position should be rejected by the new-position cap.
    third = ex.execute("NVDA", "buy", 100)
    assert third.accepted is False
    assert "new positions" in third.reason.lower()
