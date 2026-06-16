"""Exhaustive tests for the safety-critical guardrail layer.

These are the most important tests in the project: the guardrails are the only
thing standing between a hallucinating model and real money. Every rejection
path is covered.
"""

import pytest

from rhagent.guardrails import (
    Account,
    Limits,
    Order,
    RunState,
    check_halted,
    validate_order,
)


def limits(**overrides):
    base = dict(
        per_trade_max_usd=250,
        total_deployed_max_usd=2000,
        max_new_positions_per_run=2,
        max_orders_per_run=5,
        daily_loss_limit_usd=200,
    )
    base.update(overrides)
    return Limits(**base)


def account(**overrides):
    base = dict(
        buying_power_usd=10_000,
        total_position_value_usd=0,
        positions=frozenset(),
        realized_pnl_today_usd=0,
    )
    base.update(overrides)
    return Account(**base)


# --- happy paths -----------------------------------------------------------

def test_valid_buy_passes():
    ok, reason = validate_order(
        Order(symbol="AAPL", side="buy", notional_usd=100),
        account(),
        RunState(),
        limits(),
    )
    assert ok is True
    assert reason is None


def test_valid_sell_passes_even_above_deployed_cap():
    # Selling reduces exposure; deployed/buying-power caps don't apply.
    ok, reason = validate_order(
        Order(symbol="AAPL", side="sell", notional_usd=100),
        account(total_position_value_usd=1_999, positions=frozenset({"AAPL"})),
        RunState(),
        limits(),
    )
    assert ok is True, reason


# --- input sanity ----------------------------------------------------------

@pytest.mark.parametrize("symbol", ["", "  ", "BTC-USD", "aapl123", "TOOLONGSYM", "A.B"])
def test_rejects_non_equity_symbols(symbol):
    ok, reason = validate_order(
        Order(symbol=symbol, side="buy", notional_usd=100),
        account(),
        RunState(),
        limits(),
    )
    assert ok is False
    assert "symbol" in reason.lower()


def test_rejects_unknown_side():
    ok, reason = validate_order(
        Order(symbol="AAPL", side="hold", notional_usd=100),
        account(),
        RunState(),
        limits(),
    )
    assert ok is False
    assert "side" in reason.lower()


@pytest.mark.parametrize("notional", [0, -1, -100.0])
def test_rejects_non_positive_notional(notional):
    ok, reason = validate_order(
        Order(symbol="AAPL", side="buy", notional_usd=notional),
        account(),
        RunState(),
        limits(),
    )
    assert ok is False
    assert "notional" in reason.lower()


# --- dollar caps -----------------------------------------------------------

def test_rejects_order_over_per_trade_cap():
    ok, reason = validate_order(
        Order(symbol="AAPL", side="buy", notional_usd=251),
        account(),
        RunState(),
        limits(per_trade_max_usd=250),
    )
    assert ok is False
    assert "per-trade" in reason.lower()


def test_rejects_buy_exceeding_buying_power():
    ok, reason = validate_order(
        Order(symbol="AAPL", side="buy", notional_usd=200),
        account(buying_power_usd=150),
        RunState(),
        limits(),
    )
    assert ok is False
    assert "buying power" in reason.lower()


def test_rejects_buy_exceeding_total_deployed_cap():
    ok, reason = validate_order(
        Order(symbol="AAPL", side="buy", notional_usd=200),
        account(total_position_value_usd=1_900),
        RunState(),
        limits(total_deployed_max_usd=2000),
    )
    assert ok is False
    assert "deployed" in reason.lower()


# --- per-run rate limits ---------------------------------------------------

def test_rejects_when_max_orders_per_run_reached():
    ok, reason = validate_order(
        Order(symbol="AAPL", side="buy", notional_usd=100),
        account(),
        RunState(orders_placed=5),
        limits(max_orders_per_run=5),
    )
    assert ok is False
    assert "orders per run" in reason.lower()


def test_rejects_new_position_when_max_new_positions_reached():
    ok, reason = validate_order(
        Order(symbol="NVDA", side="buy", notional_usd=100),
        account(positions=frozenset({"AAPL", "MSFT"})),
        RunState(new_positions=2),
        limits(max_new_positions_per_run=2),
    )
    assert ok is False
    assert "new positions" in reason.lower()


def test_adding_to_existing_position_not_limited_by_new_position_cap():
    ok, reason = validate_order(
        Order(symbol="AAPL", side="buy", notional_usd=100),
        account(positions=frozenset({"AAPL"})),
        RunState(new_positions=2),  # at the new-position cap...
        limits(max_new_positions_per_run=2),
    )
    assert ok is True, reason  # ...but AAPL is already held, so it's allowed


# --- kill switch / halt ----------------------------------------------------

def test_halted_when_daily_loss_limit_breached():
    halted, reason = check_halted(
        account(realized_pnl_today_usd=-200),
        limits(daily_loss_limit_usd=200),
        halt_file_present=False,
    )
    assert halted is True
    assert "loss" in reason.lower()


def test_not_halted_within_daily_loss_limit():
    halted, reason = check_halted(
        account(realized_pnl_today_usd=-199.99),
        limits(daily_loss_limit_usd=200),
        halt_file_present=False,
    )
    assert halted is False
    assert reason is None


def test_halted_when_halt_file_present():
    halted, reason = check_halted(
        account(realized_pnl_today_usd=0),
        limits(),
        halt_file_present=True,
    )
    assert halted is True
    assert "halt" in reason.lower()
