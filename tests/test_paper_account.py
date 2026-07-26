from datetime import date

from rhagent.paper_account import AccountState, DEFAULT_STARTING_CASH_USD


def test_no_state_file_gives_fresh_default_account(tmp_path):
    s = AccountState.load(tmp_path / "account.json")
    assert s.cash_usd == DEFAULT_STARTING_CASH_USD
    assert s.positions == {}
    assert s.realized_pnl_today_usd == 0.0


def test_positions_and_cash_round_trip_through_save_and_load(tmp_path):
    path = tmp_path / "account.json"
    s = AccountState.load(path)
    s.apply_fill("AAPL", "buy", 250.0, price=100.0)
    s.save(path)

    reloaded = AccountState.load(path)
    assert reloaded.cash_usd == DEFAULT_STARTING_CASH_USD - 250.0
    assert reloaded.positions["AAPL"].shares == 2.5
    assert reloaded.positions["AAPL"].cost_basis_usd == 250.0


def test_total_deployed_is_enforced_against_accumulated_value_not_a_fresh_account(tmp_path):
    """The test that proves the state layer does its job: a position bought
    on a prior run (persisted, then reloaded) counts toward
    total_position_value_usd on the next run -- a fresh MockBroker account
    would report this as 0 and let a cap-busting order through."""
    path = tmp_path / "account.json"
    s = AccountState.load(path)
    s.apply_fill("AAPL", "buy", 1_900.0, price=100.0)
    s.save(path)

    reloaded = AccountState.load(path)
    account = reloaded.to_guardrail_account(prices={"AAPL": 100.0})
    assert account.total_position_value_usd == 1_900.0

    from rhagent.guardrails import Limits, RunState, Order, validate_order

    limits = Limits(
        per_trade_max_usd=250, total_deployed_max_usd=2_000,
        max_new_positions_per_run=2, max_orders_per_run=5, daily_loss_limit_usd=200,
    )
    ok, reason = validate_order(
        Order(symbol="MSFT", side="buy", notional_usd=200.0), account, RunState(), limits,
    )
    assert ok is False
    assert "deployed" in reason.lower()


def test_mark_to_market_uses_current_price_not_cost_basis(tmp_path):
    s = AccountState.load(tmp_path / "account.json")
    s.apply_fill("AAPL", "buy", 250.0, price=100.0)  # 2.5 shares
    account = s.to_guardrail_account(prices={"AAPL": 120.0})
    assert account.total_position_value_usd == 300.0  # 2.5 * 120
    assert account.position_values == {"AAPL": 300.0}


def test_missing_quote_skips_symbol_instead_of_pricing_at_zero(tmp_path):
    s = AccountState.load(tmp_path / "account.json")
    s.apply_fill("AAPL", "buy", 250.0, price=100.0)
    account = s.to_guardrail_account(prices={})  # no quote for AAPL
    assert account.position_values == {}
    assert account.total_position_value_usd == 0.0


def test_sell_realizes_pnl_against_cost_basis(tmp_path):
    s = AccountState.load(tmp_path / "account.json")
    s.apply_fill("AAPL", "buy", 250.0, price=100.0)
    s.apply_fill("AAPL", "sell", 300.0, price=120.0)  # marked value at exit
    assert s.realized_pnl_today_usd == 50.0
    assert "AAPL" not in s.positions
    assert s.cash_usd == DEFAULT_STARTING_CASH_USD + 50.0


def test_realized_pnl_resets_on_a_new_trading_day(tmp_path):
    path = tmp_path / "account.json"
    s = AccountState.load(path, starting_cash_usd=5_000)
    s.date = str(date(2026, 1, 1))
    s.apply_fill("AAPL", "buy", 250.0, price=100.0)
    s.apply_fill("AAPL", "sell", 50.0, price=20.0)  # a big loss today
    assert s.realized_pnl_today_usd == -200.0
    s.save(path)

    reloaded = AccountState.load(path)
    reloaded.roll_to_day(date(2026, 1, 2))
    assert reloaded.realized_pnl_today_usd == 0.0
    # cash already booked (buy -250, sell +50) is untouched by the day roll.
    assert reloaded.cash_usd == 5_000 - 250.0 + 50.0


def test_realized_pnl_does_not_reset_within_the_same_day(tmp_path):
    s = AccountState.load(tmp_path / "account.json")
    s.date = str(date(2026, 1, 1))
    s.realized_pnl_today_usd = -100.0
    s.roll_to_day(date(2026, 1, 1))
    assert s.realized_pnl_today_usd == -100.0


def test_daily_loss_kill_switch_fires_and_stays_fired_for_the_rest_of_the_day(tmp_path):
    from rhagent.guardrails import Limits, check_halted

    s = AccountState.load(tmp_path / "account.json")
    s.date = str(date(2026, 1, 1))
    s.realized_pnl_today_usd = -250.0  # breach a $200 daily loss limit
    limits = Limits(
        per_trade_max_usd=250, total_deployed_max_usd=2_000,
        max_new_positions_per_run=2, max_orders_per_run=5, daily_loss_limit_usd=200,
    )
    account = s.to_guardrail_account(prices={})

    halted, reason = check_halted(account, limits, halt_file_present=False)
    assert halted is True
    assert "loss" in reason.lower()

    # Still the same day, a second run later -> still halted.
    s.roll_to_day(date(2026, 1, 1))
    account2 = s.to_guardrail_account(prices={})
    halted2, _ = check_halted(account2, limits, halt_file_present=False)
    assert halted2 is True
