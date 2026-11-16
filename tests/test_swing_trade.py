import pandas as pd
import pytest

from rhagent.guardrails import Account, Limits
from rhagent.swing_trade import latest_targets, plan_orders

LIMITS = Limits(per_trade_max_usd=250.0, total_deployed_max_usd=2000.0,
                max_new_positions_per_run=2, max_orders_per_run=5,
                daily_loss_limit_usd=200.0)


def acct(position_values=None, buying_power=10_000.0, deployed=0.0):
    pv = position_values or {}
    return Account(buying_power_usd=buying_power,
                   total_position_value_usd=deployed or sum(pv.values()),
                   positions=frozenset(pv),
                   realized_pnl_today_usd=0.0,
                   position_values=pv)


def write_pos(tmp_path, sym, rows):
    df = pd.DataFrame(rows, columns=["date", "pos", "status"])
    df.to_csv(tmp_path / f"pos_{sym}.csv", index=False)


def test_latest_targets_reads_last_ok_row(tmp_path):
    write_pos(tmp_path, "AAPL", [
        ("2026-11-10", 0.0, "ok"),
        ("2026-11-11", 1.0, "ok"),
        ("2026-11-12", 0.0, "failed"),   # failed decision must not leak in
    ])
    write_pos(tmp_path, "MSFT", [("2026-11-11", 1.0, "ok")])
    targets, newest = latest_targets(tmp_path)
    assert targets == {"AAPL": 1.0, "MSFT": 1.0}
    assert newest == pd.Timestamp("2026-11-11")


def test_plan_orders_buys_equal_weight(tmp_path):
    # 4 names wanted long, $2000 cap -> $500/name but per-trade cap $250 clips.
    targets = {s: 1.0 for s in ("AAA", "BBB", "CCC", "DDD")}
    orders = plan_orders(targets, acct(), LIMITS)
    assert len(orders) == 4
    for sym, side, notional in orders:
        assert side == "buy"
        assert notional == pytest.approx(250.0)  # per-trade cap clips $500


def test_plan_orders_sells_what_record_dropped(tmp_path):
    targets = {"AAA": 1.0}
    account = acct({"AAA": 100.0, "ZZZ": 300.0})
    orders = dict(((s, (side, n)) for s, side, n in plan_orders(targets, account, LIMITS)))
    # AAA target $2000/1 = capped per-trade... check deltas instead:
    # held ZZZ not in targets -> sell full $300. AAA: target clipped buy.
    assert orders["ZZZ"][0] == "sell" and orders["ZZZ"][1] == pytest.approx(300.0)


def test_plan_orders_skips_small_deltas(tmp_path):
    # 4 names -> per_name = 2000/4 = 500. AAA already held at 496 -> delta 4
    # < min_trade 5 -> no order for AAA; the other three still buy.
    targets = {s: 1.0 for s in ("AAA", "BBB", "CCC", "DDD")}
    account = acct({"AAA": 496.0})
    orders = plan_orders(targets, account, LIMITS)
    assert not any(s == "AAA" for s, _, _ in orders)
    assert len(orders) == 3  # BBB/CCC/DDD buys


def test_plan_orders_never_exceeds_held_on_sell(tmp_path):
    targets = {"AAA": 0.0}
    account = acct({"AAA": 40.0})
    orders = plan_orders(targets, account, LIMITS)
    assert orders == [("AAA", "sell", pytest.approx(40.0))]


def test_plan_orders_short_target_means_flat(tmp_path):
    targets = {"AAA": -1.0}
    account = acct({"AAA": 100.0})
    orders = plan_orders(targets, account, LIMITS)
    assert orders == [("AAA", "sell", pytest.approx(100.0))]
    # and with nothing held, a -1 target produces no buy (no shorting)
    assert plan_orders(targets, acct(), LIMITS) == []
