"""Daily paper-trading tick: the funnel PR #36 disconnected, wired back up.

Every weekday: mark the persisted paper account to market, check the kill
switch (HALT file / daily loss limit) against the *accumulated* account, turn
today's target positions (the same series ``forward._positions`` computes,
overlay included) into buy/sell orders, and run every one of them through
``OrderExecutor`` -- guardrails first, journal always, broker second.

DRY-RUN ONLY, unconditionally: the broker constructed here is always the
in-memory mock broker. This module has no import of the live broker class and
never reads the live-trading env var -- there is no branch in this file that
can place a real order, regardless of config or environment.

Usage:
    PYTHONPATH=src python -m rhagent.paper_run
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

from .broker import MockBroker
from .config import load
from .data import get_bars
from .executor import OrderExecutor
from .forward import _positions
from .guardrails import RunState, check_halted
from .journal import Journal
from .paper_account import AccountState
from .strategy_runner import orders_from_positions

HALT_FILE = Path("HALT")
STATE_PATH = Path("journal/paper_account.json")
JOURNAL_PATH = Path("journal/paper_orders.jsonl")


def run(
    cfg=None,
    *,
    today: date | None = None,
    cache_dir: str | Path = "data",
    state_path: str | Path = STATE_PATH,
    journal_path: str | Path = JOURNAL_PATH,
    halt_file: Path = HALT_FILE,
    fetch=None,
) -> dict:
    cfg = cfg or load()
    today = today or date.today()
    journal = Journal(journal_path)

    state = AccountState.load(state_path)
    state.roll_to_day(today)

    universe = list(cfg.strategy.universe)
    start = (today - timedelta(days=400)).isoformat()
    bars = get_bars(universe, start, today.isoformat(), fetch=fetch, cache_dir=cache_dir)
    prices = {s: float(df["close"].iloc[-1]) for s, df in bars.items() if len(df)}

    account = state.to_guardrail_account(prices)
    halted, reason = check_halted(account, cfg.limits, halt_file.exists())
    if halted:
        journal.record("paper_run_halted", reason=reason)
        state.save(state_path)
        return {"halted": True, "reason": reason, "orders": 0, "accepted": 0}

    eval_dir = Path("journal/forward") / cfg.strategy.name
    pos_by_symbol = _positions(cfg, cfg.strategy.name, bars, eval_dir)
    orders = orders_from_positions(
        pos_by_symbol,
        held=set(account.positions),
        notional_usd=cfg.limits.per_trade_max_usd,
        held_values=account.position_values,
    )

    # Always a MockBroker: nothing in this function can place a real order.
    broker = MockBroker(
        buying_power_usd=account.buying_power_usd,
        total_position_value_usd=account.total_position_value_usd,
        positions=dict(account.position_values),
        realized_pnl_today_usd=account.realized_pnl_today_usd,
        quotes=prices,
    )
    # dry_run=False is required here: the persistent paper account only
    # accumulates if fills actually come back, and dry_run short-circuits
    # before the broker. That makes MockBroker the ONLY thing standing between
    # this path and a real order, so assert it rather than trusting a comment.
    # If a future change parameterizes `broker`, this raises instead of
    # silently trading.
    if not isinstance(broker, MockBroker):
        raise RuntimeError(
            f"paper_run is dry-run-only, but broker is {type(broker).__name__}, "
            "not MockBroker. Refusing to run: dry_run=False with a live broker "
            "would place real orders."
        )
    executor = OrderExecutor(
        broker=broker,
        account=account,
        limits=cfg.limits,
        run_state=RunState(),
        journal=journal,
        dry_run=False,  # safe only because of the isinstance guard above
    )

    n_accepted = 0
    for symbol, side, notional in orders:
        result = executor.execute(symbol, side, notional)
        if result.accepted:
            n_accepted += 1
            state.apply_fill(symbol, side, notional, price=prices[symbol])

    state.save(state_path)
    journal.record("paper_run_end", n_orders=len(orders), n_accepted=n_accepted)
    return {"halted": False, "orders": len(orders), "accepted": n_accepted}


def main() -> int:
    result = run()
    print(f"[paper_run] {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
