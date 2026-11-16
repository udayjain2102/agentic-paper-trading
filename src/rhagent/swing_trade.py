"""Swing-trade bridge: forward-record targets -> real orders.

Reads the latest decided positions from a forward eval dir
(``pos_<SYM>.csv``, the same freeze-to-disk cache ``_agent_positions`` and
``_model_positions`` maintain), diffs them against the broker's actual
positions, and routes every change through ``OrderExecutor`` — so the
guardrails in config.yaml (per-trade cap, deployed cap, order rate, daily
loss kill-switch) apply to real money exactly as they do on paper.

Swing format: positions are held until the target changes, not rebalanced
daily. This module only trades the DELTA — a name whose target didn't move
produces no order, which is what keeps turnover (and PDT exposure) low.

Long-only v1: a target of -1 is treated as flat (sell any held shares, never
open a short). The mean_reversion record is effectively long-only anyway.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from .config import load
from .executor import OrderExecutor
from .guardrails import Account, Limits, RunState
from .journal import Journal


def latest_targets(eval_dir: Path) -> tuple[dict[str, float], pd.Timestamp | None]:
    """Latest decided target per symbol across pos_*.csv files.

    Returns (targets, newest_decision_date). Rows with status != "ok" are
    skipped — a failed/legacy decision is not a position we should trade.
    """
    targets: dict[str, float] = {}
    newest = None
    for f in sorted(eval_dir.glob("pos_*.csv")):
        sym = f.stem[4:]
        df = pd.read_csv(f)
        if "status" in df.columns:
            ok = df[df["status"] == "ok"]
            if ok.empty:
                continue
            row = ok.iloc[-1]
        else:
            row = df.iloc[-1]
        targets[sym] = float(row["pos"])
        d = pd.Timestamp(row["date"])
        newest = d if newest is None or d > newest else newest
    return targets, newest


def strategy_targets(cfg, cache_dir: str = "data",
                     today=None) -> tuple[dict[str, float], object]:
    """Latest positions for rule-based records (mean_reversion etc.).

    These records don't freeze pos_*.csv — forward.py recomputes positions
    live from cached bars via Strategy.positions + the conviction overlay.
    We mirror that exactly (same get_bars/build/apply_conviction calls) and
    take each symbol's position on the newest shared bar date.
    """
    from datetime import date, timedelta

    import pandas as pd

    from .data import get_bars
    from .strategies import build

    today = today or date.today()
    start = (today - timedelta(days=400)).isoformat()
    bars = get_bars(cfg.strategy.universe, start, today.isoformat(),
                    cache_dir=cache_dir)
    strat = build(cfg.strategy.name, cfg.strategy.params)
    pos = {s: strat.positions(bars[s]) for s in cfg.strategy.universe
           if s in bars}
    if getattr(cfg.strategy, "overlay", "none") == "conviction":
        from .overlay import apply_conviction
        pos = {s: apply_conviction(pos[s], strat.signal(bars[s]))
               for s in pos}
    elif getattr(cfg.strategy, "overlay", "none") not in ("none", ""):
        raise SystemExit(
            f"overlay {cfg.strategy.overlay!r} not supported by swing_trade "
            "(only 'conviction' is wired into the forward path)")
    last_date = max(p.index.max() for p in pos.values())
    return ({s: float(p.asof(last_date)) for s, p in pos.items()},
            pd.Timestamp(last_date))


def plan_orders(
    targets: dict[str, float],
    account: Account,
    limits: Limits,
    *,
    min_trade_usd: float = 5.0,
) -> list[tuple[str, str, float]]:
    """Diff targets vs actual positions -> [(symbol, side, notional_usd)].

    Budget: deploy up to limits.total_deployed_max_usd equally across the
    names the record wants long. Buys are clipped to per_trade_max_usd and
    available buying power; sells never exceed the held position value.
    Names held at the broker but absent from (or flat in) the targets are
    liquidated — the record saying "flat" is an order to exit.
    """
    held_names = [s for s, t in targets.items() if t > 0]
    budget = min(limits.total_deployed_max_usd, account.buying_power_usd)
    per_name = budget / len(held_names) if held_names else 0.0

    orders: list[tuple[str, str, float]] = []
    wanted: dict[str, float] = {}
    for s, t in targets.items():
        wanted[s] = per_name if t > 0 else 0.0
    # Anything the broker holds that the record doesn't want at all.
    for s, val in account.position_values.items():
        if s not in wanted:
            wanted[s] = 0.0

    for s in sorted(set(wanted) | set(account.position_values)):
        target_val = wanted.get(s, 0.0)
        cur_val = account.position_values.get(s, 0.0)
        delta = target_val - cur_val
        if delta >= min_trade_usd:
            notional = min(delta, limits.per_trade_max_usd,
                           account.buying_power_usd)
            if notional >= min_trade_usd:
                orders.append((s, "buy", notional))
        elif delta <= -min_trade_usd:
            notional = min(-delta, cur_val)
            if notional >= min_trade_usd:
                orders.append((s, "sell", notional))
    return orders


def _build_broker(dry_run: bool):
    """MockBroker for dry-run; McpBroker over mcp_session when live."""
    if dry_run:
        from .broker import MockBroker
        return MockBroker()
    url = os.environ.get("ROBINHOOD_MCP_URL", "")
    token = os.environ.get("ROBINHOOD_MCP_TOKEN", "")
    if not (url and token):
        raise SystemExit(
            "live mode needs ROBINHOOD_MCP_URL and ROBINHOOD_MCP_TOKEN "
            "(OAuth bearer from an authenticated MCP session)")
    from .broker import McpBroker
    from .mcp_session import mcp_session
    session_cm = mcp_session(url, token)
    session = session_cm.__enter__()
    return McpBroker(session), session_cm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rhagent.swing_trade")
    p.add_argument("--eval-id", default="mean_reversion_real",
                   help="forward record dir under --out-dir")
    p.add_argument("--out-dir", default="journal/forward")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--live", action="store_true",
                   help="place real orders via the Robinhood MCP")
    p.add_argument("--journal", default="journal/swing_trades.jsonl")
    p.add_argument("--min-trade-usd", type=float, default=5.0)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    dry_run = not args.live

    cfg = load()
    limits = Limits(
        per_trade_max_usd=cfg.limits.per_trade_max_usd,
        total_deployed_max_usd=cfg.limits.total_deployed_max_usd,
        max_new_positions_per_run=cfg.limits.max_new_positions_per_run,
        max_orders_per_run=cfg.limits.max_orders_per_run,
        daily_loss_limit_usd=cfg.limits.daily_loss_limit_usd,
    )

    eval_dir = Path(args.out_dir) / args.eval_id
    if list(eval_dir.glob("pos_*.csv")):
        targets, newest = latest_targets(eval_dir)
        src = "pos_*.csv"
    else:
        targets, newest = strategy_targets(cfg)
        src = "strategy recompute"
    if not targets:
        raise SystemExit(f"no positions resolvable under {eval_dir}")
    long_n = sum(1 for t in targets.values() if t > 0)
    print(f"record {args.eval_id} ({src}): {len(targets)} symbols, "
          f"{long_n} long, newest decision {newest.date() if newest is not None else '?'}")

    built = _build_broker(dry_run)
    if isinstance(built, tuple):
        broker, session_cm = built
    else:
        broker, session_cm = built, None
    try:
        account = broker.get_account()
        journal = Journal(args.journal)
        executor = OrderExecutor(broker=broker, account=account, limits=limits,
                                 run_state=RunState(), journal=journal,
                                 dry_run=dry_run)
        orders = plan_orders(targets, account, limits,
                             min_trade_usd=args.min_trade_usd)
        print(f"{len(orders)} order(s) planned "
              f"({'DRY-RUN' if dry_run else 'LIVE'})")
        placed = rejected = 0
        for sym, side, notional in orders:
            r = executor.execute(sym, side, notional)
            tag = "PLACED" if r.placed else ("OK " if r.accepted else "REJ ")
            print(f"  {tag} {side:4s} {sym:6s} ${notional:8.2f}"
                  + (f"  ({r.reason})" if r.reason else ""))
            placed += r.placed
            rejected += (not r.accepted)
        print(f"done: {placed} placed, {rejected} rejected, "
              f"{len(orders) - placed - rejected} dry-run accepted")
    finally:
        if session_cm is not None:
            session_cm.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
