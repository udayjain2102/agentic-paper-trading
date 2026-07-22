"""Event-driven paper-trading harness.

Steps a DecisionEngine through bars one day at a time, turns position changes
into discrete ID-stamped trades, and writes an append-only ledger under
journal/papertrade/{run_id}/. Two seams keep it world-model-ready: bars come
from a MarketSource and orders are priced by a FillModel — swap either without
touching the loop. The vectorized backtest.py is untouched and remains the
fast ranking path.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from .data import get_bars
from .engine import AgentEngine, DecisionEngine, StrategyEngine
from .features import entry_features  # noqa: F401  (re-exported for callers/tests)
from .overlay import IdentityOverlay, Overlay, build_overlay


class MarketSource(Protocol):
    def bars(self) -> dict[str, pd.DataFrame]: ...


class FillModel(Protocol):
    def fill(self, symbol: str, delta: float, bar: pd.Series) -> float: ...


class HistoricalSource:
    """Real cached history via data.get_bars (offline once cached)."""

    def __init__(self, symbols, start: str, end: str, cache_dir="data") -> None:
        self.symbols = list(symbols)
        self.start, self.end, self.cache_dir = start, end, cache_dir

    def bars(self) -> dict[str, pd.DataFrame]:
        frames = get_bars(self.symbols, self.start, self.end, cache_dir=self.cache_dir)
        # Multi-symbol runs need one shared bar index; cached ranges differ
        # (later listings, gaps), so intersect to the common dates. But one
        # stunted cache (e.g. a refresh that only wrote a handful of bars)
        # would otherwise collapse the shared index for everyone and silently
        # starve every strategy's lookback. Drop such outliers first.
        if len(frames) > 1:
            lengths = sorted(len(df) for df in frames.values())
            median_len = lengths[len(lengths) // 2]
            dropped = {s: len(df) for s, df in frames.items() if len(df) < median_len / 2}
            if dropped:
                for s, n in dropped.items():
                    print(
                        f"dropping {s}: {n} bars vs median {median_len} — "
                        "cache too short, refetch or backfill",
                        file=sys.stderr,
                    )
                frames = {s: df for s, df in frames.items() if s not in dropped}
            if not frames:
                raise ValueError("all symbols dropped: cached history too short for every symbol")

        if len(frames) > 1:
            common = None
            for df in frames.values():
                common = df.index if common is None else common.intersection(df.index)
            frames = {s: df.loc[common] for s, df in frames.items()}
        return frames


class CloseFill:
    """Perfect fill at the bar's close. cost_bps is charged by the loop."""

    def fill(self, symbol: str, delta: float, bar: pd.Series) -> float:
        return float(bar["close"])


def new_run_id(now: datetime | None = None, suffix: str | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    suffix = suffix or secrets.token_hex(4)
    return f"{now.strftime('%Y-%m-%dT%H-%M-%SZ')}-{suffix}"


class PaperTrader:
    """Drive a DecisionEngine bar-by-bar and write the trade ledger.

    A trade is a period of constant nonzero position: any target change closes
    the open trade at the fill price and, if the new target is nonzero, opens
    a new one at the same bar. Open trades at end-of-data are force-closed.
    """

    def __init__(
        self,
        engine: StrategyEngine,
        source: MarketSource,
        fill: FillModel | None = None,
        cost_bps: float = 1.0,
        notional: float = 10_000.0,
        out_dir: str | Path = "journal/papertrade",
        run_id: str | None = None,
        overlay: Overlay | None = None,
        lessons: str = "",
    ) -> None:
        self.engine = engine
        self.source = source
        self.fill = fill or CloseFill()
        self.cost_bps = cost_bps
        self.notional = notional
        self.out_dir = Path(out_dir)
        self.run_id = run_id or new_run_id()
        self.overlay = overlay or IdentityOverlay()
        self.lessons = lessons

    def run(self) -> Path:
        frames = self.source.bars()
        if not frames:
            raise ValueError("no symbols: MarketSource returned no bar frames")
        for s, df in frames.items():
            if len(df) < 2:
                raise ValueError(f"history too short for {s}: {len(df)} bars")

        symbols = sorted(frames)
        index = frames[symbols[0]].index
        for s in symbols[1:]:
            if not frames[s].index.equals(index):
                raise ValueError(
                    f"bar indices differ across symbols ({symbols[0]} vs {s}); "
                    "align/intersect the cached ranges before running"
                )
        pos: dict[str, float] = {s: 0.0 for s in symbols}
        open_trades: dict[str, dict] = {}
        trades: list[dict] = []
        daily_net: list[float] = []
        seq = 0

        def close_trade(sym: str, ts, price: float, reason: str, bar_i: int) -> None:
            tr = open_trades.pop(sym)
            q = tr["qty"]
            sign = 1.0 if q > 0 else -1.0
            pnl_pct = sign * (price / tr["entry_price"] - 1.0) - (
                2.0 * abs(q) * self.cost_bps / 1e4
            )
            tr.update(
                exit_ts=str(ts),
                exit_price=price,
                exit_reason=reason,
                pnl_pct=pnl_pct,
                # Each symbol only ever gets a 1/N slice of notional (see the
                # daily_net accrual below), so the per-trade dollar figure
                # must use that same slice, not the full notional — otherwise
                # gross win/loss (sum of pnl_abs) run ~N times larger than the
                # portfolio's actual net P&L.
                pnl_abs=(self.notional / len(symbols)) * abs(q) * pnl_pct,
                holding_bars=bar_i - tr.pop("_entry_i"),
                outcome=(
                    "win" if pnl_pct > 0 else "loss" if pnl_pct < 0 else "flat"
                ),
            )
            trades.append(tr)

        for i, ts in enumerate(index):
            net_today = []
            # Snapshot once per bar, before any symbol in this bar can close a
            # trade: every overlay call this bar sees only prior-bar closes,
            # never a same-bar close from an earlier symbol in sorted order.
            closed_snapshot = pd.DataFrame(trades)
            for sym in symbols:
                bars = frames[sym]
                history = bars.iloc[: i + 1]
                bar = bars.iloc[i]
                prev = pos[sym]

                d = self.engine.decide(sym, history, prev)
                target = self.overlay.adjust(sym, history, d, closed_snapshot)
                # Don't open a fresh position on the final bar: there is no
                # future bar to hold it into, so end_of_data would immediately
                # force-close it for a 0-bar phantom round-trip. Guard here,
                # before turnover cost, so the equity curve doesn't pay for a
                # trade that never opens. Existing positions still close.
                if target != 0.0 and prev == 0.0 and i == len(index) - 1:
                    target = prev

                # accrue yesterday's position over today's move
                ret = 0.0
                if i > 0:
                    ret = prev * (
                        float(bar["close"]) / float(bars["close"].iloc[i - 1]) - 1.0
                    )
                # Unlike backtest.net_returns (which drops the final row and its
                # cost), this loop charges turnover on every bar including the
                # last, so total_return can differ slightly when the position
                # changes on the final bar.
                turnover = abs(target - prev)
                net_today.append(ret - turnover * self.cost_bps / 1e4)

                if target != prev:
                    price = self.fill.fill(sym, target - prev, bar)
                    if prev != 0.0:
                        close_trade(sym, ts, price, d.reason, i)
                    if target != 0.0:
                        seq += 1
                        open_trades[sym] = {
                            "trade_id": f"{self.run_id}#{seq:04d}",
                            "run_id": self.run_id,
                            "symbol": sym,
                            "side": "long" if target > 0 else "short",
                            "qty": target,
                            "entry_ts": str(ts),
                            "entry_price": price,
                            "entry_reason": d.reason,
                            "entry_features": entry_features(history),
                            "_entry_i": i,
                        }
                    pos[sym] = target
            daily_net.append(sum(net_today) / len(symbols))

        # force-close whatever is still open at the last bar
        last_i = len(index) - 1
        for sym in sorted(open_trades):
            last_close = float(frames[sym]["close"].iloc[-1])
            close_trade(sym, index[-1], last_close, "end_of_data", last_i)

        return self._write(symbols, index, trades, daily_net)

    def _write(self, symbols, index, trades, daily_net) -> Path:
        run_dir = self.out_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": self.run_id,
            "engine": self.engine.name,
            "symbols": symbols,
            "start": str(index[0]),
            "end": str(index[-1]),
            "cost_bps": self.cost_bps,
            "notional": self.notional,
            "overlay": self.overlay.name,
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "lessons": self.lessons,
        }
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

        with (run_dir / "trades.jsonl").open("w", encoding="utf-8") as fh:
            for tr in trades:
                fh.write(json.dumps(tr, sort_keys=True) + "\n")

        pd.DataFrame({"date": index, "net": daily_net}).to_csv(
            run_dir / "returns.csv", index=False
        )
        return run_dir


def _print_report(run_dir: Path) -> None:
    from .evaluate import aggregate, failure_buckets, load_run

    meta, trades, net = load_run(run_dir)
    print(f"run_id: {meta['run_id']}")
    print(f"engine: {meta['engine']}  symbols: {','.join(meta['symbols'])}")

    a = aggregate(trades, net)
    print("\naggregate stats")
    for k, v in a.items():
        print(f"  {k:<18}{v:>12.4f}" if isinstance(v, float) else f"  {k:<18}{v:>12}")

    b = failure_buckets(trades)
    print("\nfailure buckets (by loss share)")
    if len(b) == 0:
        print("  no trades")
    else:
        print(b.to_string(index=False, float_format=lambda x: f"{x:.2%}"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv[:1] == ["compare"]:
        p = argparse.ArgumentParser(prog="rhagent.papertrade compare")
        p.add_argument("--out-dir", default="journal/papertrade")
        args = p.parse_args(argv[1:])
        from .evaluate import compare_runs

        df = compare_runs(args.out_dir)
        if len(df) == 0:
            print(f"no runs found under {args.out_dir}")
        else:
            print(df.to_string(index=False))
        return 0

    from .strategies import REGISTRY, build

    p = argparse.ArgumentParser(prog="rhagent.papertrade")
    p.add_argument("--engine", required=True, choices=[*sorted(REGISTRY), "agent"])
    p.add_argument("--symbols", required=True,
                   help="comma-separated (NVDA,SPY) or 'all' for the config universe")
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--out-dir", default="journal/papertrade")
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--no-lessons", action="store_true",
                   help="agent engine only: skip feeding prior-run loss lessons")
    p.add_argument("--overlay", default="conviction",
                   choices=["none", "conviction", "bucket", "winprob"],
                   help="decision overlay applied to each target (default: the "
                        "locked-in conviction gate; pass 'none' for the raw strategy)")
    args = p.parse_args(argv)

    if args.symbols.strip().lower() == "all":
        # The config universe, not a glob of the cache dir: a stray CSV left in
        # data/ (an orphan backfill, a symbol dropped from the universe) is
        # never refreshed, and bars() intersects every index to the common
        # dates -- so one stale file silently clips the whole run's window.
        from .config import load

        symbols = sorted(load().strategy.universe)
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        p.error("no symbols given")

    end = date.today()
    start = end - timedelta(days=args.days)
    source = HistoricalSource(
        symbols, start.isoformat(), end.isoformat(), cache_dir=args.cache_dir
    )
    lessons = ""
    if args.engine == "agent":
        from .learn import lessons_from_runs

        lessons = "" if args.no_lessons else lessons_from_runs(args.out_dir)
        engine = AgentEngine(lessons=lessons)
    else:
        engine = StrategyEngine(build(args.engine, {}))  # long-only (shorting disabled)

    overlay = build_overlay(args.overlay)
    trader = PaperTrader(
        engine=engine, source=source, cost_bps=args.cost_bps,
        out_dir=args.out_dir, overlay=overlay, lessons=lessons,
    )
    run_dir = trader.run()
    _print_report(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
