"""Event-driven paper-trading harness.

Steps a DecisionEngine through bars one day at a time, turns position changes
into discrete ID-stamped trades, and writes an append-only ledger under
journal/papertrade/{run_id}/. Two seams keep it world-model-ready: bars come
from a MarketSource and orders are priced by a FillModel — swap either without
touching the loop. The vectorized backtest.py is untouched and remains the
fast ranking path.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from .data import get_bars
from .engine import DecisionEngine


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
        return get_bars(self.symbols, self.start, self.end, cache_dir=self.cache_dir)


class CloseFill:
    """Perfect fill at the bar's close. cost_bps is charged by the loop."""

    def fill(self, symbol: str, delta: float, bar: pd.Series) -> float:
        return float(bar["close"])


def new_run_id(now: datetime | None = None, suffix: str | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    suffix = suffix or secrets.token_hex(4)
    return f"{now.strftime('%Y-%m-%dT%H-%M-%SZ')}-{suffix}"


def entry_features(history: pd.DataFrame) -> dict:
    """Cheap lookahead-free scalars at entry, used for failure bucketing."""
    close = history["close"].astype(float)
    rets = close.pct_change().dropna()

    vol20 = float(rets.tail(20).std()) if len(rets) >= 2 else 0.0
    if pd.isna(vol20):
        vol20 = 0.0

    gap = 0.0
    if len(close) >= 2 and "open" in history:
        gap = float(history["open"].iloc[-1] / close.iloc[-2] - 1.0)

    trend5 = 0.0
    if len(close) >= 6:
        diff = float(close.iloc[-1] - close.iloc[-6])
        trend5 = 0.0 if diff == 0 else (1.0 if diff > 0 else -1.0)

    return {"vol20": vol20, "gap": gap, "trend5": trend5}


class PaperTrader:
    """Drive a DecisionEngine bar-by-bar and write the trade ledger.

    A trade is a period of constant nonzero position: any target change closes
    the open trade at the fill price and, if the new target is nonzero, opens
    a new one at the same bar. Open trades at end-of-data are force-closed.
    """

    def __init__(
        self,
        engine: DecisionEngine,
        source: MarketSource,
        fill: FillModel | None = None,
        cost_bps: float = 1.0,
        notional: float = 10_000.0,
        out_dir: str | Path = "journal/papertrade",
        run_id: str | None = None,
    ) -> None:
        self.engine = engine
        self.source = source
        self.fill = fill or CloseFill()
        self.cost_bps = cost_bps
        self.notional = notional
        self.out_dir = Path(out_dir)
        self.run_id = run_id or new_run_id()

    def run(self) -> Path:
        frames = self.source.bars()
        if not frames:
            raise ValueError("no symbols: MarketSource returned no bar frames")
        for s, df in frames.items():
            if len(df) < 2:
                raise ValueError(f"history too short for {s}: {len(df)} bars")

        symbols = sorted(frames)
        index = frames[symbols[0]].index
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
                pnl_abs=self.notional * pnl_pct,
                holding_bars=bar_i - tr.pop("_entry_i"),
                outcome=(
                    "win" if pnl_pct > 0 else "loss" if pnl_pct < 0 else "flat"
                ),
            )
            trades.append(tr)

        for i, ts in enumerate(index):
            net_today = []
            for sym in symbols:
                bars = frames[sym]
                history = bars.iloc[: i + 1]
                bar = bars.iloc[i]
                prev = pos[sym]

                d = self.engine.decide(sym, history, prev)
                target = d.target

                # accrue yesterday's position over today's move
                ret = 0.0
                if i > 0:
                    ret = prev * (
                        float(bar["close"]) / float(bars["close"].iloc[i - 1]) - 1.0
                    )
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
            "created_ts": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

        with (run_dir / "trades.jsonl").open("w", encoding="utf-8") as fh:
            for tr in trades:
                fh.write(json.dumps(tr, sort_keys=True) + "\n")

        pd.DataFrame({"date": index, "net": daily_net}).to_csv(
            run_dir / "returns.csv", index=False
        )
        return run_dir
