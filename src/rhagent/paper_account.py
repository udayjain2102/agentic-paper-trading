"""Persistent paper-trading account state.

MockBroker alone starts fresh every run -- empty positions, zero P&L -- which
makes ``total_deployed_max_usd`` and ``daily_loss_limit_usd`` meaningless: a
cap can never be approached and a loss can never accumulate if nothing ever
carries over. This module is the missing day-over-day layer: it persists
positions (as shares + cost basis, so they can be marked to market), cash, and
today's realized P&L as JSON under ``journal/``, so the guardrails compare
against the real accumulated account, not an empty one.

DRY-RUN ONLY. This file has no broker, no network, no LIVE check -- it just
reads/writes JSON and does arithmetic. The only prices it ever sees are the
last cached close from ``data/`` (the same cache ``forward.py`` reads).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict

from .guardrails import Account

DEFAULT_STARTING_CASH_USD = 5_000.0


@dataclass
class Position:
    shares: float
    cost_basis_usd: float


@dataclass
class AccountState:
    date: str
    cash_usd: float
    realized_pnl_today_usd: float
    positions: Dict[str, Position] = field(default_factory=dict)

    @classmethod
    def initial(
        cls, *, starting_cash_usd: float = DEFAULT_STARTING_CASH_USD, today: date | None = None
    ) -> "AccountState":
        return cls(
            date=str(today or date.today()),
            cash_usd=starting_cash_usd,
            realized_pnl_today_usd=0.0,
            positions={},
        )

    @classmethod
    def load(
        cls, path: str | Path, *, starting_cash_usd: float = DEFAULT_STARTING_CASH_USD
    ) -> "AccountState":
        path = Path(path)
        if not path.exists():
            return cls.initial(starting_cash_usd=starting_cash_usd)
        raw = json.loads(path.read_text(encoding="utf-8"))
        positions = {s: Position(**p) for s, p in raw.get("positions", {}).items()}
        return cls(
            date=raw["date"],
            cash_usd=raw["cash_usd"],
            realized_pnl_today_usd=raw["realized_pnl_today_usd"],
            positions=positions,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "date": self.date,
            "cash_usd": self.cash_usd,
            "realized_pnl_today_usd": self.realized_pnl_today_usd,
            "positions": {
                s: {"shares": p.shares, "cost_basis_usd": p.cost_basis_usd}
                for s, p in self.positions.items()
            },
        }
        path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    def roll_to_day(self, today: date | None = None) -> None:
        """Reset the daily realized-P&L counter the first time a new
        trading day is seen. Idempotent within the same day."""
        today_s = str(today or date.today())
        if self.date != today_s:
            self.date = today_s
            self.realized_pnl_today_usd = 0.0

    def position_values(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Mark every held (nonzero) symbol to market using `prices` (last
        cached close). A symbol missing from `prices` is skipped rather than
        priced at zero -- a stale/missing quote must never look like a loss."""
        return {
            s: pos.shares * prices[s]
            for s, pos in self.positions.items()
            if pos.shares and s in prices
        }

    def to_guardrail_account(self, prices: Dict[str, float]) -> Account:
        values = self.position_values(prices)
        return Account(
            buying_power_usd=self.cash_usd,
            total_position_value_usd=sum(values.values()),
            positions=frozenset(values),
            realized_pnl_today_usd=self.realized_pnl_today_usd,
            position_values=values,
        )

    def apply_fill(self, symbol: str, side: str, notional_usd: float, price: float) -> None:
        """Book an accepted, filled paper order against this account.

        Buys and sells here are always whole-position moves (see
        ``strategy_runner``: enter flat->1 at fixed notional, exit 1->flat at
        the full held value), so a sell always fully closes the symbol and
        realizes its P&L against cost basis in one step.
        """
        if side == "buy":
            pos = self.positions.setdefault(symbol, Position(0.0, 0.0))
            pos.shares += notional_usd / price
            pos.cost_basis_usd += notional_usd
            self.cash_usd -= notional_usd
        elif side == "sell":
            pos = self.positions.pop(symbol, Position(0.0, 0.0))
            self.realized_pnl_today_usd += notional_usd - pos.cost_basis_usd
            self.cash_usd += notional_usd
        else:
            raise ValueError(f"unknown side {side!r}")


def _selfcheck() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "account.json"

        s = AccountState.load(path)
        assert s.cash_usd == DEFAULT_STARTING_CASH_USD
        s.apply_fill("AAPL", "buy", 250.0, price=100.0)
        assert s.positions["AAPL"].shares == 2.5
        assert s.cash_usd == DEFAULT_STARTING_CASH_USD - 250.0
        s.save(path)

        # round-trip: a fresh load sees the same position + cash.
        s2 = AccountState.load(path)
        assert s2.positions["AAPL"].shares == 2.5
        assert s2.cash_usd == DEFAULT_STARTING_CASH_USD - 250.0

        # mark to market at a higher price, then exit -- realized P&L booked.
        acct = s2.to_guardrail_account({"AAPL": 120.0})
        assert acct.total_position_value_usd == 300.0
        s2.apply_fill("AAPL", "sell", 300.0, price=120.0)
        assert s2.realized_pnl_today_usd == 50.0
        assert "AAPL" not in s2.positions

        # new day resets today's realized P&L, not the cash it already booked.
        s2.roll_to_day(date(2026, 1, 2))
        assert s2.realized_pnl_today_usd == 0.0
        assert s2.cash_usd == DEFAULT_STARTING_CASH_USD + 50.0

    print("paper_account selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
