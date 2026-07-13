"""PaperTrader loop edge cases: last-bar entry suppression + force-close reason."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rhagent.papertrade import PaperTrader


@dataclass
class _Decision:
    target: float
    reason: str


class _Src:
    def bars(self):
        idx = pd.date_range("2026-07-01", periods=4, freq="D")
        px = [100.0, 101.0, 102.0, 103.0]
        df = pd.DataFrame(
            {"open": px, "high": px, "low": px, "close": px, "volume": [1] * 4},
            index=idx,
        )
        return {"AAA": df}


def _run(engine) -> list[dict]:
    out = Path(tempfile.mkdtemp())
    run_dir = PaperTrader(engine=engine, source=_Src(), out_dir=out).run()
    return [json.loads(line) for line in (run_dir / "trades.jsonl").open()]


def test_fresh_entry_on_final_bar_is_suppressed():
    """A brand-new position signalled on the last bar has no future to hold
    into; it must not book a 0-bar phantom round-trip."""

    class LateEntry:
        name = "late_entry"

        def decide(self, sym, history, prev):
            on_last_bar = len(history) == 4
            return _Decision(1.0 if on_last_bar else 0.0, "late long")

    assert _run(LateEntry()) == []


def test_open_position_force_closes_with_end_of_data_reason():
    """A position still open at the end closes at the last bar, and its exit
    reason reflects the force-close — not the stale entry signal."""

    class AlwaysLong:
        name = "always_long"

        def decide(self, sym, history, prev):
            return _Decision(1.0, "always long")

    trades = _run(AlwaysLong())
    assert len(trades) == 1
    t = trades[0]
    assert t["holding_bars"] == 3
    assert t["exit_reason"] == "end_of_data"
    assert t["entry_reason"] == "always long"
