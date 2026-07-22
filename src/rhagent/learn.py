"""Turn the paper-trade ledger into a compact 'lessons' block for the agent.

Pure: reads all run dirs, concatenates their trades, and asks evaluate where
losses concentrate. The resulting one-liner is fed back into the trading
agent's prompt so the next iteration avoids its worst setups.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .evaluate import failure_buckets, load_run


def lessons_from_runs(out_dir: str | Path = "journal/papertrade", *,
                      top_k: int = 3, min_trades: int = 5) -> str:
    dirs = sorted(Path(out_dir).glob("*/run.json"))
    frames = [load_run(p.parent)[1] for p in dirs]
    frames = [f for f in frames if len(f)]
    if not frames:
        return ""
    trades = pd.concat(frames, ignore_index=True)
    n_runs, n_trades = len(frames), len(trades)

    # ponytail: "side" is dropped outright -- the agent is long-only
    # (AgentEngine.allow_short=False), so "side=long" is the only setup it
    # can ever take, and "avoid side=long" is really "avoid trading". A
    # bucket whose loss_share doesn't exceed its own share of all trades is
    # dropped too: it's losing in proportion to its size, not more than its
    # size, so it's the book average restated as a finding, not a lesson.
    buckets = failure_buckets(trades)
    buckets = buckets[buckets["dimension"] != "side"]
    concentrated = buckets["loss_share"] > buckets["n_trades"] / n_trades
    worst = (
        buckets[concentrated]
        .query("n_trades >= @min_trades and loss_share > 0")
        .head(top_k)
    )
    if len(worst) == 0:
        wr = float((trades["outcome"] == "win").mean())
        return (f"Lessons from {n_runs} prior paper-trades ({n_trades} trades). "
                f"No setup cleared {min_trades} trades with disproportionate "
                f"losses; overall win_rate {wr:.0%}.")

    parts = "; ".join(
        f"{r.dimension}={r.bucket} (loss_share {r.loss_share:.0%}, "
        f"win_rate {r.win_rate:.0%}, n={r.n_trades})"
        for r in worst.itertuples()
    )
    return (f"Lessons from {n_runs} prior paper-trades ({n_trades} trades). "
            f"Losses concentrate in: {parts}. Prefer avoiding or downsizing "
            f"these setups.")


if __name__ == "__main__":
    print(lessons_from_runs())
