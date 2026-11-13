"""Experiment: train the trade filter on 15 years, fine-tune on the years after.

Same filter as trade_filter.py, different training schedule. A model is fitted
on trades closed in 1995-2009, then fine-tuned (boosting continued on new data
only) on the following years. Two versions, because fine-tuning on all of the
remaining 15 years leaves almost nothing to test on:

  10y fine-tune (2010-2019), tested on 2020 to 2025-05   <- the real test
  15y fine-tune (2010-2024), tested on 2025 onward       <- as asked; ~470 days

Each is compared with the 15-year model left un-tuned and with no filter at
all, on the same held-out days. Bar: beat the unfiltered rule's Sharpe with the
CI of the difference above zero.

    PYTHONPATH=src uv run --with scikit-learn --with pandas python experiments/trade_filter_split.py data_long
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402
from trade_filter import (COST_BPS, FILL, basket, market_features,  # noqa: E402
                          symbol_features, trades_of)
from rhagent.data import _read_csv  # noqa: E402
from rhagent.strategies import build  # noqa: E402

MODEL = dict(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0,
             early_stopping=False, warm_start=True)
TRAIN_END = "2010-01-01"
SCHEDULES = {"ft10": ("2020-01-01", "2020-01-01", "2025-05-31"),   # tune_end, test_start, test_end
             "ft15": ("2025-01-01", "2025-01-01", None)}


def main(data_dir: str) -> None:
    bars = {p.stem: _read_csv(p) for p in sorted(Path(data_dir).glob("*.csv"))}
    strat = build("mean_reversion", {})
    raw = {s: strat.positions(b) for s, b in bars.items()}
    mkt = market_features(bars)
    trades = pd.concat([trades_of(s, bars[s], raw[s], symbol_features(b, mkt))
                        for s, b in bars.items()], ignore_index=True)
    cols = [c for c in trades.columns if c not in ("symbol", "entry", "exit", "pnl")]
    win = trades["pnl"] > 0

    def filtered(model, seen, test):
        """Positions with the test-window trades the model scores below the win
        rate of the trades it was fitted on set to flat."""
        skip = test[model.predict_proba(test[cols])[:, 1] < win[seen.index].mean()]
        pos = {s: p.copy() for s, p in raw.items()}
        for t in skip.itertuples():
            pos[t.symbol].loc[t.entry:t.exit] = 0
        return pos, len(skip)

    base = trades[trades["exit"] < TRAIN_END]
    for name, (tune_end, test_start, test_end) in SCHEDULES.items():
        tune = trades[(trades["entry"] >= TRAIN_END) & (trades["exit"] < tune_end)]
        test = trades[trades["entry"] >= test_start]
        sl = slice(test_start, test_end)

        m15 = HistGradientBoostingClassifier(**MODEL).fit(base[cols], win[base.index])
        pos15, skip15 = filtered(m15, base, test)
        # fine-tune: keep the 200 trees learned on 1995-2009, add 200 fitted on the newer trades only
        mft = HistGradientBoostingClassifier(**MODEL).fit(base[cols], win[base.index])
        mft.set_params(max_iter=400).fit(tune[cols], win[tune.index])
        posft, skipft = filtered(mft, tune, test)

        nets = {"raw": basket(bars, raw).loc[sl], "train15": basket(bars, pos15).loc[sl],
                f"train15_{name}": basket(bars, posft).loc[sl]}
        n_test = int((test["entry"] <= (test_end or "2100")).sum())
        print(f"\n== {name}: train {len(base)} trades to 2009, fine-tune {len(tune)} to {tune_end[:4]}, "
              f"test {n_test} trades; skipped {skip15} un-tuned, {skipft} fine-tuned")
        config = {"cost_bps": COST_BPS, "fill": FILL, "model": MODEL, "train_end": TRAIN_END,
                  "tune_end": tune_end, "test": [test_start, test_end], "features": cols,
                  "keep_rule": "score >= win rate of the trades last fitted on"}
        for variant, net in nets.items():
            row = ledger.record("trade_filter_split", f"{variant}.{name}", net, config,
                                note="survivorship-biased universe; price-only data")
            print(f"  {variant:<13} sharpe {row['sharpe']:.2f} [{row['sharpe_ci_lo']:.2f}, {row['sharpe_ci_hi']:.2f}]"
                  f"  ann {row['ann_return']:.2%}  maxDD {row['max_drawdown']:.2%}  n={row['n_days']}")
        for a, b in ((f"train15_{name}", "raw"), (f"train15_{name}", "train15"), ("train15", "raw")):
            c = ledger.compare(nets[a], nets[b])
            print(f"  {a} - {b}: Sharpe diff {c['diff']:+.2f}  95% CI [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")


if __name__ == "__main__":
    main(sys.argv[1])
