"""Experiment: can a model learn which mean-reversion entries to skip?

Takes every entry the locked mean-reversion rule makes on long price history,
trains a gradient-boosted classifier to predict whether the trade ends
profitable after costs, and skips entries it scores below the training base
rate. Walk-forward by calendar year: the model that judges year Y was fitted
only on trades that had already closed before Y began.

Bar, fixed before the first run: the filtered basket must beat the unfiltered
rule's Sharpe at 7bp / next-open fills, with the paired-bootstrap CI of the
difference above zero. Results go to the ledger either way.

    uv run --with duckdb python experiments/fetch_prices.py data_long
    PYTHONPATH=src uv run --with scikit-learn --with pandas python experiments/trade_filter.py data_long
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402
from rhagent.backtest import net_returns  # noqa: E402
from rhagent.data import _read_csv  # noqa: E402
from rhagent.overlay import apply_conviction  # noqa: E402
from rhagent.strategies import build  # noqa: E402

COST_BPS, FILL = 7.0, "next_open"
FIRST_TEST_YEAR = 2003          # ~8 years of trades before the first prediction
SELECTION_START = "2025-06-02"  # the locked preset was chosen on data from here on
MODEL = dict(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)


def market_features(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = pd.concat({s: b["close"] for s, b in bars.items()}, axis=1)
    ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    spy = close["SPY"]
    return pd.DataFrame({
        "mkt_ret5": spy.pct_change(5), "mkt_ret20": spy.pct_change(20),
        "mkt_vol20": spy.pct_change().rolling(20).std(),
        "mkt_dist_ma200": spy / spy.rolling(200).mean() - 1,
        # how many names are dipping at once: a lone dip vs a market-wide selloff
        "breadth_dip": ((close - ma20) / sd20 < -1).sum(axis=1) / close.notna().sum(axis=1),
        "breadth_up": (close > ma20).sum(axis=1) / close.notna().sum(axis=1),
    })


def symbol_features(b: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """Everything here is known at the close of the entry bar -- no lookahead."""
    c, r = b["close"], b["close"].pct_change()
    f = pd.DataFrame(index=c.index)
    f["z20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    f["vol20"] = r.rolling(20).std()
    f["vol_ratio60"] = f["vol20"] / r.rolling(60).std()
    for k in (1, 5, 20, 60, 250):
        f[f"ret{k}"] = c.pct_change(k)
    f["gap"] = b["open"] / c.shift(1) - 1
    f["dist_high20"] = c / c.rolling(20).max() - 1
    f["dist_ma200"] = c / c.rolling(200).mean() - 1
    f["volume_ratio"] = b["volume"] / b["volume"].rolling(20).mean()
    f["dow"] = c.index.day_of_week
    return f.join(mkt)


def trades_of(sym: str, b: pd.DataFrame, pos: pd.Series, feats: pd.DataFrame) -> pd.DataFrame:
    net = net_returns(b, pos, COST_BPS, FILL).reindex(b.index).fillna(0.0)
    prev = pos.shift(1).fillna(0)
    entries, exits = pos.index[(pos == 1) & (prev == 0)], pos.index[(pos == 0) & (prev == 1)]
    rows = []
    for t0 in entries:
        later = exits[exits > t0]
        if not len(later):
            continue  # still open at the end of the data: no outcome to learn from
        t1 = later[0]
        rows.append({"symbol": sym, "entry": t0, "exit": t1,
                     "pnl": float(net.loc[t0:t1].sum()), **feats.loc[t0].to_dict()})
    return pd.DataFrame(rows)


def basket(bars, pos, cost=COST_BPS, fill=FILL) -> pd.Series:
    df = pd.concat({s: net_returns(bars[s], pos[s], cost, fill) for s in pos}, axis=1)
    return df[df.notna().sum(axis=1) >= 30].mean(axis=1)


def main(data_dir: str) -> None:
    bars = {p.stem: _read_csv(p) for p in sorted(Path(data_dir).glob("*.csv"))}
    strat = build("mean_reversion", {})
    raw = {s: strat.positions(b) for s, b in bars.items()}
    mkt = market_features(bars)
    feats = {s: symbol_features(b, mkt) for s, b in bars.items()}
    trades = pd.concat([trades_of(s, bars[s], raw[s], feats[s]) for s in bars], ignore_index=True)
    cols = [c for c in trades.columns if c not in ("symbol", "entry", "exit", "pnl")]
    trades["score"], trades["keep"] = float("nan"), True

    for year in range(FIRST_TEST_YEAR, trades["entry"].dt.year.max() + 1):
        train = trades[trades["exit"] < pd.Timestamp(year, 1, 1)]
        test = trades.index[trades["entry"].dt.year == year]
        if not len(test):
            continue
        model = HistGradientBoostingClassifier(**MODEL).fit(train[cols], train["pnl"] > 0)
        trades.loc[test, "score"] = model.predict_proba(trades.loc[test, cols])[:, 1]
        trades.loc[test, "keep"] = trades.loc[test, "score"] >= (train["pnl"] > 0).mean()

    filtered = {s: p.copy() for s, p in raw.items()}
    for t in trades[~trades["keep"]].itertuples():
        filtered[t.symbol].loc[t.entry:t.exit] = 0
    conviction = {s: apply_conviction(raw[s], strat.signal(bars[s])) for s in bars}

    oos = trades[trades["score"].notna()]
    print(f"{len(trades)} trades, {len(oos)} judged out-of-sample from {FIRST_TEST_YEAR}")
    print(oos.groupby(oos["keep"].map({True: "kept", False: "skipped"}))["pnl"]
          .agg(n="size", win_rate=lambda x: (x > 0).mean(), avg_pnl="mean").to_string())

    start = f"{FIRST_TEST_YEAR}-01-01"
    nets = {"raw": basket(bars, raw), "conviction": basket(bars, conviction),
            "ml_filter": basket(bars, filtered)}
    config = {"cost_bps": COST_BPS, "fill": FILL, "universe": len(bars), "model": MODEL,
              "first_test_year": FIRST_TEST_YEAR, "features": cols,
              "keep_rule": "score >= training win rate"}
    for window, sl in (("oos", slice(start, "2025-05-31")), ("since_selection", slice(SELECTION_START, None))):
        print(f"\n== {window}")
        for name, net in nets.items():
            row = ledger.record("trade_filter", f"{name}.{window}", net.loc[sl], config,
                                note="survivorship-biased universe; price-only data")
            print(f"  {name:<11} sharpe {row['sharpe']:.2f} [{row['sharpe_ci_lo']:.2f}, {row['sharpe_ci_hi']:.2f}]"
                  f"  ann {row['ann_return']:.2%}  maxDD {row['max_drawdown']:.2%}  n={row['n_days']}")
        for base in ("raw", "conviction"):
            c = ledger.compare(nets["ml_filter"].loc[sl], nets[base].loc[sl])
            print(f"  ml_filter - {base}: Sharpe diff {c['diff']:+.2f}  95% CI [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")


if __name__ == "__main__":
    main(sys.argv[1])
