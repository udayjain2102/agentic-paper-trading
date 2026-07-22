"""Evaluation over paper-trade ledgers.

Pure functions over the files PaperTrader writes: the per-trade ledger, the
aggregate scorecard, failure buckets (where do losses concentrate), and the
run-to-run comparison. Return metrics reuse backtest.result_from_returns so
the numbers match the vectorized path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import result_from_returns


def load_run(run_dir: str | Path) -> tuple[dict, pd.DataFrame, pd.Series]:
    run_dir = Path(run_dir)
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"not a run directory (no run.json): {run_dir}")
    meta = json.loads(meta_path.read_text())

    records = [
        json.loads(line)
        for line in (run_dir / "trades.jsonl").read_text().splitlines()
        if line.strip()
    ]
    trades = pd.DataFrame(records)
    if len(trades):
        feats = pd.json_normalize(trades.pop("entry_features")).add_prefix("feat_")
        trades = pd.concat([trades, feats], axis=1)

    rets = pd.read_csv(run_dir / "returns.csv", parse_dates=["date"])
    net = rets.set_index("date")["net"]
    return meta, trades, net


def aggregate(trades: pd.DataFrame, net: pd.Series) -> dict:
    res = result_from_returns(net.astype(float))
    if len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "total_return": res.total_return,
            "sharpe": res.sharpe, "max_drawdown": res.max_drawdown,
            "avg_holding_bars": 0.0,
        }
    pnl = trades["pnl_abs"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "n_trades": int(len(trades)),
        "win_rate": float((trades["outcome"] == "win").mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "total_return": res.total_return,
        "sharpe": res.sharpe,
        "max_drawdown": res.max_drawdown,
        "avg_holding_bars": float(trades["holding_bars"].mean()),
    }


def _bucket_labels(trades: pd.DataFrame) -> dict[str, pd.Series]:
    vol = trades["feat_vol20"].astype(float)
    try:
        vol_bucket = pd.qcut(vol, 3, labels=["low", "med", "high"], duplicates="drop")
    except ValueError:  # too few distinct values to cut
        vol_bucket = pd.Series("all", index=trades.index)
    gap = trades["feat_gap"].astype(float)
    gap_bucket = pd.Series("flat", index=trades.index)
    gap_bucket[gap < -0.005] = "down"
    gap_bucket[gap > 0.005] = "up"
    holding = pd.Series(
        ["short" if h < 5 else "long" for h in trades["holding_bars"]],
        index=trades.index,
    )
    labels = {
        "vol": vol_bucket.astype(str),
        "gap": gap_bucket,
        "holding": holding,
        "symbol": trades["symbol"],
        "side": trades["side"],
    }

    # NaN feature values (trades from ledgers written before the feature
    # existed) stay NaN so groupby drops them from that dimension.
    if "feat_dow" in trades:
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        labels["dow"] = pd.Series(
            [dow_names[int(round(v))]
             if pd.notna(v) and 0 <= int(round(v)) <= 4 else None
             for v in trades["feat_dow"].astype(float)],
            index=trades.index, dtype=object,
        )

    if "feat_dist_high20" in trades:
        dist_high20 = trades["feat_dist_high20"].astype(float)
        near_high = pd.Series(None, index=trades.index, dtype=object)
        near_high[dist_high20.notna()] = "far_below"
        near_high[dist_high20 > -0.05] = "mid"
        near_high[dist_high20 > -0.005] = "at_high"
        labels["near_high"] = near_high

    return labels


def failure_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    cols = ["dimension", "bucket", "n_trades", "win_rate", "loss_share"]
    if len(trades) == 0:
        return pd.DataFrame(columns=cols)

    pnl = trades["pnl_abs"].astype(float)
    total_loss = float(-pnl[pnl < 0].sum())

    rows = []
    for dim, labels in _bucket_labels(trades).items():
        for bucket, idx in trades.groupby(labels).groups.items():
            sub = trades.loc[idx]
            sub_pnl = sub["pnl_abs"].astype(float)
            bucket_loss = float(-sub_pnl[sub_pnl < 0].sum())
            rows.append({
                "dimension": dim,
                "bucket": str(bucket),
                "n_trades": int(len(sub)),
                "win_rate": float((sub["outcome"] == "win").mean()),
                "loss_share": bucket_loss / total_loss if total_loss > 0 else 0.0,
            })
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values("loss_share", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def spy_benchmark(dates: pd.DatetimeIndex, cache_dir: str | Path = "data") -> dict:
    """SPY buy-and-hold return over the same window as `dates` (a strategy's
    return-series index).

    Windowed to whatever SPY.csv rows fall inside [dates.min(), dates.max()]
    -- if the price cache lags the strategy's window, the reported end date
    trails it too. The window actually used is always returned alongside the
    return so a stale cache can't quietly be read as covering the full
    strategy window.
    """
    empty = {"return": 0.0, "start": None, "end": None, "n_days": 0}
    spy_path = Path(cache_dir) / "SPY.csv"
    if len(dates) == 0 or not spy_path.exists():
        return empty
    spy = pd.read_csv(spy_path, parse_dates=["date"]).set_index("date")["close"].sort_index()
    start, end = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    window = spy[(spy.index >= start) & (spy.index <= end)]
    if len(window) < 2:
        return empty
    return {
        "return": float(window.iloc[-1] / window.iloc[0] - 1.0),
        "start": str(window.index[0].date()),
        "end": str(window.index[-1].date()),
        "n_days": int(len(window)),
    }


def compare_runs(base_dir: str | Path) -> pd.DataFrame:
    base_dir = Path(base_dir)
    rows = []
    for meta_path in sorted(base_dir.glob("*/run.json")):
        meta, trades, net = load_run(meta_path.parent)
        a = aggregate(trades, net)
        won = int((trades["outcome"] == "win").sum()) if len(trades) else 0
        lost = int((trades["outcome"] == "loss").sum()) if len(trades) else 0
        notional = float(meta.get("notional", 10_000.0))
        net_pnl = notional * float(a["total_return"])
        rows.append({
            "run_id": meta["run_id"],
            "engine": meta["engine"],
            "n_trades": a["n_trades"],
            "won": won,
            "lost": lost,
            "win_rate": a["win_rate"],
            "profit_factor": a["profit_factor"],
            "net_pnl": net_pnl,
            "total_return": a["total_return"],
            "sharpe": a["sharpe"],
            "max_drawdown": a["max_drawdown"],
        })
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
