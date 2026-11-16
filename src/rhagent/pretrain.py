"""Pretrain a small torch MLP on broad market history, then score a holdout.

This is the experiment the 65-name price_model couldn't run: daily prediction
on 65 correlated mega-caps is too thin to find anything, but the same causal
features over ~7k US symbols (including delisted names, via the
``act_symbol`` column of siddharthmb/stocks-ohlcv) give the model real
cross-sectional breadth to learn from.

Split discipline: ``--train-end`` (default 2016-12-31) bounds TRAINING by
label resolution date, so no label that resolves inside the holdout can leak
in. Everything after is scored, never trained on. The metric that matters is
the daily cross-sectional rank IC of predicted P(beat cost) vs realized
next-day return, with a t-stat across days -- the same honesty bar the
65-name walk-forward failed.

Experiment-only module: needs ``pip install torch lightgbm`` (both imported
lazily so the rest of the package never requires them). Result of running
it: see docs/price-model-engine.md -- the broad-universe signal does not
survive to the tradeable/liquid universe or our 65 names.

    PYTHONPATH=src python -m rhagent.pretrain \\
        --csv data/hf/ohlcv.csv --train-end 2016-12-31 --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .price_model import FEATURES, build_panel

# Extended candidates emitted by price_model._symbol_frame but kept out of
# FEATURES so the live engine is untouched. "rich" = these raw columns plus
# a same-day cross-sectional percentile rank of every raw feature — the
# rank tells the model where a stock sits vs the universe *today*, which raw
# values can't express.
RICH_RAW = ["ret60", "ret120", "on5", "in5", "range20"]
RANK_BASE = FEATURES + RICH_RAW


def rich_features(panel: pd.DataFrame) -> list[str]:
    """Add ``<feat>_xr`` cross-sectional rank columns in place; return the
    full rich feature list."""
    for c in RANK_BASE:
        panel[f"{c}_xr"] = panel.groupby("date")[c].rank(pct=True)
    return RANK_BASE + [f"{c}_xr" for c in RANK_BASE]


def load_hf_csv(path: str | Path) -> dict[str, pd.DataFrame]:
    """siddharthmb/stocks-ohlcv -> {symbol: bars DataFrame}.

    Columns: date, act_symbol, open, high, low, close, volume. ``act_symbol``
    keeps delisted instruments under their own names, which is what makes
    this less survivorship-flattered than our 65-name universe.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"act_symbol": "symbol"})
    out = {}
    for sym, g in df.groupby("symbol", sort=False):
        b = g.set_index("date").sort_index()
        b = b[["open", "high", "low", "close", "volume"]].astype(np.float64)
        # Bad ticks and zero/negative prints poison both features and labels.
        b = b[(b["open"] > 0) & (b["high"] > 0) & (b["low"] > 0) & (b["close"] > 0)]
        if len(b):
            out[sym] = b
    return out


def train_mlp(X, y, *,
              hidden: tuple[int, ...] = (64, 32), epochs: int = 3,
              batch: int = 65_536, lr: float = 1e-3, seed: int = 0,
              device: str = "cpu", log_every: int = 0):
    """Small MLP binary classifier. Returns (model, mu, sd) for scoring."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    Xs = (X - mu) / sd

    layers = []
    d = Xs.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, 1)]
    model = nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xs)
    steps_per_epoch = (n + batch - 1) // batch
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xs[idx].to(device)
            yb = y[idx].to(device)
            loss = lossf(model(xb).squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if log_every:
            print(f"  epoch {ep + 1}/{epochs} loss {tot / n:.4f}", flush=True)
    return model, mu, sd


def predict(model, mu, sd, X, batch: int = 262_144, device: str = "cpu"):
    import torch

    Xs = (X - mu) / sd
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch):
            z = model(Xs[i:i + batch].to(device)).squeeze(-1)
            out.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(out)


def train_gbm(X: np.ndarray, y: np.ndarray, *,
              num_boost_round: int = 400, lr: float = 0.05,
              num_leaves: int = 31, min_child: int = 200,
              seed: int = 0):
    """LightGBM binary classifier -- usually stronger than a small MLP on
    tabular features. Returns a booster; predict() gives P(y=1) directly."""
    import lightgbm as lgb

    params = dict(objective="binary", metric="binary_logloss",
                  learning_rate=lr, num_leaves=num_leaves,
                  min_data_in_leaf=min_child, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=seed, verbose=-1)
    return lgb.train(params, lgb.Dataset(X, y),
                     num_boost_round=num_boost_round)


def rank_ic(test: pd.DataFrame, prob_col: str = "prob") -> pd.Series:
    """Per-date Spearman IC of prob vs fwd_ret (Pearson on ranks -- no scipy)."""
    ics = []
    dates = []
    for d, g in test.groupby("date"):
        g = g.dropna(subset=[prob_col, "fwd_ret"])
        if len(g) < 20:
            continue
        ics.append(g[prob_col].rank().corr(g["fwd_ret"].rank()))
        dates.append(d)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), dtype=float).dropna()


def quintile_spread(test: pd.DataFrame, prob_col: str = "prob",
                    q: int = 5) -> pd.DataFrame:
    """Mean next-day return by predicted-probability quintile."""
    t = test.dropna(subset=[prob_col, "fwd_ret"]).copy()
    t["q"] = t.groupby("date")[prob_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False))
    return t.groupby("q")["fwd_ret"].agg(["mean", "count"])


def walk_forward(panel: pd.DataFrame, *, train_end: pd.Timestamp,
                 feats: list[str], hidden: tuple[int, ...], epochs: int,
                 batch: int, lr: float, seed: int, model_kind: str = "mlp",
                 device: str = "cpu") -> tuple[pd.Series, pd.DataFrame]:
    """Yearly expanding-window retrain: for each test year Y, fit on rows
    whose label resolved by Dec 31 of Y-1, score every symbol-day in Y.

    This is the decay test the single-split run can't answer: if a *fresh*
    model still shows ~0 IC on liquid names, the patterns are gone, not just
    the 2016 snapshot. Returns (pooled daily IC series, predictions frame).
    """
    all_ic = []
    pred_frames = []
    for year in sorted(panel["date"].dt.year.unique()):
        if year <= train_end.year:
            continue
        cutoff = pd.Timestamp(f"{year - 1}-12-31")
        train = panel[panel["label_date"] <= cutoff].dropna(subset=["y"])
        test = panel[panel["date"].dt.year == year]
        if len(train) < 50_000 or len(test) < 1_000:
            continue
        if model_kind == "gbm":
            Xtr = train[feats].to_numpy(dtype=np.float32)
            ytr = train["y"].to_numpy(dtype=np.float32)
            booster = train_gbm(Xtr, ytr, seed=seed)
            test = test.copy()
            test["prob"] = booster.predict(
                test[feats].to_numpy(dtype=np.float32))
            del Xtr, ytr, booster
        else:
            import torch  # lazy: keeps libomp single-threaded-runtime
                           # conflicts with lightgbm away on gbm runs
            Xt = torch.from_numpy(train[feats].to_numpy(dtype=np.float32))
            yt = torch.from_numpy(train["y"].to_numpy(dtype=np.float32))
            model, mu, sd = train_mlp(Xt, yt, hidden=hidden, epochs=epochs,
                                      batch=batch, lr=lr, seed=seed,
                                      device=device, log_every=0)
            Xte = torch.from_numpy(test[feats].to_numpy(dtype=np.float32))
            test = test.copy()
            test["prob"] = predict(model, mu, sd, Xte, device=device)
            del Xt, yt, Xte, model
        ics = rank_ic(test)
        t = ics.mean() / ics.std() * np.sqrt(len(ics)) if len(ics) > 2 else np.nan
        print(f"  {year}: train {len(train):,} | IC {ics.mean():+.4f} "
              f"t {t:+.2f} ({len(ics)} days)", flush=True)
        all_ic.append(ics)
        pred_frames.append(test[["date", "symbol", "fwd_ret", "prob"]])
    if not all_ic:
        return pd.Series(dtype=float), pd.DataFrame()
    pooled = pd.concat(all_ic).sort_index()
    preds = pd.concat(pred_frames, ignore_index=True)
    return pooled, preds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rhagent.pretrain")
    p.add_argument("--csv", required=True, help="stocks-ohlcv style CSV")
    p.add_argument("--train-end", default="2016-12-31",
                   help="train on labels resolved on/before this date")
    p.add_argument("--cost-bps", type=float, default=7.0)
    p.add_argument("--horizon", type=int, default=1,
                   help="label = close-to-close return over this many bars "
                        "(5 = weekly-ish signal, ~5x less turnover)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=65_536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", default="64,32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-dollar-vol", type=float, default=0.0,
                   help="drop symbols whose median daily dollar volume "
                        "(close*volume, pre-train-end bars only) is below "
                        "this -- keeps the liquid universe where a signal "
                        "could actually be traded")
    p.add_argument("--max-symbols", type=int, default=0,
                   help="debug: cap the universe size")
    p.add_argument("--out", default="data/pretrain_model.pt")
    p.add_argument("--pred-out", default="",
                   help="optional path to pickle holdout predictions "
                        "(date, symbol, fwd_ret, prob) for subset analysis")
    p.add_argument("--walk-forward", action="store_true",
                   help="yearly expanding-window retrain instead of the "
                        "single train_end split -- the regime-decay test")
    p.add_argument("--feature-set", choices=["base", "rich"], default="base",
                   help="rich = extended price features + same-day "
                        "cross-sectional ranks")
    p.add_argument("--model", choices=["mlp", "gbm"], default="mlp")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    print("loading", args.csv, flush=True)
    bars = load_hf_csv(args.csv)
    if args.max_symbols:
        bars = dict(list(bars.items())[: args.max_symbols])
    train_end = pd.Timestamp(args.train_end)
    if args.min_dollar_vol > 0:
        keep = []
        for sym, b in bars.items():
            pre = b.loc[b.index <= train_end]
            if len(pre) < 60:
                continue
            med = float((pre["close"] * pre["volume"]).median())
            if med >= args.min_dollar_vol:
                keep.append(sym)
        bars = {s: bars[s] for s in keep}
        print(f"liquidity filter: {len(bars)} symbols "
              f"(median $vol >= {args.min_dollar_vol:,.0f})", flush=True)

    print(f"{len(bars)} symbols", flush=True)

    print("building feature panel (this is the heavy step)...", flush=True)
    panel = build_panel(bars, args.cost_bps, horizon=args.horizon)
    del bars
    panel["date"] = pd.to_datetime(panel["date"])
    panel["label_date"] = pd.to_datetime(panel["label_date"])
    print(f"panel: {len(panel):,} rows", flush=True)

    if args.feature_set == "rich":
        feats = rich_features(panel)
    else:
        feats = list(FEATURES)
    panel[feats] = panel[feats].replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=feats).reset_index(drop=True)
    print(f"feature set '{args.feature_set}': {len(feats)} features, "
          f"{len(panel):,} usable rows", flush=True)

    hidden = tuple(int(h) for h in args.hidden.split(","))
    if args.walk_forward:
        ics, preds = walk_forward(panel, train_end=train_end, feats=feats,
                                  hidden=hidden, model_kind=args.model,
                                  epochs=args.epochs, batch=args.batch,
                                  lr=args.lr, seed=args.seed)
        t_stat = (ics.mean() / ics.std() * np.sqrt(len(ics))
                  if len(ics) > 2 else float("nan"))
        print(f"\npooled walk-forward rank IC (horizon={args.horizon}d): "
              f"mean {ics.mean():+.4f}  t {t_stat:+.2f}  n {len(ics)} days",
              flush=True)
        if len(preds):
            print("quintiles (0=lowest predicted prob):")
            print(quintile_spread(preds).to_string(
                float_format=lambda x: f"{x:+.5f}"))
        if args.pred_out and len(preds):
            preds.to_pickle(args.pred_out)
            print(f"predictions -> {args.pred_out}", flush=True)
        return 0

    train = panel[panel["label_date"] <= train_end].dropna(subset=["y"])
    test = panel[panel["date"] > train_end]
    print(f"train {len(train):,} rows | holdout {len(test):,} rows "
          f"({test['date'].min().date()} -> {test['date'].max().date()})",
          flush=True)

    Xtr = train[feats].to_numpy(dtype=np.float32)
    ytr = train["y"].to_numpy(dtype=np.float32)
    test = test.copy()
    if args.model == "gbm":
        booster = train_gbm(Xtr, ytr, seed=args.seed)
        test["prob"] = booster.predict(test[feats].to_numpy(dtype=np.float32))
        model = None
    else:
        import torch
        Xt = torch.from_numpy(Xtr)
        yt = torch.from_numpy(ytr)
        model, mu, sd = train_mlp(Xt, yt, hidden=hidden, epochs=args.epochs,
                                  batch=args.batch, lr=args.lr, seed=args.seed,
                                  log_every=1)
        del Xt, yt
        Xte = torch.from_numpy(test[feats].to_numpy(dtype=np.float32))
        test["prob"] = predict(model, mu, sd, Xte)
        del Xte

    ics = rank_ic(test)
    t_stat = ics.mean() / ics.std() * np.sqrt(len(ics)) if len(ics) > 2 else float("nan")
    print(f"\nholdout rank IC (horizon={args.horizon}d): "
          f"mean {ics.mean():+.4f}  t {t_stat:+.2f}  n {len(ics)} days",
          flush=True)
    print("quintiles (0=lowest predicted prob):")
    print(quintile_spread(test).to_string(float_format=lambda x: f"{x:+.5f}"))

    if args.pred_out:
        test[["date", "symbol", "fwd_ret", "prob"]].to_pickle(args.pred_out)
        print(f"predictions -> {args.pred_out}", flush=True)

    if model is not None:
        torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                    "hidden": hidden, "features": feats,
                    "cost_bps": args.cost_bps, "train_end": args.train_end,
                    "horizon": args.horizon, "seed": args.seed},
                   args.out)
        print(f"saved -> {args.out}", flush=True)
    else:
        booster.save_model(str(args.out) + ".txt")
        print(f"saved -> {args.out}.txt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
