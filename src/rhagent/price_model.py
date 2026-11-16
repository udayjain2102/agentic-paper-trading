"""Price-only cross-sectional model — the ``model`` engine.

A small logistic regression over lookahead-free per-symbol features predicts,
for each symbol-day, the probability that the next day's close-to-close return
beats the trading cost (``cost_bps``). Each decision date it holds every
symbol scoring above ``threshold`` — optionally capped at the ``top_k``
highest scores — and is flat otherwise. It has no language, no news, and no
prior knowledge: it can only have learned patterns present in the price
history it was trained on.

Lookahead-freedom is structural, not conventional: a position for date ``d``
is produced by a model trained exclusively on rows whose label had already
resolved by ``d`` (``label_date <= d``). The batch bake-off refits monthly;
the live forward tick refits on the full resolved history for each new bar —
both are expanding-window and never score a day they trained on.

Features (all computable from bars up to and including the decision date):
ret1/ret5/ret20, vol20 (20d return stdev), vol_ratio (volume vs 20d mean),
dist_ma20 (close vs 20d mean), zscore20, and the equal-weight market's
ret1/ret5 that day. Same family features.py already computes for entry
bucketing, vectorized here because the model needs them for every
symbol-day, not just entries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret1", "ret5", "ret20", "vol20", "vol_ratio",
    "dist_ma20", "zscore20", "mkt_ret1", "mkt_ret5",
]


def _symbol_frame(sym: str, bars: pd.DataFrame,
                  horizon: int = 1) -> pd.DataFrame:
    close = bars["close"].astype(float)
    ret1 = close.pct_change()
    out = pd.DataFrame(index=bars.index)
    out["symbol"] = sym
    out["ret1"] = ret1
    out["ret5"] = close.pct_change(5)
    out["ret20"] = close.pct_change(20)
    out["vol20"] = ret1.rolling(20).std()
    if "volume" in bars.columns:
        vol = bars["volume"].astype(float)
        out["vol_ratio"] = vol / vol.rolling(20).mean()
    else:
        out["vol_ratio"] = np.nan
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    out["dist_ma20"] = close / ma20 - 1.0
    out["zscore20"] = (close - ma20) / sd20
    # Extended candidates (NOT in FEATURES — available to experiments like
    # pretrain --feature-set rich, invisible to the live engine's feature
    # list): longer momentum, overnight/intraday 5d decomposition, range.
    out["ret60"] = close.pct_change(60)
    out["ret120"] = close.pct_change(120)
    if {"open", "high", "low"} <= set(bars.columns):
        opn, high, low = bars["open"].astype(float), bars["high"].astype(float), bars["low"].astype(float)
        overnight = opn / close.shift(1) - 1.0
        intraday = close / opn - 1.0
        out["on5"] = overnight.rolling(5).sum()
        out["in5"] = intraday.rolling(5).sum()
        out["range20"] = ((high - low) / close).rolling(20).mean()
    else:
        out["on5"] = out["in5"] = out["range20"] = np.nan
    # Label: the close-to-close return over the next ``horizon`` bars.
    # label_date is the date that return resolves — the cutoff the
    # walk-forward filters on. Any single-day |move| over 50% inside the
    # resolution window is a split or a bad tick, not a return anyone could
    # trade -- mask it out of the label entirely (features stay, they're
    # computed from past data only). For horizon=1 the window check and the
    # cumulative check coincide, reproducing the original mask exactly.
    fwd = close.shift(-horizon) / close - 1.0
    daily_bad = (close.pct_change().abs() > 0.5).astype(np.float64)
    rev = daily_bad.iloc[::-1]
    trail = rev.rolling(horizon, min_periods=1).max().iloc[::-1]
    window_bad = trail.shift(-horizon).fillna(0.0).astype(bool)
    bad = window_bad | (fwd.abs() > 0.5 * horizon)
    fwd = fwd.mask(bad)
    out["fwd_ret"] = fwd
    label_date = pd.Series(
        list(bars.index[horizon:]) + [pd.NaT] * horizon, index=bars.index)
    out["label_date"] = label_date.mask(bad)
    return out


def build_panel(bars: dict[str, pd.DataFrame], cost_bps: float = 7.0,
                horizon: int = 1) -> pd.DataFrame:
    """One row per (symbol, date) with features, forward return, and label.

    ``y`` is 1 when the ``horizon``-day close-to-close return beats
    ``cost_bps``. Rows missing any feature or the forward return are dropped.
    """
    frames = []
    for sym, df in bars.items():
        f = _symbol_frame(sym, df, horizon)
        f["date"] = f.index
        frames.append(f.reset_index(drop=True))
    panel = pd.concat(frames, ignore_index=True)

    # Market leg: equal-weight mean of the universe's own ret1/ret5 per date.
    mkt = (panel.groupby("date")[["ret1", "ret5"]].mean()
           .rename(columns={"ret1": "mkt_ret1", "ret5": "mkt_ret5"}))
    panel = panel.merge(mkt, left_on="date", right_index=True, how="left")

    # y is NaN where fwd_ret is (the last bar): those rows can be scored but
    # never trained on — label_date is NaT there, so every cutoff filter
    # excludes them. Keeping them is what lets positions_on decide the most
    # recent bar, whose label by definition hasn't resolved yet.
    panel["y"] = np.where(np.isfinite(panel["fwd_ret"]),
                          (panel["fwd_ret"] > cost_bps / 1e4).astype(float),
                          np.nan)
    # inf is as unusable as NaN (zscore/vol_ratio divide by rolling stats that
    # can be 0 on halted/flat names) -- fold it into the dropna below or it
    # poisons standardization and the whole fit goes NaN.
    panel[FEATURES] = panel[FEATURES].replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=FEATURES)
    return panel.reset_index(drop=True)


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                  iters: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IRLS logistic regression. Returns (weights, feat_mean, feat_std).

    X is standardized inside; the intercept is unpenalized. IRLS converges in
    a handful of iterations on this size problem — far cheaper than gradient
    descent and deterministic, which matters because forward ticks refit.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xa = np.column_stack([np.ones(len(Xs)), Xs])
    n, k = Xa.shape
    w = np.zeros(k)
    reg = np.ones(k) * l2
    reg[0] = 0.0  # don't penalize the intercept
    I = np.eye(k)
    for _ in range(iters):
        z = np.clip(Xa @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        wt = np.clip(p * (1.0 - p), 1e-4, None)
        g = Xa.T @ (p - y) + reg * w
        H = (Xa.T * wt) @ Xa + l2 * I
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w, mu, sd


def _predict_proba(X: np.ndarray, w: np.ndarray, mu: np.ndarray,
                   sd: np.ndarray) -> np.ndarray:
    Xs = (X - mu) / sd
    Xa = np.column_stack([np.ones(len(Xs)), Xs])
    z = np.clip(Xa @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _select(probs: pd.Series, threshold: float, top_k: int | None) -> list[str]:
    """Symbols whose P(beat cost) clears ``threshold``, capped at ``top_k``
    highest scores when set. probs is indexed by symbol for one date."""
    picked = probs[probs > threshold]
    if top_k is not None:
        picked = picked.sort_values(ascending=False).head(top_k)
    return list(picked.index)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit on ``train`` rows, return P(beat cost) for each ``test`` row."""
    w, mu, sd = _fit_logistic(train[FEATURES].to_numpy(),
                              train["y"].to_numpy())
    return _predict_proba(test[FEATURES].to_numpy(), w, mu, sd)


def walkforward_positions(bars: dict[str, pd.DataFrame], *,
                          cost_bps: float = 7.0, threshold: float = 0.5,
                          top_k: int | None = None,
                          min_train_rows: int = 20_000,
                          fold: str = "M", horizon: int = 1
                          ) -> dict[str, pd.Series]:
    """Monthly walk-forward positions for every symbol.

    For each fold: fit on rows whose label resolved before the fold's first
    date, score every symbol-day inside the fold, hold the selections. Days
    before the first viable fold (or with no selection) are flat 0.
    Returns {symbol: position Series indexed by that symbol's bar dates}.
    """
    panel = build_panel(bars, cost_bps, horizon)
    panel["_period"] = panel["date"].dt.to_period(fold)

    pos = {s: pd.Series(0.0, index=df.index) for s, df in bars.items()}
    for period in sorted(panel["_period"].unique()):
        cutoff = panel.loc[panel["_period"] == period, "date"].min()
        train = panel[panel["label_date"] < cutoff]
        if len(train) < min_train_rows:
            continue
        test = panel[panel["_period"] == period]
        probs = _fit_predict(train, test)
        scored = pd.DataFrame({"date": test["date"].to_numpy(),
                               "symbol": test["symbol"].to_numpy(),
                               "prob": probs})
        for d, grp in scored.groupby("date"):
            p = grp.set_index("symbol")["prob"]
            for sym in _select(p, threshold, top_k):
                if sym in pos and d in pos[sym].index:
                    pos[sym].loc[d] = 1.0
    return pos


def positions_on(bars: dict[str, pd.DataFrame], dates,
                 *, cost_bps: float = 7.0, threshold: float = 0.5,
                 top_k: int | None = None, horizon: int = 1) -> dict:
    """Live decisions: {date: {symbol: (target, prob)}} for each date in
    ``dates``, each from a model trained only on labels resolved by that
    date. Used by the forward tick for newly-decided bars."""
    panel = build_panel(bars, cost_bps, horizon)
    out: dict = {}
    for d in sorted(pd.Timestamp(x) for x in dates):
        train = panel[panel["label_date"] <= d]
        test = panel[panel["date"] == d]
        if len(train) == 0 or len(test) == 0:
            out[d] = {}
            continue
        probs = _fit_predict(train, test)
        scored = pd.DataFrame({"symbol": test["symbol"].to_numpy(),
                               "prob": probs})
        chosen = set(_select(scored.set_index("symbol")["prob"],
                             threshold, top_k))
        out[d] = {s: (1.0 if s in chosen else 0.0, float(p))
                  for s, p in zip(scored["symbol"], scored["prob"])}
    return out
