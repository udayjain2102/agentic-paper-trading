# Spec: `model_intl` — within-market international long/short record

Date: 2026-11-16. Status: proposal, not built.
Prior art: `docs/international-markets.md` (in the sibling checkout) — the
experiment that motivates this.

## Thesis

The price-only model finds ~zero signal on US liquid names but **+0.012
within-market rank IC (t=5.0)** on international liquid names, concentrated
in KR/TW/FR/UK, stable across all 10 holdout years (2017→2026). The pooled
+0.031 headline is mostly *between-market* ranking (+0.081) — country/macro
betting, not stock selection — so this spec trades **only the within-market
component**: rank stocks inside each market, never across markets.

## Universe

From `data_intl/intl_ohlcv_liquid.csv` (median USD dollar-vol ≥ $10M,
pre-2017, static-FX approximation):

| market | suffixes | liquid names |
|---|---|---|
| Korea | .KS/.KQ | ~20 |
| Taiwan | .TW | ~18 |
| UK | .L | ~42 |
| France | .PA | ~30 |

~110 symbols total. Small on purpose: these are the markets where the
within-market IC cleared t>3 in the holdout. Japan (+0.009, t=1.8) and the
null markets (DE/CH/CA/AU/SE) are excluded — adding them dilutes the sleeve
with noise the data says isn't there.

## Signal

Same model family as the `model` engine (`rhagent.price_model`): logistic
regression on the 9 base features, expanding-window retrain on resolved
labels only. Two adaptations vs the US record:

1. **Market-relative ranking at decision time.** `positions_on` returns
   per-symbol probabilities; the intl engine groups symbols by market
   suffix and ranks *within* each group. The model's `mkt_ret1/5` features
   will be the intl-bloc equal-weight mean (build_panel behavior) — a known
   imperfection, acceptable for v1, noted as a v2 item (per-market market
   leg).
2. **Long/short, not long-only.** The IC is symmetric — quintile spread is
   top-minus-bottom. Target position per market: long the top ~25%, short
   the bottom ~25%, flat the middle. With KR=20 names that's ~5 long / ~5
   short per side — thin but tradeable for a paper record.

## Portfolio construction

- Equal-weight within each market's selected names.
- Equal-weight across the four markets (each market = 25% of gross
  exposure) — prevents the UK sleeve (42 names) dominating Korea (20).
- Gross exposure ~100% (50/50 long-short) — it's a market-neutral-ish
  sleeve; net exposure drifts with factor tilts, which is fine for a
  measurement record.
- Rebalance: daily signal, but **trade only on rank changes** — hold unless
  a name enters/leaves its market's top or bottom quartile. This is the
  turnover control; measure both daily and weekly rebalance in the
  backtest and pick the cheaper one.

## Costs — the make-or-break assumption

US runs used 7 bps/side. International reality is worse and asymmetric:

| market | proposed paper cost | rationale |
|---|---|---|
| UK (.L) | 10 bps/side | deep, FX-friendly, closest to US microstructure |
| FR (.PA) | 15 bps/side | liquid large caps, wider than UK |
| KR (.KS) | 20 bps/side | retail-heavy tape, wider spreads, KRW FX |
| TW (.TW) | 25 bps/side | thinnest of the four, TWD FX, day-trade limits |

**Gate: the strategy must show positive net returns at these costs in the
walk-forward backtest, and remain positive at 2x cost.** If the edge only
survives at US-level costs, it's a microstructure mirage (stale closes,
bid-ask bounce) — kill it.

## FX handling

Local-currency returns flatters or punishes depending on FX drift. For the
paper record, convert daily returns to USD with Yahoo FX series
(`USDKRW=X`, `USDTWD=X`, `GBPUSD=X`→inverse, `EURUSD=X`→inverse) fetched
alongside prices. If FX plumbing proves fiddly, v1 may record local-currency
returns with an explicit FX-beta caveat — but the backtest gate above must
use USD returns.

## Shorting

Paper record ignores borrow cost and availability (disclosed limitation).
KR/TW/FR/UK large caps are generally borrowable via swaps; real implementation
would pay ~30-100 bps annualized borrow on the short side plus recalls.

## Backtest gate (before any live record)

1. Walk-forward on `data_intl` (2017→2026 holdout, expanding-window retrain,
   `pretrain.py --walk-forward` machinery or `price_model` batch path):
   simulate the exact portfolio rule above in USD terms at the cost table.
2. Report: net annualized return, Sharpe, max DD, turnover, per-market P&L,
   and the same at 2x costs. Compare against equal-weight buy-and-hold of
   the same universe and against a random-quartile baseline.
3. **Pass bar: positive net Sharpe > ~0.5 at stated costs AND positive at
   2x costs.** Below that, the edge doesn't survive reality — document and
   stop.

## Live record design (if the gate passes)

- New eval dir `journal/forward/model_intl`, engine name e.g.
  `model_intl` — a variant of `_model_positions` that (a) reads bars from a
  second cache dir (`data_intl/cache/`), (b) applies the within-market
  quartile rule, (c) emits long/short targets.
- Config: an `[intl]` block or `--universe-file`/`--cache-dir` flags on
  `rhagent.forward`; the universe list derived from suffix groups, not
  hardcoded.
- Data ops: commit `scripts/fetch_intl_ohlcv.py`; push `data_intl/cache/`
  to `paper-state` once (same mechanism as `data/`); add a refresh step in
  `paper_cron.sh` (the per-symbol cache makes daily refresh cheap — only
  new bars fetched).
- Dashboard: the record appears automatically once `returns.csv` exists;
  verify the records table renders long/short books sanely.

## Risks / honest caveats

- **Stale closes**: TW/KR names can have non-synchronous closes; the IC may
  partly measure "yesterday's US move predicts today's Asia open" — which
  is real but may already be arbitraged by the open. Test: lag the signal
  one day and see how much IC survives (cheap check, do it first).
- **Small cross-sections**: TW=18 names → quartiles of ~4-5 names. Noisy;
  the record needs months to say anything.
- **Survivorship**: hand-picked current listings; no delisted names. The
  backtest flatters.
- **FX noise**: ~13% annualized FX vol on KRW/TWD legs adds variance the
  stock signal must overcome.
- **Model staleness**: expanding-window retrain adapts, but if the
  2017-2026 effect decays (like the US did), the record will show it —
  that's what it's for.

## Milestones

1. Lag-1-day IC check on existing preds (30 min) — kill signal if it
   collapses.
2. Walk-forward backtest at the cost table (few hours compute).
3. If pass: engine + config + cron + paper-state cache (~1 day).
4. Record accumulates; review at ~40 trading days alongside the others.
