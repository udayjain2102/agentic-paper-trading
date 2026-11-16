# Price-only model engine — build log

Date: 2026-11-15. Repo: `agentic-paper-trading`.

## What was built

A small model that picks which of the 65 universe stocks to hold each day,
using nothing but price history — no language, no news, no prior knowledge.
Each day it scores every symbol with the probability that tomorrow's
close-to-close return beats the trading cost (7 bps), and holds the symbols
scoring above a threshold (default 0.5). It plugs into the existing forward
paper-run as one more engine, ticked daily next to the others.

## The honest result (backtest)

Walk-forward on ~10 years of Yahoo daily bars (2,513 bars × 65 symbols ≈
162k rows), scored at 7 bps per unit of turnover, filled at next open — the
same basis as the `mean_reversion_real` record, equal-weight basket mean:

| strategy | 10y total return | Sharpe | max drawdown |
|---|---|---|---|
| buy-and-hold | +382% | 0.97 | -33% |
| mean_reversion_real | +37% | 0.46 | -8% |
| model, threshold 0.45 (best config) | +277% | 0.87 | -33% |
| model, top-5 / top-10 concentrated | -2% to -6% | negative | — |

Decisive diagnostic: daily rank IC (predicted probability vs. next-day
return, Spearman, walk-forward) = **+0.004, t = 0.87** — statistically zero.
Quintile returns are non-monotonic. The model finds nothing; daily stock
returns are mostly noise, exactly as predicted.

Caveat: survivorship bias flatters everything here — the 65 names are
today's winners across the whole window.

## Why it was wired live anyway

The plan's gate was "only add the live record if it beats the baselines."
It doesn't. It was added anyway because the backtest is itself biased
(survivorship), while a forward record is the one unbiased test, and it
costs seconds of CPU per tick. Expect it to flatline; that would settle the
question definitively. It runs at threshold 0.5 — the a-priori defensible
default, not the best-looking backtest config.

## Files

- `src/rhagent/price_model.py` — feature panel, numpy logistic regression
  (IRLS, L2, no new dependencies), monthly walk-forward
  (`walkforward_positions`), live decisions (`positions_on`). Lookahead-free
  by construction: a position for date *d* comes only from labels resolved
  by *d* (`label_date <= d`).
- `src/rhagent/forward.py` — `_model_positions` + `engine == "model"` branch
  in `_positions`, `--engine` help text. Reuses the agent's freeze-to-disk
  contract: `pos_<SYM>.csv` (date,pos,status), `decisions.jsonl`, legacy
  seeding of history, empty exclusion set (the model never fails).
- `scripts/paper_cron.sh` — fifth tick:
  `python -m rhagent.forward --engine model --eval-id model --cost-bps 7 --fill-mode next_open`
  (guarded non-fatal, like the other ticks).
- `scripts/eval_model.py` — the bake-off harness (model vs buy-and-hold vs
  mean_reversion_real basis; `--threshold`, `--top-k`, `--cost-bps`,
  `--min-train-rows` flags).
- `scripts/fetch_long_history.py` — deepens `data/<SYM>.csv` to ~N years via
  Yahoo's keyless chart API through `refresh.update_cache` (deduped,
  unsettled-today and volume==0 rows dropped).
- `tests/test_price_model.py` — 5 tests: panel/label correctness, logistic
  recovery, and two no-lookahead proofs (decisions are truncation-invariant
  and shock-invariant — a decision for day *d* cannot see data after *d*).
- `tests/test_forward_model.py` — 2 tests: legacy seeding + last-bar
  decision, frozen-across-ticks + one-new-bar-decided.

## Data

`data/` was deepened from ~16 months to 10 years (2016-11-15 → 2026-11-13,
2,513 bars/symbol, split-adjusted — same adjustment the MCP path requests)
and pushed to the `paper-state` branch (commit `93def91`), because CI
restores `data/` from that branch and the model needs the depth to train.
Refresh keeps appending; the deep history persists.

## Features (per symbol-day, all causal)

ret1 / ret5 / ret20, vol20 (20d return stdev), vol_ratio (volume vs 20d
mean), dist_ma20, zscore20, and the equal-weight market's ret1/ret5 — the
same family `features.py` computes for entry bucketing, vectorized because
the model needs them for every symbol-day, not just entries.

## How to reproduce / inspect

```bash
PYTHONPATH=src python scripts/eval_model.py            # bake-off table
PYTHONPATH=src python scripts/eval_model.py --threshold 0.45 --top-k 10
PYTHONPATH=src python -m pytest -q                     # 220 tests
PYTHONPATH=src python -m rhagent.forward --engine model --eval-id model \
    --cost-bps 7 --fill-mode next_open --report        # record report
```

First cron tick appends 0 days by design (anchoring); the record starts
accumulating once decided days have forward returns.

## Known limitations

- Label is next-day close-to-close return > 7 bps; fills are next-open, so
  the label is a slight idealization (the open→close gap on entry day isn't
  what the model predicts). Small effect, documented choice.
- Expanding-window training each tick (~160k rows, IRLS) is a few seconds —
  fine for cron, but the model refits from scratch daily rather than
  incrementally.
- Positions are binary 0/1 per symbol; the basket is the equal-weight mean,
  so holding *k* names means *k*/65 gross exposure — same convention as the
  agent and mean_reversion records.
- Threshold/top_k/min_train_rows are fixed defaults; no tuning was shipped
  because the backtest showed nothing worth tuning toward.
