# Experiments

Research runs that are not part of the daily paper run. Every run is saved so
later runs can be compared against earlier ones on the same days.

- `results.jsonl` — one row per recorded variant (Sharpe, CI, return, drawdown, config, git sha).
- `runs/<run_id>/returns.csv` — that variant's full daily return series.
- `ledger.py` — `record()`, `compare()` (paired bootstrap of the Sharpe difference), and a table printer.

```bash
uv run --with duckdb python experiments/fetch_prices.py data_long
PYTHONPATH=src uv run --with scikit-learn --with pandas python experiments/trade_filter.py data_long
PYTHONPATH=src python experiments/ledger.py
```

Rule: fix the pass/fail bar before the first run, record the result either way,
and never delete a row.

## Log

| Date | Experiment | Bar | Verdict |
|---|---|---|---|
| 2026-11-13 | `trade_filter`: gradient-boosted classifier skips mean-reversion entries it scores below the base win rate; walk-forward by year, 2003 to 2025-05 | Beat the unfiltered rule's Sharpe at 7bp/next-open, CI of the difference above zero | **Failed.** Sharpe 0.43 vs 0.54 unfiltered (diff -0.11, CI -0.25 to +0.03). The model ranks trades slightly correctly (kept: 75.0% wins, skipped: 73.3%), but skipped trades were still profitable, so skipping them only removed return. |
