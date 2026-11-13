# Experiments

Research runs that are not part of the daily paper run. Every run is saved so
later runs can be compared against earlier ones on the same days.

- `results.jsonl` — one row per recorded variant (Sharpe, CI, return, drawdown, config, git sha).
- `runs/<run_id>/returns.csv` — that variant's full daily return series.
- `ledger.py` — `record()`, `compare()` (paired bootstrap of the Sharpe difference), and a table printer.

```bash
uv run --with duckdb python experiments/fetch_prices.py data_long
PYTHONPATH=src uv run --with-requirements requirements.txt --with scikit-learn python experiments/trade_filter.py data_long
PYTHONPATH=src uv run --with-requirements requirements.txt --with scikit-learn python experiments/trade_filter_split.py data_long
PYTHONPATH=src uv run --with-requirements requirements.txt --with duckdb --with scikit-learn python experiments/other_markets.py data_long
PYTHONPATH=src python experiments/ledger.py
```

Rule: fix the pass/fail bar before the first run, record the result either way,
and never delete a row.

## Log

| Date | Experiment | Bar | Verdict |
|---|---|---|---|
| 2026-11-13 | `trade_filter`: gradient-boosted classifier skips mean-reversion entries it scores below the base win rate; walk-forward by year, 2003 to 2025-05 | Beat the unfiltered rule's Sharpe at 7bp/next-open, CI of the difference above zero | **Failed.** Sharpe 0.43 vs 0.54 unfiltered (diff -0.11, CI -0.25 to +0.03). The model ranks trades slightly correctly (kept: 75.0% wins, skipped: 73.3%), but skipped trades were still profitable, so skipping them only removed return. |
| 2026-11-13 | `trade_filter_split`: same filter, trained on 1995-2009 then fine-tuned on later years; tested on held-out 2020 to 2025-05 (10-year fine-tune) and 2025 onward (15-year fine-tune) | Beat the unfiltered rule's Sharpe, CI of the difference above zero | **Failed.** Held-out 2020 to 2025-05: 0.30 fine-tuned vs 0.36 unfiltered (diff -0.06, CI -0.34 to +0.20). Fine-tuning helped the model a little relative to leaving it un-tuned (+0.06, CI -0.34 to +0.43), not enough to matter. |
| 2026-11-13 | `other_markets`: the locked rule, nothing re-tuned, on 200 other US stocks and 60 Hong Kong stocks, 2006 to 2025-05 | None set: this measures how much of the edge is specific to the 65 chosen names | Unfiltered rule: Sharpe 0.38 (CI -0.03 to 0.81) on other US stocks, 0.23 in Hong Kong at 7bp and 0.09 at 20bp. Conviction overlay: 0.16 on other US stocks and negative in Hong Kong. The edge is weaker off the chosen universe and the overlay does not carry over. A handful of unadjusted spin-off days are in the data. |
