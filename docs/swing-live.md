# Swing live trading bridge

`src/rhagent/swing_trade.py` turns a forward record's position targets into
real orders through the existing guardrailed `OrderExecutor`. It exists so
`mean_reversion_real` — the only honest positive record — can be deployed in
swing format (positions held until the target changes, not daily-rebalanced).

## How it works

1. **Targets.** If the record dir has `pos_*.csv` (agent/model engines), the
   latest `status == "ok"` row per symbol is the target. Rule-based records
   (mean_reversion) don't freeze positions, so targets are recomputed exactly
   the way `forward.py` does: `get_bars` on the 400-day cache →
   `Strategy.positions` → conviction overlay → last bar date.
2. **Diff.** `plan_orders` compares targets against `broker.get_account()`
   position values. Only deltas above `--min-trade-usd` (default $5) become
   orders. Names the broker holds that the record doesn't want are sold in
   full. A target of -1 means flat — v1 is long-only, never opens shorts.
3. **Sizing.** `min(total_deployed_max_usd, buying_power) / n_long_names`
   per name, clipped to `per_trade_max_usd`. With the default config
   ($2,000 deployed cap) and ~15 held names that's ~$133/name.
4. **Execution.** Every order goes through `OrderExecutor.execute` →
   `validate_order` → journal (`journal/swing_trades.jsonl`). Nothing can
   bypass the guardrails.

## Usage

```bash
# dry-run (MockBroker, logs intended orders only)
PYTHONPATH=src python -m rhagent.swing_trade --eval-id mean_reversion_real --dry-run

# live — requires OAuth bearer token env vars
ROBINHOOD_MCP_URL=... ROBINHOOD_MCP_TOKEN=... \
PYTHONPATH=src python -m rhagent.swing_trade --eval-id mean_reversion_real --live
```

## Known gaps before real money

- **`McpBroker` tool names are unverified** against the live Robinhood MCP
  (`list_tools` output needed). `get_account`/`get_quote`/`place_order`
  mappings are best-guess.
- **Token path.** The MCP is OAuth-only inside an interactive client
  session; a static bearer token for headless runs may not exist. See
  `docs/going-live.md`.
- **Gradual fill.** `max_new_positions_per_run: 2` means a fresh book fills
  over several runs — intentional, but raise it in config if you want faster
  deployment.
- **Per-trade clip.** Orders are clipped to `per_trade_max_usd` ($250), so a
  concentrated target may under-fill until the cap is raised or positions
  accumulate across runs.
- **No PDT check.** Swing holding periods mostly avoid the pattern-day-trader
  rule, but verify account type before sizing.
