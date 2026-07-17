# Going live: the runbook

This is the checklist for the day the forward paper record has earned enough
trust to risk real money. Nothing here should be new — it's a restatement of
decisions already made about this project (see the dashboard/forward-record
docs) and the guardrails already in code — collected in one place so going
live is a checklist, not a design discussion.

**Do not start this checklist until the bar below is met.** If you're reading
this because you're excited, not because the bar is met, stop and wait.

## 1. The trust bar (must be true before step 2)

- The forward paper record (`journal/forward/<engine>/returns.csv`, as shown
  on the dashboard) has run **unbroken for months**, not weeks — a short hot
  streak is noise, not evidence.
- It clears whatever Sharpe/drawdown/win-rate bar was set when the record
  started (see the dashboard scorecard, not just total return — a smooth
  losing curve and a lucky spiky one both fail this).
- No forced restarts of the record (a fresh `journal/forward/<engine>/` dir
  resets the clock — check the `run.json` `start` date is genuinely old).
- You've read every `journal/runs.jsonl` / forward `run.json` entry that looks
  unusual, not just the summary stats.

If any of these is false, the answer is "keep paper trading," not "go live
with a smaller amount." Small real money still means real money.

## 2. Funding and account prerequisites

- [ ] Robinhood account funded with only the amount you're prepared to lose —
      guardrails cap *order* size, not total account risk from a strategy that
      turns out to be wrong in a way backtesting didn't catch.
- [ ] You understand PDT (pattern day trader) rules if the account is under
      $25k and the strategy trades frequently enough to trip them — this system
      does not check PDT status for you.
- [ ] You've read Robinhood's API terms of service for automated trading.
- [ ] Taxes: every filled order is a taxable event; the system journals fills
      but does not do tax accounting.

## 3. Authenticate the Robinhood MCP for real trading

The MCP is OAuth-only (see [`paper-cron-setup.md`](paper-cron-setup.md)) —
there is no static token to mint, so the *live* order path can only run
inside an authenticated session:

- **Interactive**: in a `claude` session in this repo, run `/mcp`, pick
  `robinhood-trading` (or the connector name shown), complete OAuth. Confirm
  `mcp__robinhood-trading__*` tools are live before proceeding.
- **Cloud routine**: attach the `robinhood` claude.ai connector
  (`claude.ai/customize/connectors`) to a scheduled routine
  (`claude.ai/code/routines`). As of 2026-07-16 this connector was not yet
  appearing as routine-eligible (only auth tools exposed) — re-check this
  before relying on it; if still broken, live ticks must run from an
  interactive session, same as the historical paper loop.

Either way: **confirm the MCP's actual read/order tool names** against its
`list_tools` output and check they still match the mapping in
`src/rhagent/broker.py`'s `McpBroker` — that mapping was written against a
snapshot of the API and may have drifted.

## 4. Flip the switch — smallest possible blast radius first

1. In `config.yaml`, **lower the limits further than your final target** for
   the first live session:
   - `per_trade_max_usd`: start at the smallest amount that isn't a rounding
     error (e.g. $25–50), not the default $250.
   - `max_orders_per_run`: 1, not 5.
   - `total_deployed_max_usd`: enough for a couple of trades, not the full
     book.
2. Set `LIVE=true` in `.env` (or the environment the runner/routine actually
   reads it from — confirm this, don't assume).
3. Run **one manual tick** (`python -m rhagent.runner`, or `STRATEGY_MODE=true`
   for the locked-in strategy) with a human watching, not on a schedule yet.
   Confirm in the Robinhood app that the order shown in `journal/runs.jsonl`
   is the order that actually appears on the account.
4. Only after that one tick matches expectations, put it back on a schedule
   (cron, GitHub Action, or routine) — and watch the first several scheduled
   runs closely before treating it as unattended.

## 5. Standing safety checks (apply forever after going live, not just day 1)

- `touch HALT` in the project root stops all trading on the next run —
  know this before you need it, and check it actually works (place a HALT
  file, run a tick, confirm it aborts) as part of step 4's manual test.
- The daily realized-loss kill switch (`daily_loss_limit_usd` in
  `config.yaml`) halts the rest of the day once tripped — set this to an
  amount you'd actually notice, and confirm it's still sized in dollars you
  care about now that you're live (it was set with paper P&L in mind).
- Re-run this checklist's step 1 bar-check periodically, not just once — a
  strategy that clears the bar once can still degrade; live P&L is the new
  ground truth, and a live losing streak past the same bar that governed
  paper trading is a reason to flip `LIVE=false` again, not raise the bar.

## What NOT to do

- Don't raise `per_trade_max_usd`/`total_deployed_max_usd` back to "normal"
  until the smallest-blast-radius period (step 4) has itself run long enough
  to trust — treat it as a second, shorter version of the step-1 bar.
- Don't skip the manual-tick step because the paper loop "already proved
  itself" — paper trading never touched a real broker's order API, real
  slippage, or real fills; the manual tick is the first time those exist.
- Don't add new strategies, overlays, or universe symbols at the same time as
  going live — isolate the "is real money different from paper" question from
  "is this new idea good."
