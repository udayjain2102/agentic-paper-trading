# Findings — questioning everything against the original goal

**Date:** 2026-07-16
**The original goal, in your words:** build agentic trading, teach it step by step
what works and what doesn't through paper trading, have it paper-trade live
markets on its own — and eventually earn enough trust to be given real money.

Everything below is judged against that sentence and nothing else.

---

## The one-paragraph verdict

The project quietly turned from *"teach an agent through paper trading"* into
*"build a quant research pipeline."* The agent barely learns (it gets one
sentence of feedback per run), the "on its own" part is **broken right now**
(the daily GitHub Actions run has failed every day since it was set up), and
your original grading metrics were never deleted — they were **buried** under
newer panels on a dashboard that now leads with things that don't serve the
goal. The fix is not more features; it is repairing one pipeline, deleting
about a third of the surface area, and putting the grading back on top.

---

## Finding 1 — The thing that's supposed to learn isn't the thing learning

The repo has two brains:

- **The LLM agent** (`agent.py`, `engine.py` AgentEngine, Nemotron via NVIDIA's
  API). This is the "agentic" part of the original goal.
- **The quant pipeline** (`strategies/`, `search/`, `gate/`, `overlay.py`,
  `evaluate_robust.py`). Rule-based mean-reversion with statistical gates.

All of the real learning machinery — parameter search, out-of-sample gates,
loss-bucket overlays, the robust bake-off — lives on the **quant** side. The
locked-in "winner" in `config.yaml` is plain mean-reversion plus a conviction
gate: **no LLM anywhere in it.**

The agent's entire step-by-step education is one string: `learn.py` condenses
all prior paper trades into a single sentence ("losses concentrate in
vol_regime=high…") and prepends it to the prompt. That's the whole feedback
loop. There is no record of *which* lessons were fed to *which* run, so you
can't even see whether the agent improves run over run — which was the point.

**Question to answer:** is the goal to teach the *agent*, or to find a
tradeable quant signal? They are different projects. Right now you're 80%
invested in the second while describing the first.

## Finding 2 — "Paper trade live markets on its own" is broken today

- The `daily-paper-run` GitHub Actions workflow **failed on 2026-07-14 and
  2026-07-15** (and will keep failing): the `paper-state` branch it stores the
  cache/record on was **never seeded**, and the `ROBINHOOD_MCP_URL` /
  `ROBINHOOD_MCP_TOKEN` secrets are **empty**. It dies in 25 seconds at
  "paper-state not found on origin". One-time setup in
  `md/paper-cron-setup.md` was never completed.
- The `mean_reversion` forward record ends **2026-07-13** — it's already stale.
- The `agent` and `pairs` forward records contain **one day each**
  (2026-07-09) and have been dead since. They're corpses on the dashboard.

So the single most goal-critical artifact — the genuine out-of-sample forward
track record, the thing every doc says the promotion decision waits on — is
not growing. Every day it stays broken is out-of-sample evidence you never
get back.

This matters double because of the end goal: **trust for real money is
earned by exactly one artifact — an unbroken, honest, live forward record.**
Backtests can't earn it (they're history you already peeked at), and neither
can more features. A 6–12 month green forward track is the only currency that
buys the `LIVE=true` flip, and the clock on it isn't running.

## Finding 3 — Your original grading metrics didn't disappear; they got buried

The original grading was two things, and both still exist in `evaluate.py`:

1. **The trade scorecard** — win rate, avg win/loss, profit factor, total
   return, Sharpe, max drawdown, avg holding period.
2. **Failure buckets** — where losses concentrate (vol regime, gap direction,
   holding length, symbol).

They still render per run on the dashboard. But the page now *leads* with:
overview cards → forward table → "research pulse" → all-runs table → bake-off
panel → **code graph** (?!) → and only then run details, which your uncommitted
change additionally collapses into closed `<details>` folds. The grading you
built the project around is below four newer panels and behind a click.

Also: the *signal-level* grading (IC, ICIR, half-life — the whole §2 of
ARCHITECTURE.md) appears **nowhere** on the dashboard. It only exists as CLI
output from `factor`, `search`, and `gate`. If you remember grading you can no
longer find, this is it.

And the two grading systems disagree about what "good" means: the search/gate
path ranks by **ICIR** and its real-data verdict was *viable = 0*; the bake-off
path ranks by **robust Sharpe** and crowned conviction-gated mean-reversion.
The crowned winner never passed the ICIR gates — it went around them. Two
judges, two rulings, and the config comment admits the winner's 95% CI still
spans zero. **Pick one grading system** (recommendation: the trade-level one —
scorecard + failure buckets + robust Sharpe — because it's the one the forward
record can confirm) and demote the other to an offline research tool.

## Finding 4 — Feature-over-feature: what each piece actually earns

Judged against the goal. "Keep" means it directly serves teach-by-paper-trading.

| Piece | Verdict | Why |
|---|---|---|
| `guardrails.py`, `executor.py`, `broker.py`, `runner.py`, `journal.py` | **Keep** | The safety funnel. Small, tested, done. Stop touching it. |
| `papertrade.py` + `evaluate.py` + `learn.py` | **Keep — this IS the goal** | Bar-by-bar paper trading, scorecard, failure buckets, lessons. The original loop. |
| `forward.py` + `refresh.py` + CI workflow | **Keep, but FIX (Finding 2)** | The "on its own" part. Currently broken. |
| `strategies/` + `backtest.py` + `compare.py` | Keep | The baseline the agent is compared against. Cheap. |
| `engine.py` AgentEngine | Keep, underused | The agent path exists but nothing exercises it on a schedule. |
| `search/` + `gate/` + `factor/` (~600 lines) | **Question** | Serious machinery (Bonferroni, deflated Sharpe, IC decay) whose one real-data run said *nothing is viable*, and which the current winner bypassed. It's a second, competing grading system. Park it or accept it as offline-research-only; stop letting it define "the math" of the project. |
| `overlay.py` — 3 overlays + identity | Keep 1, park 2 | Conviction won and is wired forward. BucketFilter and WinProbGate lost the bake-off (WinProbGate is *inert at default settings* by design). They're now maintenance surface. |
| `evaluate_robust.py` | Keep | It's the honest judge of the bake-off; small. |
| `scripts/make_dashboard.py` (692 lines) | **Shrink** | The single biggest file in the repo — bigger than the paper-trade engine it reports on. Two near-duplicate render paths (`_run_section` vs `_run_detail`, ~90% identical; `render` vs `render_all`). A "code graph" panel that has nothing to do with trading. The dashboard grew features the way the repo did. |
| `journal/forward/agent`, `journal/forward/pairs` | **Delete or restart deliberately** | One-day dead records polluting the "Now" table. |
| Stale worktree branches (5 local `worktree-*` / feature branches) | Delete | Merged or abandoned. |
| README vs reality | Fix | README says "Claude via the Anthropic API"; the actual agent is Nemotron on NVIDIA's API. `config.yaml` comment says overlay wiring into the forward path "is a follow-up" — it shipped in commit `ee2f7ef`. Docs describing a system that doesn't exist is how you got confused. |

## Finding 5 — What "learning step by step" would actually require (and doesn't exist)

Today, honestly: the agent gets one sentence of lessons, there is no scheduled
agent paper-run, and no artifact shows learning over time. To meet the original
goal you need exactly three things, all small:

1. **A cadence** — the fixed CI run (or local cron) paper-trades the agent
   regularly, so the ledger and lessons actually accumulate.
2. **A memory** — each agent run records the lessons text it was given
   (one field in `run.json`). Now "step by step" is auditable.
3. **A report card over time** — one dashboard line/chart: agent win-rate and
   net P&L per run, in run order, next to the mean-reversion baseline. That
   single chart *is* the project: "is the agent getting better than the rule?"

None of these are new systems. All three plug into code that already exists.

## The refocus plan, in order

1. **Fix the daily run** (highest value, ~30 min): seed the `paper-state`
   branch per `md/paper-cron-setup.md`, set the two repo secrets. Until this
   is green, nothing else matters — the forward record is the project's only
   irreplaceable asset.
2. **Delete the dead forward tracks** (`agent`, `pairs` one-day records) or
   restart them on the fixed cadence, deliberately.
3. **Reorder the dashboard around the goal**: top = forward equity curve +
   scorecard + failure buckets + current lessons text; then the all-runs table;
   bake-off below; code-graph panel gone. Collapse the two duplicate render
   paths into one while there.
4. **Pick the one grading system** (trade-level scorecard/buckets/robust
   Sharpe). Mark `search/`, `gate/`, `factor/` as offline research tools in the
   docs, or park them on a branch.
5. **Close the agent loop**: schedule agent paper-runs, stamp lessons into
   `run.json`, add the agent-vs-baseline-over-time chart. This is the original
   goal, and it's ~3 small changes away.
6. **Freeze feature intake** until 1–5 are done. Every new overlay, panel, or
   gate so far has moved the project *away* from the sentence at the top of
   this file.

## The road to real money (so the endpoint is written down)

Trust is a ladder, and each rung is evidence, not code:

1. **Forward record growing daily, unattended** (fix in step 1 above) — the
   system runs itself and the numbers are honest.
2. **The record clears its own bar** — the forward Sharpe/drawdown confirms
   what the bake-off claimed, over months, not days. Define the bar *now*,
   before the data exists (e.g. "3+ months, positive net of costs, max DD
   within the backtest's, CI lower bound above zero") so future-you can't
   move the goalposts.
3. **The agent beats or matches the rule baseline** on the same forward
   cadence (step 5 above) — otherwise the honest conclusion is to fund the
   rule, not the agent.
4. **Small real money behind the existing guardrails** — the caps in
   `config.yaml` ($250/trade, $2k deployed, $200/day kill switch) were built
   for exactly this moment; tighten them, flip `LIVE=true`, and treat the
   first live month as one more evaluation gate, not a finish line.

Nothing on this ladder is a new feature. It is all waiting on the pipeline
fixed in step 1 and on time passing.

## What NOT to build next (things the current trajectory suggests)

- No fourth overlay, no ParamTune, until the forward record is growing again.
- No synthetic market / world model (§5 of ARCHITECTURE) — the seams exist,
  leave them empty; you don't have enough real forward data to validate a fake
  market against.
- No new dashboard sections. The dashboard needs subtraction, not addition.
- No live trading (`LIVE=true`). The config itself says the winner's CI spans
  zero. The forward record — once fixed — is the only thing that can earn that
  flip.
