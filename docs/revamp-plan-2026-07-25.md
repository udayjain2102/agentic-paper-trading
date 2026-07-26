<!-- /autoplan restore point: /Users/adijain/.gstack/projects/udayjain2102-papertrading-model-/main-autoplan-restore-20260725-113632.md -->
# Revamp plan — 2026-07-25

Branch: `main` | Repo: `udayjain2102/papertrading-model-` | Mode: SELECTIVE EXPANSION

## Context

Stated goal (`.md/FINDINGS.md`): *build agentic trading, teach it step by step what
works and what doesn't through paper trading, have it paper-trade live markets on its
own — and eventually earn enough trust to be given real money.*

### What the 2026-07-22 audit found, and what has since been fixed

`.md/AUDIT-2026-07-22.md` is the most accurate document in the repo. PRs #36–#39
landed on 2026-07-22 and closed four of its findings. Verified against `origin/main`
and `origin/paper-state` on 2026-07-25:

| 07-22 finding | Status |
|---|---|
| Agent failure rate 87% (109 of 113 were parse-fails) | **Fixed.** 07-22: 61 ok / 4 failed. 07-23: 62 ok / 3 failed. ~5%. |
| No SPY benchmark anywhere | **Fixed.** `make_dashboard.py:96,527,762` renders SPY buy-and-hold per leg. |
| Same-close fills flatter mean-reversion | **Fixed.** `paper_cron.sh:45` ticks `mean_reversion_real` at 7bp, `next_open`. |
| No CI test gate | **Fixed.** `.github/workflows/tests.yml`. |

### What is still true

1. **The local checkout is a diverged fork.** Local `main` sits on `530357e`/`2959291`
   off `30f7d47`; `origin/main` is 10 commits ahead. The two local commits are the
   gitignored `data/` cache committed by accident, plus `docs/audit-prompt.md` and
   five `journal/runs.jsonl` lines. A `git stash` entry ("local make_dashboard
   latest-first sort") is superseded by PR #26.

2. **Failed ticks still count as decisions.** `AgentEngine.decide()` returns
   `current_pos` on exception (`engine.py`), and `forward.py` writes that day into
   `returns.csv`. At ~5% this is small but wrong: a day the model never answered is
   not a flat opinion.

3. **The daily run takes ~50 minutes.** `_agent_positions` issues one LLM call per
   symbol per undecided bar: 65 calls/day against a ~18 req/min endpoint. This is now
   a pure cost/latency problem, **not** the cause of finding 2 — the parse-fail
   epidemic that caused the contamination was fixed by #39.

4. **`scripts/make_dashboard.py` is 1004 lines** — 2.5x `papertrade.py` (401), the
   largest file in the repo, and the most-churned (26 commits in 30 days).

5. **`graphify-out/` is committed** — 132 files, 3.4 MB, 51% of 257 tracked files.

6. **Branch graveyard**: ~30 local, 12 remote, mostly merged `worktree-*`.

7. **Doc sprawl**: 26 markdown files; `FINDINGS.md` and `AUDIT-2026-07-17.md` both
   describe a broken pipeline that now works.

8. **`paper-state` is the only copy of the forward record.** No backup, and
   `deploy_api/trigger-run.js` is a deliberately unauthenticated append trigger.

### Premise (confirmed with the user, 2026-07-25)

**Premise A.** The pipeline works. The two real defects are the ~5% contamination and
the 50-minute runtime. Everything else is cleanup. This assumes the agent is worth the
plumbing — an assumption recorded here deliberately, because the counter-evidence is
real: of the agent's genuine verdicts to date, all said flat.

### Correction to an earlier draft of this plan

An earlier draft ordered "exclude failed decisions" before "batch the call" and scoped
exclusion per-symbol. That is worse than the bug. `forward.py:126` admits a day only
when all 65 legs are present:

```python
full = df.notna().sum(axis=1) == len(df.columns)
```

Excluding failed decisions per-symbol means one failure kills the whole day. At a 5%
per-symbol failure rate, `0.95^65 ≈ 3.6%` of days survive — the record silently
freezes, and on the dashboard that is indistinguishable from a broken pipeline.
Batching must land first: with one call per day, failure becomes day-granular by
construction and the exclusion is then trivially correct.

## Goals

- A day the model did not answer never enters the return series.
- The daily run finishes in minutes.
- The repo is the size of the problem it solves.
- Docs describe the system that exists.

## Non-goals

- No new strategies, overlays, gates, or dashboard panels.
- No live trading. `LIVE=true` stays off.
- Not closing the LIVE gap (the guardrail stack sits on no scheduled path). Deferred
  to TODOS, not "healthy" — see Risks.

## Plan

### Step 1 — Reconcile the checkout

- `git branch local-main-backup` **and push it** (a local-only backup on a repo whose
  state lives remotely is one `git branch -D` from gone).
- Reset local `main` to `origin/main`.
- Carry over `docs/audit-prompt.md`; drop the superseded stash.
- Verify: `git status` clean, `pytest` green (expect 215 passed).

### Step 2 — Batch the agent into one call per day

- Replace the per-symbol `decide()` loop in the forward agent path with a single
  prompt carrying a compact table of all universe symbols (last close, 5d momentum,
  20d vol, current position), returning one JSON object of `{symbol: target}`.
- **Validate the response against `cfg.strategy.universe`**: missing symbols and
  hallucinated tickers are a failed day, not a partial one.
- `max_tokens` already defers to `cfg.agent.max_tokens` (16000) since #34, so the
  batch reply has budget. Drop per-symbol `reason` to one reason for the day.
- Keep `DecisionEngine` per-symbol for the paper-trade path.
- Verify: one model call per tick regardless of universe size; a malformed/partial
  batch reply marks the day failed rather than writing partial positions.

### Step 3 — Exclude failed days from the return series

- A day whose batch call failed is excluded from `returns.csv` entirely, not recorded
  as a held position.
- Surface the excluded-day count on the dashboard so a frozen record is loud rather
  than merely empty.
- Verify: a test injecting a failing `complete()` asserts the day is absent from
  `returns.csv` and counted as excluded.

### Step 4 — Restart the agent forward track clean

- **Tag `paper-state` first** (`git tag paper-state-pre-restart-2026-07-25`) — it is
  the only copy of the record, and Step 4 mutates it.
- Move `journal/forward/agent/` to `journal/forward/agent-v1-contaminated/` and start
  `agent-v2/`. Version, don't delete: the series stays additive and the boundary is
  visible instead of being a hole.
- Record the restart reason **in the record directory**, next to the data it explains
  — not in `README.md`, the most drift-prone file in the repo.
- **Write down what does and does not force a restart**, and freeze the agent prompt
  after this. The 07-17 audit already restarted this track once; two restarts in eight
  days, with no rule, means the record can never outlive the gap between prompt edits.

### Step 5 — Shrink the dashboard

- Name the panels to cut before cutting. Collapse duplicated render paths.
- Verify: `test_dashboard.py` passes; the generated page still renders the scorecard,
  failure buckets, forward curve, SPY benchmark, and lessons text.

### Step 6 — Repo hygiene

- `git rm -r --cached graphify-out/` and gitignore it.
- Delete merged local and remote branches (several are pinned by worktrees under
  `.claude/worktrees/` and can only go once those do).
- Consolidate to one accurate `README.md`, one `ARCHITECTURE.md`, one current audit.
  Archive the superseded plans/specs and the two stale audits.

## Risks

- **Step 2 is a behavior change, not a speedup.** A cross-sectional batch ranker sees
  a different problem than 65 isolated single-name calls. Its decisions will differ.
  Step 4's versioned restart is what makes that legible.
- **Step 2 forks the agent's identity.** `learn.py` lessons and `memory.py` reflection
  derive from the per-symbol paper-trade ledger. After batching, the thing being
  taught is not the thing generating the training signal. Accepted for now and
  recorded here; revisit if the lessons text starts describing decisions the forward
  agent no longer makes.
- **Step 4 mutates the only copy of the irreplaceable asset.** The tag is the
  mitigation.
- Step 1 rewrites local git history. The pushed backup branch is the mitigation.
- **The LIVE gap is now a known blocker, not a healthy system.** `guardrails.py` /
  `executor.py` / `broker.py` are well-tested and on no scheduled path. The plan can
  complete perfectly and `LIVE=true` would still be unflippable. Deferred, written
  down, not filed under "healthy".

---

## CEO Review — 11 sections (SELECTIVE EXPANSION, 2026-07-25)

Outside voice: Codex not installed; Claude subagents used. Tagged `[codex-unavailable]`.

### Section 1 — Architecture

Live path today (`origin/main`), and the delta this plan introduces:

```
  daily-paper-run.yml (cron 17 11 * * 1-5 UTC)
     |
     +-- scripts/paper_cron.sh
           |-- git restore data/ journal/  <- origin/paper-state   [SPOF: only copy]
           |-- rhagent.refresh --fetch     -> Yahoo v8 (keyless)   [SPOF: unauth dep]
           |-- rhagent.forward             -> mean_reversion, cost 1bp, close fill
           |-- rhagent.forward             -> mean_reversion_real, 7bp, next_open
           |-- rhagent.forward --engine agent
           |     |
           |     |  BEFORE:  _agent_positions(sym) x65  -> engine.decide() x65 -> 65 LLM calls
           |     |  AFTER :  _batch_positions(universe) -> engine.decide_batch() -> 1 LLM call
           |     |
           |     +-- _net_series -> backtest.net_returns -> returns.csv
           +-- push data/ journal/ -> paper-state

  DEAD (no scheduled caller): runner.py -> executor.py -> guardrails.py -> broker.py
```

**Coupling introduced.** Batching couples all 65 symbols into one success/failure unit.
That is the point (it makes failure day-granular), but it is a real coupling change:
today one bad symbol costs 1/65 of a day, after Step 2 it costs the day. Justified only
because `_net_series` already requires full coverage — the coupling exists in the math
already; batching just makes the failure mode match it.

**Scaling.** Universe growth was previously linear in LLM calls (65 -> 100 symbols =
+35 calls). After batching it is linear in *prompt tokens*, which is far cheaper but
introduces a new ceiling: at some universe size the response exceeds `cfg.agent.max_tokens`.
Not a concern at 65; note the ceiling.

**Rollback.** Steps 2-3 are a git revert of one module plus a re-tick. Step 4 is
recoverable only via the `paper-state` tag. Step 6 (`git rm --cached`) is trivially
reversible. Reversibility overall: 4/5.

### Section 2 — Error & Rescue Map

```
  CODEPATH                      | WHAT CAN GO WRONG              | EXCEPTION
  ------------------------------|--------------------------------|------------------
  engine.decide_batch (NEW)     | model timeout                  | APITimeoutError
                                | 429 rate limit                 | RateLimitError
                                | reply truncated at max_tokens  | TruncatedResponse
                                | no JSON object in reply        | ValueError
                                | JSON missing symbols           | ValueError  <- NEW
                                | JSON has unknown tickers       | ValueError  <- NEW
                                | target not in {-1,0,1}         | ValueError
  forward._batch_positions (NEW)| batch failed -> whole day      | (handled)
  forward.tick                  | cache missing a symbol         | KeyError
  refresh._fetch_yahoo          | Yahoo 404 / schema change      | (stderr, continues)
  paper_cron.sh                 | paper-state fetch fails        | exit 1

  EXCEPTION            | RESCUED? | ACTION                        | OPERATOR SEES
  ---------------------|----------|-------------------------------|------------------
  APITimeoutError      | Y        | SDK retries x8, then fail day | "timeout:" + excluded
  RateLimitError       | Y        | SDK backoff, then fail day    | "rate-limited:" + excluded
  TruncatedResponse    | Y        | fail the day                  | "truncated:" + excluded
  ValueError (parse)   | Y        | fail the day                  | "parse-fail:" + excluded
  ValueError (symbols) | Y  NEW   | fail the day, log which syms  | "symbol-mismatch:" + excluded
  KeyError (cache)     | N  GAP   | crashes the tick              | CI red        <- pre-existing
  Yahoo per-sym fail   | N  GAP   | stderr in a green run         | nothing       <- pre-existing
```

Two GAPs are pre-existing and out of this plan's scope; both are recorded in TODOS
below. The new codepath has no unrescued error, and critically **no failure is
recorded as a decision** any more — that is the whole point of Step 3.

### Section 3 — Security & Threat Model

| # | Threat | Likelihood | Impact | Mitigated by plan? |
|---|---|---|---|---|
| S1 | `trigger-run.js` unauthenticated; shares `concurrency: paper-run` with the daily tick, so anyone with the URL can queue runs that delay/starve the daily forward tick | Med | **High** | **No** — flagged, deferred |
| S2 | Same endpoint appends to the `journal/papertrade` archive on `paper-state` | Med | Med | No — same fix |
| S3 | `paper-state` is the only copy of the record | Low | **High** | Partly — Step 4 tags it once |
| S4 | Yahoo is an unauthenticated dep with no schema validation | Med | Med | No — pre-existing |
| S5 | LLM prompt injection via symbol data | Very low | Low | N/A — inputs are numeric |

S1 is the finding worth acting on: the in-code comment claiming the blast radius is
"CI spam" is wrong, because of the shared concurrency group. Cheapest real fix is not
auth — it is moving `research-run` out of the shared group so it can never queue ahead
of the daily tick. A shared-secret header is the complete fix.

### Section 4 — Data flow & edge cases

```
  BATCH DECISION FLOW (new)
  universe+bars --> prompt --> model --> parse --> validate --> positions --> returns
       |             |          |         |           |            |            |
     [empty?]    [too long?] [timeout?] [no JSON?] [missing sym?] [NaN?]   [day excluded?]
     [<2 bars?]  [>max_tok?] [429?]     [trunc?]   [extra sym?]   [all flat?]
```

| Edge case | Handled? | How |
|---|---|---|
| Batch reply omits some symbols | **NEW — must handle** | Fail the day (Step 2 validation) |
| Batch reply invents a ticker | **NEW — must handle** | Fail the day |
| Existing `pos_*.csv` cache present when code switches to batch | **GAP** | Step 4 restart sidesteps it; must not silently half-read old cache |
| Universe changes mid-record | Pre-existing | Full-coverage rule freezes the day |
| Two runs race on `paper-state` | Handled | `concurrency: paper-run` (which is also S1) |
| Catch-up after a multi-day gap | Changed | Was N days x 65 calls; now N calls |

### Section 5 — Code quality

- Batching must not duplicate the prompt-feature computation (close/mom5/vol20) that
  `_prompt` already does. Extract once, reuse for both paths, or the DRY violation
  lands immediately.
- `AgentEngine` gains a second public method; the `DecisionEngine` protocol still has
  one. Acceptable — the protocol describes the per-bar path, batch is an additional
  capability, not a protocol change.
- Watch cyclomatic complexity in the new validate step: it has at least 5 failure
  branches. Keep it a flat sequence of guard clauses, not nested ifs.

### Section 6 — Test review

```
  NEW CODEPATHS
    engine.decide_batch: happy / timeout / truncated / no-JSON / missing-sym / extra-sym
    forward._batch_positions: day ok / day failed
    forward: excluded-day accounting

  NEW ERROR PATHS        -> one test each, per Section 2
  NEW INTEGRATIONS       -> none (same NVIDIA client)
```

| Test | Exists? | Priority |
|---|---|---|
| one model call per tick regardless of universe size | no | P1 |
| failed batch -> day absent from `returns.csv` | no | P1 |
| partial batch reply -> day failed, no partial positions written | no | P1 |
| hallucinated ticker -> day failed | no | P2 |
| excluded-day counter increments | no | P2 |
| existing per-symbol cache is not half-read after switch | no | P1 |

The 2am-Friday test is "failed batch -> day absent". The hostile-QA test is
"reply with 64 of 65 symbols and one invented ticker".

### Section 7 — Performance

Runtime 50 min -> expected < 2 min. Calls/day 65+1 -> 2. This is the plan's clearest
win and it is the one thing measurable on the next CI run. No N+1, no DB, no cache
pressure. New ceiling: response size vs `max_tokens` at large universes.

### Section 8 — Observability

| Signal | Today | After |
|---|---|---|
| per-decision status | yes (#36/#39) | yes, per day |
| excluded-day count | **no** | **yes** (Step 3) |
| runtime per stage | implicit in CI log | unchanged |
| record frozen vs empty | **indistinguishable** | distinguishable via excluded count |
| alert when record stops growing | **no** | **no** — deferred to TODOS |

The frozen-vs-empty distinction is the highest-value observability item and it is in
scope via Step 3.

### Section 9 — Deployment & rollout

No migrations, no feature flags, no staging. Rollout is: merge to `main`, next 12:44
UTC cron picks it up. Risk window is one day. Post-deploy verification: the CI log
should show one agent call, a sub-2-minute agent stage, and `returns.csv` gaining a row
(or an explicit excluded-day line). Rollback: revert + let the next tick re-run.

Deploy risk is genuinely low because the daily job is idempotent per day.

### Section 10 — Long-term trajectory

- **Debt added:** the agent's forward path and paper-trade path now decide differently
  (batch vs per-symbol). Recorded in Risks.
- **Debt removed:** contamination, 50-min runtime, 3.4 MB of unrelated files.
- **Reversibility: 4/5.**
- **The 1-year question:** a new reader in 12 months sees a versioned agent record with
  a written restart rule and an excluded-day count. That reads honestly. The prior two
  restarts were not recorded that way, which is why nobody could tell what the record
  meant.
- **Path dependency:** the LIVE gap is untouched. This plan does not move the project
  closer to `LIVE=true`; it makes the evidence trustworthy, which is a precondition,
  not progress toward the flip.

### Section 11 — Design & UX

Deferred to the design phase below (dashboard has real UI scope).

### Phase 3.5 — DX review

Product type: single-operator CLI research system. No external developers, so no
competitive benchmark was fabricated. TTHW measured at ~2 min (venv, pip, pytest).

| Dimension | Score | Note |
|---|---|---|
| Time to hello world | 8/10 | 3 commands |
| Getting-started accuracy | 9/10 | ANTHROPIC_API_KEY / phantom cron line fixed upstream |
| CLI ergonomics | 7/10 | consistent `python -m rhagent.<mod>`; `PYTHONPATH=src` papercut |
| Error messages | 8/10 | typed failure reasons (#36/#39) |
| Docs findability | **3/10** | 26 files, 3 audits, no index, 2 describe a fixed problem |
| Restart/upgrade path | 4/10 | no written rule for what forces a record restart |

Only actionable DX defect is docs findability, already covered by Step 6. Step 4's
"what forces a restart" rule fixes the second.

---

## Required outputs

### NOT in scope

| Item | Why deferred |
|---|---|
| Close the LIVE gap (put the daily path behind guardrails) | Premise C was considered and not chosen; large, and not required to make evidence trustworthy |
| Offline IC backtest of the agent prompt | Premise B considered and not chosen; would answer "is the agent worth it" |
| Authenticate `trigger-run.js` | Security finding S1; small but outside the chosen premise |
| Alert when the forward record stops growing | Real observability gap; needs a notification channel that does not exist yet |
| Yahoo schema validation | Pre-existing gap, unrelated to this plan |
| Re-unify batch and per-symbol agent identity | Consequence of Step 2; revisit only if lessons text drifts |

### What already exists (and is therefore NOT rebuilt)

| Sub-problem | Existing code | Reused? |
|---|---|---|
| Typed decision failure reasons | `engine.py` (#36/#39) | Yes — Step 3 consumes `status` |
| SPY benchmark | `make_dashboard.py:96,527,762` | Yes — must survive Step 5 |
| Realistic fills record | `paper_cron.sh:45` (`mean_reversion_real`) | Yes — untouched |
| CI test gate | `.github/workflows/tests.yml` | Yes — gates Steps 2-3 |
| Full-coverage day rule | `forward.py:126` | Yes — Step 2 makes failure match it |
| Prompt feature computation | `engine._prompt` | Yes — extract, do not duplicate |

### Dream state delta

```
  CURRENT                      THIS PLAN                  12-MONTH IDEAL
  record is honest but         record is honest,          months of unbroken
  slow and ~5% polluted   -->  fast, and versioned   -->  record + a written
  LIVE gap open                LIVE gap still open        bar it clears, behind
  1004-line dashboard          smaller dashboard          the guardrail stack
```

This plan closes the evidence-quality gap. It does not move toward `LIVE=true` — that
needs the guardrail stack on the scheduled path, which is deferred.

### Failure Modes Registry

```
  CODEPATH              | FAILURE MODE        | RESCUED | TEST | OPERATOR SEES  | LOGGED
  ----------------------|---------------------|---------|------|----------------|-------
  decide_batch          | timeout/429/trunc   | Y       | P1   | excluded day   | Y
  decide_batch          | missing/extra sym   | Y NEW   | P1   | excluded day   | Y
  _batch_positions      | day failed          | Y       | P1   | excluded count | Y
  forward.tick          | cache KeyError      | N GAP   | N    | CI red         | Y
  refresh._fetch_yahoo  | per-symbol failure  | N GAP   | N    | *nothing*      | stderr only
  record stops growing  | silent freeze       | N GAP   | N    | flat curve     | N
```

Three CRITICAL GAPs, all pre-existing and all deferred (see NOT in scope). The
record-stops-growing gap is partly mitigated by Step 3's excluded-day count, which
makes a frozen record visually distinct from an empty one.

---

## Phase 2 — Design review (dashboard)

Outside voice: Claude designer subagent `[codex-unavailable]`. All critical findings
independently verified against `origin/main` before acceptance.

### CRITICAL — the Step 4 rename chain (found by design, confirmed)

```
  Step 4 renames journal/forward/agent/ -> agent-v2/
    -> make_dashboard.py:262 still reads forward_dir / "agent"
      -> _forward_leg() finds no run.json -> {"days": 0, "pnl": 0.0}
        -> line 518: const days = agent.days || base.days || 0
           (agent.days is 0, so the BASELINE's day count satisfies the gate)
          -> line 520: days >= 5 -> agent.pnl(0.0) < base.pnl
            -> renders "BASELINE LEADS" + "Baseline forward P&L ahead of the agent"
```

A pipeline failure renders as a competitive result. Three defects compose:
1. the dashboard path is not updated with the rename,
2. `_forward_leg` returns a zero stub indistinguishable from a real flat record,
3. the verdict gate falls back to the other leg's day count.

**Plan changes.** Step 4 must update `make_dashboard.py:262` in the same commit as the
rename. Steps 3 and the empty/frozen states land BEFORE Step 4. Fix the `||` fallback
to require the agent leg's own day count.

### CRITICAL — no empty state, no frozen state

`_forward_leg` (line 75) returns `{"days": 0, "pnl": 0.0}` on a missing record, rendered
through `money()` at 34px. **A leg that never ran and a leg that ran flat are
byte-identical output.** And nothing compares `leg.end` to today, so a record frozen
since June renders like one updated this morning — the `upd` pill timestamps the render,
not the data, so a nightly rebuild of a dead record shows a fresh timestamp forever.

Fix: `present: false` flag out of `_forward_leg`; three distinct treatments for
never-ran (dash + reason) / frozen (amber border, muted number, last-tick date) / live.

### CORRECTION — Step 5's acceptance criteria were wrong

The earlier Step 5 said the page "still renders the scorecard, failure buckets, forward
curve, SPY benchmark, and lessons text." Verified against `origin/main`:

- **There is no forward equity curve.** The only chart is `_equity_curve(locked_dir)`
  (line 246) — in-sample. The one series worth watching over time has no visualization.
- **Lessons text is not rendered.** `lessons_from_runs(base_dir)` is computed and
  shipped in the payload (line 274); `DATA.lessons` has zero references.

Acceptance criteria corrected below.

### Panels to cut (the naming work Step 5 asked for)

| Panel | Lines | Verdict | Why |
|---|---|---|---|
| Engine leaderboard | 401-404, 724-736 | **Cut** (~17) | `max by pnl` per engine — the exact statistic the bake-off panel above it exists to discount |
| Per-run detail drawer | 866-940 | **Cut** (~85) | Re-renders the row, then says "ledger available from the CLI" |
| Guardrails · armed | 353-362, 647-669 | **Cut** (~40) | Hardcoded `width:100%` bars and a hardcoded "0 BREACHES"; guards an unscheduled path |
| "Where we win" | 426-431, 781-790 | **Cut** (~27) | Ranked by `win_share` — a size chart labelled a skill chart |
| Dead `lessons` payload | 274 | **Cut** | Computed, never rendered |
| Overlay bake-off | 364-382 | Collapse | Changes monthly; belongs under "why this config is locked" |
| Runs table + scorecard | 385-416 | Collapse | Trim 13 tiles to 5 |

~380-400 lines, landing near 600, before any render-path collapsing.

### Revised target order

1. Header pills (add data freshness, distinct from render time)
2. Verdict — with present/frozen states and the coverage bar
3. **Forward equity curve — agent vs baseline vs SPY, restart boundary annotated** (new)
4. Loss buckets (keep the caveat box)
5. Runbook + reflections
6. `<details>` Research provenance — everything in-sample

### Excluded-day count — placement

Not a panel (respects the no-new-panels non-goal). It goes in `sub(leg)` (line 526) as a
denominator on the number it qualifies: `12 of 15 sessions scored`, plus a 3px
green/amber coverage bar under the agent card. Escalation: <5% grey, 5-20% amber,
>20% or 3 consecutive excluded -> `RECORD DEGRADED`. Drops the unreadable 65-ticker
list from `sub()`, so no net pixels added.

### Accessibility

`.cr-btn` sets `border:none` with no focus style anywhere in the stylesheet (289-304) —
the page is not keyboard-navigable. `:focus-visible{outline:2px solid var(--accent)}` is
one line and the highest-value fix. Sort headers and run rows are click-handled `<div>`s
with no role and no keyboard activation. 10.5px monospace body text at 665/776/788.

### Revised Step 5 acceptance criteria

`test_dashboard.py` passes; the page renders the verdict card with correct empty/frozen
states, a forward equity curve including SPY, loss buckets, and the coverage/excluded
strip. Lessons text either gets rendered or its payload gets deleted — not left dead.

---

## Phase 3 — Eng review

Outside voice: Claude engineer subagent `[codex-unavailable]`. All critical findings
verified against `origin/main` before acceptance.

### CRITICAL C1 — Step 3 as written is mathematically wrong. REDESIGNED.

`backtest.py:72` compounds `equity = (1.0 + net).cumprod()`; `:76` annualizes with
`sqrt(252)`. Deleting days from `returns.csv` means the product of remaining days no
longer equals the account, and Sharpe is annualized over a series with holes.

And the contamination survives the deletion regardless: `backtest.py:46`
`turnover = pos.diff().abs()`, so day D+1's cost is charged against `pos[D]` — the
phantom position held on the day the model never answered.

The plan conflated two different questions:
- *What did the paper account earn?* -> the failed day **must** be included; it held.
- *How good are the agent's decisions?* -> the failed day must be excluded.

**Redesign.** Do not mutate the return series. Keep `returns.csv` as the honest
as-traded account curve, add a `status` column marking non-decision days, and report
two numbers: as-traded return and decision-coverage. This requires **no change to
`backtest.py` math at all** — it is strictly less code than the original Step 3.

### CRITICAL C2 — the obvious implementation is a silent no-op that passes CI

`backtest.py:44`: `pos = positions.reindex(close.index).fillna(0).astype(float)`.
A NaN "excluded" sentinel becomes a genuine flat position; `net_returns` then returns
`net[fwd.notna()].fillna(0.0)`, so no NaN ever reaches `forward.py:126`, full coverage
is satisfied, and the day is admitted. **The fix would appear to work while changing
nothing.** Every agent forward test uses a one-symbol universe (`test_memory.py:108`,
`test_forward_execution_model.py:25`), so this ships green.

Exclusion must be an explicit set of failed dates carried alongside the positions, never
encoded as NaN inside a position series.

### CRITICAL C3 — Step 2 silently kills the memory loop

`memory.py:47` `for f in sorted(eval_dir.glob("pos_*.csv"))`. Batching away the
per-symbol files empties `recent_outcomes`, `reflect` receives
`"(no decisions to review)"` (`memory.py:99`), the model writes a confident reflection
about nothing, and `run.json` records `reflected: true`. **A green run that has stopped
learning.**

Fix: keep writing `pos_<sym>.csv` from the batch path. The storage layout is an internal
contract; batching is about call count, not storage. Cheapest fix, no `memory.py` change.

### CRITICAL C4 — Step 4's rename races the cron and misses a hardcoded path

`paper_cron.sh:52` hardcodes `--eval-id agent`. Rename without editing it and the next
tick recreates an empty `journal/forward/agent/`, hits the fresh-record branch at
`forward.py:157`, and anchors a new one-day record that the dashboard renders as the
agent's track.

Worse, `journal/` lives only on `paper-state`, and `paper_cron.sh:62-64` does
`rsync -a --delete journal/` back onto it. A rename landing mid-run is deleted and the
old directory resurrected. **Disable the schedule for the rename.** The `paper-state`
tag protects git history, which was never at risk — the bot never force-pushes.

### CRITICAL C5 — the restart does not restart the memory

`memory.py:16` `DEFAULT_PATH = "journal/agent_memory.md"` sits outside the record
directory. Renaming `forward/agent/` carries up to 40 v1 reflections — written from
decisions the batch agent will never make — straight into v2. Version the memory file
alongside the record, or state plainly in the restart note that memory carries over.

### PRE-FLIGHT (blocks Step 2) — the 45s timeout may not fit a batch reply

`engine.py:49` sets `timeout=45` with `max_retries` unset (SDK default), while
`engine.py:145-153` documents a 16000-token call at 60-120s. A 65-row table in and 65
JSON entries out is a far longer generation than a 2-field verdict. **If the batch call
exceeds 45s, Step 2 fails 100% on day one.** Measure one real batch call before writing
any of Step 2; raise the timeout and add one explicit retry.

### W1 — the stated verification criterion is false

`_agent_positions` (`forward.py:52`) loops over every *uncached bar*, not just today's.
After a weekend, CI outage, or missing `NVIDIA_API_KEY`, N undecided bars means N batch
calls. The verifiable claim is "one call per undecided bar", not "one call per tick".

### W4 — test gaps (all P1)

- Multi-symbol agent forward coverage: **zero today**. This is why C2 ships green.
- `test_memory.py:171` is the only test of the "cached bars must not re-call the model"
  contract, and it calls `_agent_positions` directly — the function Step 2 deletes. That
  property must survive the rewrite.
- `forward._selfcheck` is behind `if sys.argv[1:2] == ["selfcheck"]` and CI runs pytest
  only, so its agent-path assertions have **never run in CI**. Free win: make it a test.
- Missing: partial reply fails the day; hallucinated ticker fails the day; scored days
  plus excluded days equals trading days elapsed (the invariant that catches C1 drift).

### W5 — `research-run.yml` has no test gate

`daily-paper-run.yml:38-41` runs pytest before touching state. The
anonymously-triggerable `research-run.yml` does not — whatever is on `main` runs against
`paper-state` unvalidated. Give it the pytest gate and its own concurrency group.

### Revised step order

```
  0. PRE-FLIGHT: measure one batch call against the 45s timeout   <- blocks Step 2
  1. Reconcile checkout
  2. Batch the agent (keep pos_*.csv writes for memory.py)
  3. Add status column + coverage reporting (do NOT delete days)
  3b. Dashboard: empty/frozen states, coverage strip, agent path
  4. Rename record + memory + paper_cron.sh:52 + dashboard:262, schedule DISABLED
  5. Shrink dashboard
  6. Hygiene
```

---

## Final gate decisions (2026-07-25)

### User challenge — resolved: IC check added as a gate

Both reviewers independently argued Premise A's assumption (the agent is worth the
plumbing) is unsupported: every genuine verdict to date said flat, the self-written
memory is content-free, and `learn.py:17` reads lessons from `journal/papertrade/*`,
which `research-run.yml:40` populates with **mean_reversion** runs — so part of what the
agent is "taught" is the baseline strategy's results, not its own.

**Decision: add T0.5 and let it gate T2-T5.** Replay the batch prompt over ~280 cached
days offline and measure the information coefficient of its cross-sectional positions
against forward returns. No rate limit applies to a backfill you are willing to wait on.

- IC distinguishable from zero -> proceed with T2-T5 as planned.
- IC indistinguishable from zero -> the honest conclusion is that the agent is an open
  research question and the rule is what gets funded. T2-T5 become optional, and the
  project stops paying for 65 daily calls that produce no signal.

This is the one item that can invalidate the rest of the plan, so it runs first.

### Scope — forward equity curve accepted (net-negative on panels)

The "no new dashboard panels" non-goal is relaxed for exactly one addition: a forward
equity curve (agent vs baseline vs SPY, restart boundary annotated). It replaces the
in-sample curve in the primary position. Net panel count still falls, because the
leaderboard, per-run drawer, guardrails panel and win-buckets are all cut.

### Security — research-run hardening accepted (partial)

Accepted: own concurrency group, plus the pytest gate `daily-paper-run.yml:38-41`
already has. Not accepted for now: the shared-secret header on `trigger-run.js`.
This removes the starvation path and the untested-write path; the endpoint stays open.

### Final step order

```
  T0    PRE-FLIGHT   measure a batch call against the 45s timeout
  T0.5  GATE         offline IC of the batch prompt over ~280 cached days
                     |- no signal -> STOP, re-scope, fund the rule
                     |- signal    -> continue
  T1    reconcile the checkout
  T2    batch the agent (keep pos_*.csv for memory.py)
  T3    status column + coverage reporting (no day deletion)
  T3b   dashboard: empty/frozen states, coverage strip, verdict-gate fix
  T4    rename record + memory + paper_cron.sh:52 + make_dashboard.py:262
        (schedule DISABLED for this step)
  T5    multi-symbol + partial-reply + invariant tests
  T6    research-run: own concurrency group + pytest gate
  T7    dashboard cut (~390 lines) + forward equity curve
  T8    hygiene: graphify-out, branches, docs
```

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | issues_open | 7 proposals, 4 accepted, 6 deferred |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | codex not installed |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | issues_open | 11 issues, 5 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | issues_open | score 4/10 → 7/10, 8 decisions |
| DX Review | `/plan-devex-review` | Developer experience gaps | 1 | clean | score 6/10 → 7/10, TTHW 2min |

- **CROSS-MODEL:** Codex unavailable in all four phases; every outside voice was a Claude
  subagent, tagged `[codex-unavailable]`. Cross-model consensus was therefore NOT
  available — findings rest on single-model review plus direct verification against
  `origin/main`. Four subagent claims were refuted by that verification (`max_tokens=256`,
  no CI test gate, README drift, `max_retries=8`), which is the argument for verifying
  rather than accepting.
- **VERDICT:** CEO + ENG + DESIGN + DX reviewed; plan revised and approved with 3
  overrides. Not cleared to ship — no code has been written; T0 and T0.5 gate all
  implementation.

**UNRESOLVED DECISIONS:**
- Shared-secret header on `trigger-run.js` — deliberately deferred; endpoint stays open
- Whether the batch and per-symbol agent identities get re-unified — revisit only if the lessons text drifts
- What pre-committed bar the forward record must clear to earn `LIVE=true` — still unwritten, and no step in this plan writes it

---

## GATE RESULTS — 2026-07-26

### T0: batching REJECTED by measurement

The plan's own pre-flight condition fired. A 65-symbol batch call never returned:
one clean `APITimeoutError` at 136.5s (SDK retries x 45s), one manual kill at 353s,
one no-return at a 300s cap. Endpoint healthy (trivial prompt: 10.5s), harness correct
("detailed thinking off" present, `max_tokens=16000`, `temperature=0`).

Scaling test (N = 1, 5, 10, 20, 40, `timeout=120`, `max_retries=0`):

| N | median latency | median completion_tokens | reliable under 45s? |
|---|---|---|---|
| 1 | 7.4s | 146 | **yes** (5.25s / 9.45s) |
| 5 | 61.6s | 1536 | no (90.1s / 33.2s) |
| 10 | 40.6s | 1044 | no (53.6s / 27.6s) |
| 20 | 40.6s | 1220 | no (31.6s / 49.5s) |
| 40 | timeout | — | no |

No symbol hallucination in any successful run; coverage was always N/N.

**Root cause: response verbosity, not batch size.** Completion tokens cluster at
1000-2000 for N=5, 10 and 20 alike — a 20-symbol ask costs about the same as a
5-symbol ask. Variance at fixed N (90s vs 33s at N=5) exceeds variance across N. The
model emits 1-2k tokens of rambling per call despite "detailed thinking off", and
`max_tokens=16000` leaves room for it. Chunking cannot fix this: no chunk size in
5-20 is reliable, and N=1 is 65 calls/day, i.e. the status quo.

**Consequence:** the branch `fix/agent-batch-decide` (one batched call per bar, 215
tests passing but every model call mocked) is built on a configuration that measures
~50% timeout. Do not merge it without re-measuring.

### The real cause of the 50-minute run

```
  measured N=1 latency        7.4s median, reliable
  65 serial calls x 7.4s    = ~8 min
  observed daily runtime    = ~50 min
                              -------
  unexplained               = ~42 min retry burn / rate-limit backoff
```

`git grep ThreadPool|asyncio|concurrent.futures` over `src/` returns nothing — every
call is strictly serial. At the endpoint's ~18 req/min ceiling, 65 calls pipelined is a
~4 minute floor.

### DECISION (user, 2026-07-26): Step 2 replaced

**Step 2 is now bounded concurrency, not batching.** Run the existing per-symbol calls
through a worker pool capped at the rate limit. Keeps the only configuration measured
reliable, and dissolves three risks the review raised: no identity fork between the
forward and paper-trade paths, no `memory.py:47` glob breakage, no "behavior change,
not just a speedup". Strictly smaller diff than batching.

Design constraint for the implementation: bars within a symbol are sequentially
dependent (`pos` feeds the next bar's `current_pos`); symbols are independent.
Parallelize across symbols only.

### T0.5 IC gate: NOT RUN, still open

0 of 40-60 target replay days completed — at 130s+ per failed batch call a sequential
replay is hours. The replay script exists and is unexecuted. **Deferred until after the
runtime fix**, at which point a 50-day x 65-symbol replay drops from ~65 hours to
roughly 3-5 hours as a background job.

The question the gate exists to answer — does this agent produce anything other than
flat — remains unanswered.
