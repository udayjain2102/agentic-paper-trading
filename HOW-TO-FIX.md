# How to fix — the companion to FINDINGS.md

Concrete steps, in order. Each fix says what to do, how long it takes, and how
to verify it worked. Do them in this order — Fix 1 is the only urgent one;
everything else can happen at leisure once the record is growing.

---

## Fix 1 — Get the daily paper-run green (~30 min, do this first)

The workflow fails at "paper-state not found on origin" and both MCP secrets
are empty. Two one-time setup steps were never done; they're documented in
`docs/paper-cron-setup.md`. Condensed:

**1a. Add the two repo secrets.**
GitHub → repo **Settings → Secrets and variables → Actions**:

- `ROBINHOOD_MCP_URL` — the hosted Robinhood MCP endpoint.
- `ROBINHOOD_MCP_TOKEN` — the bearer token. Get it the same way as going
  live in the README: in a `claude` session in this directory run `/mcp`,
  pick `robinhood-trading`, complete the OAuth flow, copy the token.

**1b. Seed the `paper-state` branch from your local cache + record.**
CI must not cold-start — mean-reversion needs the lookback history and the
forward anchor must be preserved. From your main checkout, clean tree:

```bash
git fetch origin
git worktree add -f --detach /tmp/seed origin/main
cp -r data journal /tmp/seed/
( cd /tmp/seed
  git switch -c paper-state
  git add -Af data journal
  git -c user.name=paper-bot -c user.email=paper-bot@users.noreply.github.com \
      commit -m "seed paper-state: cache + forward record"
  git push -u origin paper-state )
git worktree remove -f /tmp/seed
```

**1c. Smoke-test now, don't wait for the schedule.**
Actions tab → **daily-paper-run** → **Run workflow**. A good run restores
state, fetches, prints `appended N day(s)`, and pushes `paper-state` if
anything changed. (`appended 0 days` is normal until the whole 65-name basket
settles the next trading day — not an error.)

**Verify:** the manual run is green; the record catches up past 2026-07-13
after the next trading day; check with
`git fetch && git show origin/paper-state:journal/forward/mean_reversion/run.json`.

**Ongoing watch:** if runs later fail with an auth error, the MCP token
rotated — update the secret. Consider making failures loud (the workflow can
notify on failure) so it never silently dies for two days again.

## Fix 2 — Clear the dead forward tracks (~5 min)

`journal/forward/agent/` and `journal/forward/pairs/` hold one day each
(2026-07-09) and pollute the dashboard's "Now" table with corpses.

- **If you don't care about those tracks:** delete the two directories (and
  the same paths on `paper-state` after Fix 1b, or delete before seeding).
- **If you want the agent track (you do — see Fix 5):** restart it on the
  fixed cadence by adding a second tick to `scripts/paper_cron.sh`
  (`--engine agent --eval-id agent`), accepting that the anchor restarts today.

**Verify:** dashboard "Now" table shows only records that are actually growing.

## Fix 3 — Put the grading back on top of the dashboard (~1–2 h)

All edits in `scripts/make_dashboard.py`:

1. **Reorder `render_all`** so page one reads, top to bottom: forward
   equity curve + forward table → latest-run scorecard + failure buckets →
   current `lessons_from_runs()` text (one paragraph, it's the system's
   memory — today it renders nowhere) → all-runs table → bake-off → run
   details. The overview cards can stay on top.
2. **Delete the code-graph panel** (`_code_health` and its call site). It's
   repo tooling, not trading state.
3. **Merge the duplicate render paths**: `_run_section` and `_run_detail` are
   ~90% identical — keep one (the `<details>` variant works for both if you
   pass `open_=True` for the single-run page), and let `render` reuse it.
4. **Commit or drop the pending working-tree diff** — it's fine (collapsible
   run details, safe anchors), but it's been sitting uncommitted; decide.

**Verify:** `python -m pytest tests/test_dashboard.py` passes;
`python scripts/make_dashboard.py --open` — the first screenful answers
"how is the live record doing and where do losses concentrate?"

## Fix 4 — One grading system, stated in writing (~30 min, docs only)

Decide: **trade-level grading is the project's judge** — scorecard + failure
buckets (`evaluate.py`) + robust Sharpe (`evaluate_robust.py`), confirmed by
the forward record. The IC/ICIR machinery (`factor/`, `search/`, `gate/`) is
an offline research tool for *generating candidates*, not the judge.

- Add a short "How a strategy is graded" section to README saying exactly
  that, with the promotion bar from FINDINGS.md ("the road to real money"),
  written down **now**, before the forward data exists.
- Mark §2 of ARCHITECTURE.md as the candidate-generation path, not "the math"
  of the whole system.
- No code changes needed; nothing has to be deleted — it just stops competing
  for the definition of "good".

**Verify:** a stranger reading the README can answer "what number decides
whether this goes live?" in one sentence.

## Fix 5 — Close the agent learning loop (~half a day)

Three small changes, all in existing files:

1. **Cadence:** schedule agent paper-trade runs — either a second tick in
   `scripts/paper_cron.sh` (`forward --engine agent`, after Fix 2) plus a
   periodic `rhagent.papertrade --engine agent` run, or a local cron. Mind the
   NVIDIA rate limit (~18 requests/min sustained) when sizing runs.
2. **Memory:** stamp the lessons into the run — in `papertrade.py`, where
   `AgentEngine(lessons=lessons)` is built, write the `lessons` string into
   `run.json`. One field. Now "step by step" is auditable: every run records
   what it was taught.
3. **Report card:** one new dashboard chart — agent net P&L / win rate per
   run in chronological order, with the mean-reversion baseline beside it.
   That chart *is* the original project: "is the agent getting better than
   the rule?" (This is the one allowed dashboard addition; it replaces the
   code-graph panel's slot.)

**Verify:** after 2–3 scheduled agent runs, `run.json` files show differing
lessons text, and the chart has points on it.

## Fix 6 — Kill the doc drift and stale branches (~20 min)

- README says the agent is "Claude (via the Anthropic API)"; the shipped agent
  is `nvidia/llama-3.3-nemotron-super-49b-v1.5` (config.yaml). Fix whichever
  is wrong — the doc or the config — so they agree.
- config.yaml comment says wiring the overlay into the forward path "is a
  follow-up"; it shipped (`ee2f7ef`). Delete the stale comment.
- Delete merged/abandoned local branches (`worktree-*`, `docs-durable-run`,
  `headless-refresh-cron`) and their remotes where merged:
  `git branch --merged main` is the safe list.

**Verify:** `git branch -a` is short; README, ARCHITECTURE.md, and config.yaml
tell the same story.

---

## After all six

Feature freeze holds until Fix 1 has produced **two consecutive weeks of green
scheduled runs**. Then the only work is watching the forward record climb the
trust ladder in FINDINGS.md — and the only code worth writing is whatever the
failure buckets say is losing money.
