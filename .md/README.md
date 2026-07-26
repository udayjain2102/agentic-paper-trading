# Docs index

What each doc under `.md/` (and the top-level docs it's paired with) is for.
If you're new here, read top to bottom in this order.

| Doc | What it's for |
|---|---|
| [`../README.md`](../README.md) | Start here. What the system actually runs today, and how it's graded. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the two decision brains (LLM agent, quant pipeline) work, the math behind strategy grading, and the safety funnel. |
| [`AUDIT-2026-07-22.md`](AUDIT-2026-07-22.md) | **The current audit.** Most accurate description of what's live vs. dead, verified against `origin/main` by execution. Read this to know what's real. |
| [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) | One-page diagram of the agent's decision flow (trigger → brains → safety funnel → execution → record). |
| [`going-live.md`](going-live.md) | The checklist for flipping `LIVE=true` — not yet cleared, see the trust bar at the top. |
| [`paper-cron-setup.md`](paper-cron-setup.md) | One-time setup for the daily GitHub Actions paper-trade run. |
| [`../learning.md`](../learning.md) | Notes from benchmarking this repo against an open-source alternative (Vibe-Trading). |

## `archive/` is history, not current state

`.md/archive/` holds superseded audits and shipped/deleted feature plans —
each file has a one-line header saying what superseded it. Nothing there
describes the system as it runs today; check the table above for that.

- `AUDIT-2026-07-17.md`, `FINDINGS.md`, `HOW-TO-FIX.md` — earlier audits whose
  findings were fixed (daily CI pipeline, agent parse-fail rate) or whose
  action items were completed the day they were written.
- `superpowers/plans/`, `superpowers/specs/` — implementation plans and design
  specs for features that have since shipped (and are now documented in
  `ARCHITECTURE.md`) or were built, baked off, and deleted.

## Repo-root planning docs

`docs/audit-prompt.md` and `docs/revamp-plan-2026-07-25.md` are working
planning documents, not part of this index — they track in-flight work, not
the shipped system.
