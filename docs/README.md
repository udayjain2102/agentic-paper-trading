# Docs index

What each doc under `docs/` is for. Start with the top-level
[`../README.md`](../README.md), then come here for depth.

| Doc | What it's for |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the two decision brains (LLM agent, quant pipeline) work, the math behind strategy grading, and the safety funnel. |
| [`going-live.md`](going-live.md) | The checklist for the day someone wires a live broker into `paper_run.py` and flips `LIVE=true` — not yet applicable, see the trust bar at the top. |
| [`paper-cron-setup.md`](paper-cron-setup.md) | One-time setup for the daily GitHub Actions paper-trade run. |
| [`safety-drills.md`](safety-drills.md) | Evidence the kill switches fire in the real `paper_run` path, not just unit tests. |
| [`audit-prompt.md`](audit-prompt.md) | A reusable skeptical-audit prompt for auditing this repo from a fresh session. |
| [`revamp-plan-2026-07-25.md`](revamp-plan-2026-07-25.md) | Working plan and the IC gate result (viable: 0, but real). |

## `archive/` is history, not current state

Superseded audits and shipped/deleted feature plans. Each file has a header
saying what superseded it or why it's stale. Nothing there describes the
system as it runs today.

- `AUDIT-2026-07-17.md`, `AUDIT-2026-07-22.md`, `FINDINGS.md`, `HOW-TO-FIX.md`,
  `HOW_IT_WORKS.md` — earlier audits and design docs whose findings were
  fixed or overtaken by later commits.
- `learning.md` — a real, evidence-backed comparison against an open-source
  alternative (Vibe-Trading), run 2026-07-21. Nothing in the codebase reads
  it; kept as a record of what was tried and found.
- `superpowers/plans/`, `superpowers/specs/` — implementation plans and
  design specs for features that have since shipped or were built, baked
  off, and deleted.
