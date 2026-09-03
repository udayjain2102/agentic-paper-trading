# Docs index

What each doc under `docs/` is for. Start with the top-level
[`../README.md`](../README.md), then come here for depth.

| Doc | What it's for |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the two decision brains (LLM agent, rule strategy) work, the math behind strategy grading, and the safety funnel. |
| [`going-live.md`](going-live.md) | The checklist for the day someone wires a live broker into `paper_run.py` and flips `LIVE=true` — not yet applicable, see the trust bar at the top. |
| [`paper-cron-setup.md`](paper-cron-setup.md) | One-time setup for the daily GitHub Actions paper-trade run. |
| [`safety-drills.md`](safety-drills.md) | Evidence the kill switches fire in the real `paper_run` path, not just unit tests. |
| [`audit-prompt.md`](audit-prompt.md) | A reusable skeptical-audit prompt for auditing this repo from a fresh session. |

Earlier audits, plans, and specs (`docs/archive/`, `revamp-plan-2026-07-25.md`)
were deleted on 2026-09-03; they described a system that no longer exists and
are in git history before that date. The keep-or-kill bar that supersedes
them is in the top-level README.
