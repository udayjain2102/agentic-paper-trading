# Branch cleanup proposal (2026-07-26)

**PROPOSAL ONLY — nothing below has been run.** Deleting branches is
destructive; review and run these yourself.

Generated from `git branch --merged origin/main`, `git branch -r --merged
origin/main`, and `git worktree list` on this checkout, with `origin`
fetched/pruned first.

## Safe to delete now — local branches merged into `origin/main`, no worktree

These branches' tip commits are already ancestors of `origin/main` and are
not checked out in any `.claude/worktrees/*` worktree.

```bash
git branch -d worktree-agent-a2c0b8f9b57ffdf9f
git branch -d worktree-agent-a7d42c81629bb76c9
git branch -d worktree-agent-adf053b7cb781a340
git branch -d worktree-audit-0722-iter2
```

## Safe to delete now — remote branches merged into `origin/main`

```bash
git push origin --delete dashboard-live-status
git push origin --delete fix-agent-observability
git push origin --delete pages-live-dashboard
git push origin --delete worktree-agent-memory-loop
git push origin --delete worktree-richer-trade-learning
```

## Merged, but blocked by an active worktree — remove the worktree first

These local branches are merged into `origin/main` but are checked out in a
worktree under `.claude/worktrees/`, so `git branch -d` will refuse (or would
detach the worktree) until the worktree is removed:

| Branch | Worktree |
|---|---|
| `agent-headtohead` | `.claude/worktrees/audit-0722-iter2` |
| `check-refresh` | `.claude/worktrees/agent-memory-loop` |
| `fix-short-cache-symbols` | `.claude/worktrees/richer-trade-learning` |
| `worktree-agent-a30571fdf590dc1bc` | `.claude/worktrees/agent-a30571fdf590dc1bc` |
| `worktree-floating-juggling-noodle` | `.claude/worktrees/floating-juggling-noodle` |
| `worktree-mean-reversion-stop-loss` | `.claude/worktrees/mean-reversion-stop-loss` |

Once you've confirmed a worktree's work is no longer needed:

```bash
git worktree remove .claude/worktrees/<dir>   # or --force if it has untracked state
git branch -d <branch>
```

The remote counterpart for `fix-short-cache-symbols` and
`worktree-mean-reversion-stop-loss` is also merged and can be deleted
independently of the local worktree (deleting a remote ref never touches a
local worktree):

```bash
git push origin --delete fix-short-cache-symbols
git push origin --delete worktree-mean-reversion-stop-loss
```

## Not merged — needs a human decision, not a mechanical delete

These are ahead of `origin/main` (real, possibly-still-wanted work) or are
special-purpose and should not be deleted by a cleanup pass:

- `paper-state` (local + remote) — **never delete**; it's the only copy of
  the forward paper-trade record and cache, not a feature branch.
- `local-main-backup` (local + remote) — the Step-1 safety backup from
  `docs/revamp-plan-2026-07-25.md`; keep until that plan's Step 1 is
  confirmed done and no longer needed.
- `docs/consolidate-md` (local, pinned by `.claude/worktrees/agent-adf053b7cb781a340`)
  — appears to be another attempt at this same doc-consolidation task; review
  and reconcile with this branch's work before deleting either.
- `feat/vibe-trading-learnings`, `add-gross-loss-tile`,
  `feat-dashboard-run-trigger`, `fix-runs-table-sort-order`,
  `surface-truncation`, `worktree-remove-run-passphrase`,
  `worktree-trading-control-room` — local branches not ancestors of
  `origin/main`; each may hold unmerged work or be a stale rebase artifact.
  Inspect with `git log origin/main..<branch>` before deciding.
- `fix/agent-batch-decide`, `worktree-audit-fixes-0722`,
  `worktree-agent-a4ea2c2ad9d79dce1`, `worktree-agent-a9a9fcbc98c937820`,
  `worktree-agent-aaa22f3c91f841306`, `worktree-fix-gross-pnl-notional`,
  `worktree-ledger-collapse`, `worktree-symbols-all-config-universe`,
  `add-drawer-gross-loss-tile`, `pr36-merge`, `slim-dashboard` — all pinned by
  active worktrees and not merged into `origin/main`; likely in-progress or
  superseded work. Check `git log origin/main..<branch>` per branch before
  touching.

## Verification after running any of the above

```bash
git branch --merged origin/main    # should shrink to just `main`
git branch -r --merged origin/main # should shrink to origin/main, origin/HEAD
git worktree list                  # confirm removed worktrees are gone
```
