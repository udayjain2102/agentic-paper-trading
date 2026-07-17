# Durable daily paper-run (GitHub Actions)

Runs `refresh --fetch` + `forward` tick every weekday on GitHub's runners, so the
forward record grows without your laptop being on. The cumulative cache (`data/`)
and record (`journal/`) are gitignored, so they live on a dedicated `paper-state`
branch that the workflow reads and writes each run.

Workflow: `.github/workflows/daily-paper-run.yml` → `scripts/paper_cron.sh`.

Data comes from Yahoo's chart API — keyless, so **no secrets are required**. The
Robinhood MCP is used instead only if `ROBINHOOD_MCP_URL`/`ROBINHOOD_MCP_TOKEN`
are ever set (its OAuth only completes inside an interactive Claude session, so
in practice CI always uses Yahoo).

## One-time setup

### 1. Seed `paper-state` with your current cache + record

CI must start from the cache you already have — mean-reversion needs the full
lookback history, and this preserves the existing forward anchor. Do **not** let
it cold-start. From your live checkout (which has `data/` and `journal/`), on a
clean working tree:

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

### 2. Enable and smoke-test

- **Actions** tab → enable workflows if prompted.
- Open **daily-paper-run** → **Run workflow** (the `workflow_dispatch` button) to
  trigger a run now instead of waiting for the schedule. Watch the log: it should
  restore state, fetch, report `appended N day(s)`, and push `paper-state` only if
  something changed.

## Notes

- **Schedule is UTC.** `17 11 * * 1-5` = 11:17 UTC weekdays. Edit the cron in the
  workflow to move it; GitHub has no per-timezone cron and may start a few minutes
  late under load.
- **`appended 0 days` is normal** until the whole 65-name basket has settled the
  next trading day (the realized-day guard). Not an error.
- **Dead names.** A symbol Yahoo stops serving is skipped with a warning; the
  full-coverage guard in `forward.py` then freezes the record until the name is
  dropped from the universe.
- **Reading the record.** The latest cache + record always sit on the
  `paper-state` branch; `git fetch && git show origin/paper-state:journal/...` or
  regenerate the dashboard from a checkout of it.
- **Paper only.** No broker token is set (`LIVE` unset), so nothing trades.
