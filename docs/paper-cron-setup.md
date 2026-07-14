# Durable daily paper-run (GitHub Actions)

Runs `refresh --fetch` + `forward` tick every weekday on GitHub's runners, so the
forward record grows without your laptop being on. The cumulative cache (`data/`)
and record (`journal/`) are gitignored, so they live on a dedicated `paper-state`
branch that the workflow reads and writes each run.

Workflow: `.github/workflows/daily-paper-run.yml` → `scripts/paper_cron.sh`.

## One-time setup

### 1. Add the two repo secrets

GitHub → repo **Settings → Secrets and variables → Actions → New repository secret**:

- `ROBINHOOD_MCP_URL` — the hosted MCP endpoint (public https).
- `ROBINHOOD_MCP_TOKEN` — the long-lived bearer token.

### 2. Seed `paper-state` with your current cache + record

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

### 3. Enable and smoke-test

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
- **Token expiry.** If runs start failing with an auth error, the MCP token
  rotated — update the `ROBINHOOD_MCP_TOKEN` secret.
- **Reading the record.** The latest cache + record always sit on the
  `paper-state` branch; `git fetch && git show origin/paper-state:journal/...` or
  regenerate the dashboard from a checkout of it.
- **Paper only.** No broker token is set (`LIVE` unset), so nothing trades.
