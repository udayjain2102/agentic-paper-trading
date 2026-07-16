# Forward paper-run: cadence and its OAuth constraint

The forward record (`journal/forward/*`) only grows when something ticks it
daily: `refresh --fetch` (or the stdin path) to update `data/*.csv`, then
`rhagent.forward` per engine. This doc covers how that tick actually runs
today, and why the originally-planned GitHub Actions workflow doesn't work.

## Why headless CI can't do this

Robinhood's agent MCP (`https://agent.robinhood.com/mcp/trading`) is
**OAuth-only** — there is no dashboard or API-key page that issues a separate,
long-lived bearer token for server-to-server use. The OAuth handshake only
completes inside an interactive Claude Code session (`/mcp` → `robinhood-trading`),
and the resulting credential isn't an extractable static secret you can paste
into a GitHub Actions repo secret.

That means `rhagent.refresh --fetch` and `McpBroker` — both of which need
`ROBINHOOD_MCP_URL`/`ROBINHOOD_MCP_TOKEN` as plain env vars for a direct,
unattended HTTP call — cannot authenticate in **any** headless context, not
just GitHub's runners. A plain cron job on an always-on box would hit the same
wall. `.github/workflows/daily-paper-run.yml` is kept in the repo but its
schedule is disabled (see below) because it can never succeed as designed.

If Robinhood's agent product ever adds a non-interactive API-key mechanism,
headless CI becomes possible again — see "If a static token becomes
available" at the bottom.

## The actual working path: an interactive Claude session

The Robinhood MCP tools (`mcp__robinhood-trading__*`) are only live inside a
Claude Code session that has completed `/mcp` OAuth. So the daily tick has to
run from inside one. Two ways to do that:

**Manual, on any trading day:**

```bash
# 1. In a live Claude Code session (MCP already connected via /mcp):
#    ask Claude to fetch fresh bars (get_equity_historicals) and pipe the
#    payload into refresh — this is the "stdin" path in refresh.py, and it
#    doesn't need ROBINHOOD_MCP_URL/TOKEN at all.
... | PYTHONPATH=src .venv/bin/python -m rhagent.refresh --cache-dir data

# 2. Tick both forward records:
PYTHONPATH=src .venv/bin/python -m rhagent.forward
PYTHONPATH=src .venv/bin/python -m rhagent.forward --engine agent --eval-id agent

# 3. Rebuild the dashboard:
PYTHONPATH=src .venv/bin/python scripts/make_dashboard.py
```

**Scheduled, without babysitting it:** run this as a recurring `/loop` inside
a Claude Code session left open on a machine that stays on (e.g. `/loop 1d`
with a prompt that does steps 1–3 above). This keeps the record growing
without a fully headless CI job — the session just needs to be alive and
MCP-authenticated once a day.

## Reading the record

`journal/forward/<engine>/` holds the growing record directly in your working
tree (gitignored, local) — no `paper-state` branch round-trip is needed since
nothing is running off-box. `git show`/dashboard regen work the same as
before, just against your local `journal/`.

## If a static token becomes available later

Should Robinhood's agent MCP ever add a separate long-lived token/API-key
mechanism:

1. Add `ROBINHOOD_MCP_URL` / `ROBINHOOD_MCP_TOKEN` as repo secrets (GitHub →
   Settings → Secrets and variables → Actions).
2. Seed a `paper-state` branch with your current `data/` + `journal/` (see
   git history for the exact seeding commands — they're straightforward:
   worktree, copy, commit, push).
3. Re-enable the `schedule:` trigger in
   `.github/workflows/daily-paper-run.yml` (currently commented out) and
   smoke-test via **Actions → daily-paper-run → Run workflow**.
