#!/usr/bin/env bash
# Unattended daily forward paper-run for an always-on box (GitHub Actions or a
# cron host) -- laptop-independent. The cumulative cache (data/) and forward
# record (journal/) are gitignored, so they live on a dedicated `paper-state`
# branch: this script restores them, refreshes over the MCP, ticks, and pushes
# the updated state back.
#
# Requires only a git remote it can push to (in CI, the default GITHUB_TOKEN
# with contents:write). Data comes from Yahoo chart API (keyless) unless
# ROBINHOOD_MCP_URL/TOKEN are set, in which case the MCP is used. Run from repo root.
#
# paper-state must be seeded once from a checkout that already has the full cache
# + record -- see docs/paper-cron-setup.md. A cold start would give the strategy
# too little lookback history and orphan the existing forward record.
set -euo pipefail

STATE_BRANCH="${STATE_BRANCH:-paper-state}"
REFRESH_DAYS="${REFRESH_DAYS:-10}"
export PYTHONPATH="${PYTHONPATH:-src}"

echo "== restore cache + record from ${STATE_BRANCH} =="
if git fetch origin "${STATE_BRANCH}" 2>/dev/null; then
  # data/ and journal/ are tracked on paper-state (force-added), so restore pulls
  # them in despite the local .gitignore.
  git restore --source="origin/${STATE_BRANCH}" -- data journal
else
  echo "!! ${STATE_BRANCH} not found on origin -- seed it first (see docs/paper-cron-setup.md)" >&2
  exit 1
fi

echo "== refresh cache (last ${REFRESH_DAYS} days) =="
python -m rhagent.refresh --fetch --cache-dir data --days "${REFRESH_DAYS}"

echo "== tick forward record =="
python -m rhagent.forward

echo "== tick forward record (agent) =="
# The agent tick needs NVIDIA_API_KEY (one LLM call per symbol per new bar).
# Without the key it can't run; don't let it kill the strategy record above.
if [ -n "${NVIDIA_API_KEY:-}" ]; then
  python -m rhagent.forward --engine agent --eval-id agent \
    || echo "!! agent tick failed -- strategy record still persisted" >&2
else
  echo "NVIDIA_API_KEY not set -- skipping agent tick"
fi

echo "== persist cache + record to ${STATE_BRANCH} =="
tmp="$(mktemp -d)"
cleanup() { git worktree remove -f "${tmp}" 2>/dev/null || true; }
trap cleanup EXIT
git worktree add -f --detach "${tmp}" "origin/${STATE_BRANCH}"
rsync -a --delete data/ "${tmp}/data/"
rsync -a --delete journal/ "${tmp}/journal/"
git -C "${tmp}" add -Af data journal
if git -C "${tmp}" diff --cached --quiet; then
  echo "no state change to commit"
else
  git -C "${tmp}" \
    -c user.name="paper-bot" \
    -c user.email="paper-bot@users.noreply.github.com" \
    commit -q -m "paper-state $(date -u +%FT%TZ)"
  git -C "${tmp}" push origin "HEAD:${STATE_BRANCH}"
  echo "pushed updated state"
fi
