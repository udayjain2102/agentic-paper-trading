# agent_v2 — richer prompt + flagship model forward record

Date: 2026-11-16. Branch: `agent-v2`.

## Why this exists

The live LLM agent records (`agent`, `agent_ctx`) are losing money —
`agent` −3.3% / Sharpe −1.95 over 64 days, `agent_ctx` −2.1% / −1.96 over
39 days (as of 2026-11-11), while both mean-reversion records are positive.
Two independent problems were found and addressed:

1. **Starvation, now healed.** `decisions.jsonl` shows 62% of all rows are
   `status="failed"`, almost entirely HTTP 404/410 — NVIDIA retired
   `nemotron-3-super-120b` on 2026-10-19 and the account 404'd on every call
   from 2026-10-28 through 2026-11-04 (0 ok decisions/day). The switch to
   `nvidia/nemotron-3.5-super-vl-120b-a12b` on 2026-11-12 plus the retry rule
   healed the backlog; by 2026-11-12 all 65 symbols answered ok.

2. **Input poverty.** The per-symbol prompt gave the model exactly four
   numbers — `last_close`, `momentum_5d`, `vol_20d`, `current_pos` (plus an
   optional two-line market block). No trend structure, no mean-reversion
   distance, no volume, no position within the 52-week range. An LLM asked
   to trade on that is guessing; the −3% record is consistent with noise.

## What changed

### `src/rhagent/engine.py`

- `AgentEngine(rich_features=True)` renders a full technical snapshot per
  symbol instead of the 4-field line: `ret1d/5d/20d/60d`, `vol20d`,
  `volratio` (20d vs 60d vol), `distma20`, `distma50`, `rsi14`,
  `volu_ratio` (last volume vs 20d avg), `range52w` (position in the
  trailing-252-bar high/low), `current_pos`. Every lookback degrades
  gracefully on short history; no NaN/inf can reach the prompt.
- Helpers `_ret`, `_vol`, `_rsi14` added; `_features` dispatches to
  `_features_rich` when the flag is set. Default path is byte-identical —
  the control records are untouched.

### `src/rhagent/forward.py`

- `_build_agent(..., rich_features=False, model="")` — `model=""` still
  defers to `cfg.agent.model`.
- New CLI flags on `rhagent.forward`: `--rich-features` and `--model`.
  Both require `--engine agent`. `main()` now always builds the agent for
  `engine == "agent"` so the flags reach every call site (previously only
  `--market-context` did).

### `scripts/paper_cron.sh`

- Sixth forward record, `agent_v2`:
  ```
  python -m rhagent.forward --engine agent --eval-id agent_v2 \
    --market-context --rich-features \
    --model nvidia/nemotron-3-ultra-550b-a55b
  ```
  Guarded non-fatal like the other agent ticks. `--market-context` doubles
  as the opt-out of the shared memory/reflection path (`tick_and_reflect`
  treats `market_context` agents as context records and skips reflection),
  so agent_v2 cannot pollute `journal/agent_memory.md`.

### `tests/test_agent_engine.py`

- `test_rich_features_prompt_fields_and_short_history` — asserts all 12
  fields appear, no `nan`/`inf` leaks, and a 2-bar history still renders.

## Model bake-off (2026-11-16, live, one real AAPL decision prompt each)

| model | latency | result |
|---|---|---|
| nvidia/nemotron-3.5-super-vl-120b-a12b (current) | 6.9s | clean JSON |
| **nvidia/nemotron-3-ultra-550b-a55b** | **4.2s** | clean JSON — picked for v2 |
| nvidia/nemotron-3.5-lightning-30b-a3b | 11.5s | clean JSON |
| mistralai/mistral-large-4-1t-a52b | 27.6s | clean JSON |
| stepfun-ai/step-5-preview | 80.7s | clean JSON, too slow |
| deepseek-ai/deepseek-v4.1-flash | timeout | unusable |
| z-ai/glm-5.3-flash | timeout | unusable |
| google/gemma-4-31b-it | timeout | unusable |
| openai/gpt-oss-20b | timeout | unusable |

Single-sample evidence — enough to pick a plausible flagship, not a
ranking. Verified separately that ultra answers the *rich + context*
prompt in 16.7s with clean JSON and a coherent reason.

## What agent_v2 does NOT change

- Decision space stays `{-1, 0, +1}` per symbol, one call per symbol per
  bar, `decide_all`'s 8-worker pool under the 18/min pace limit.
- Same exclusion/retry/successor rules in `_agent_positions` — a failed
  symbol still excludes its whole day from the return series.
- Control records (`agent`, `agent_ctx`, `agent_open`) keep their exact
  prompts and models; the comparison is clean.
- Confound to remember when reading results: agent_v2 changes features,
  context, AND model at once vs the controls. If it wins we won't know
  which lever did it without further records.

## Cost

65 calls/day × ~5-17s each, paced at 18/min → roughly 4-6 minutes added to
the daily workflow per agent record; agent_v2 adds one more such block.

## Open levers not yet pulled

- Portfolio-level single call (one prompt, all 65 names, JSON map of
  targets) — 65x cheaper, allows diversification reasoning, but re-opens
  the batching question the code comments document as previously rejected.
- Conviction/confidence field in the JSON schema (logged, not traded).
- Per-record reflection/lessons once the live record is long enough to
  derive them honestly.
