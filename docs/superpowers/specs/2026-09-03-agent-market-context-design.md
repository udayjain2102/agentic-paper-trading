# Agent market-context experiment — design (2026-09-03)

## Why

The forward agent record (`journal/forward/agent/`) decides each symbol in
isolation from three numbers about that symbol alone (last close, 5-day
momentum, 20-day vol; `engine.py:_features`). On 2026-07-29 it held 22
shorts on negative-momentum names into a +1.7% SPY session and lost 1.73%
while the rule record gained 0.80%. That single day is nearly the whole gap
between the two records. The hypothesis: given the market's own move and
breadth, the model stops taking contrarian per-symbol bets into a broad
one-directional day.

This is a **fourth forward record run beside the existing three**, not a
change to any of them. The existing agent record is the control and keeps
running unmodified until its own judgment date.

## What changes

### `src/rhagent/engine.py`

- `AgentEngine.__init__` gains `market_context: bool = False`.
- `decide_all` computes, once per bar, from the `histories` it already
  receives:
  - `spy_1d`: SPY's last-bar return; `spy_5d`: SPY's return over up to 5 prior
    bars (same fallback rule as `_features`' momentum when history is short).
    If `SPY` is not in `histories`, both render as `n/a`.
  - `breadth_5d`: fraction of symbols in `histories` whose 5-day momentum is
    positive.
  - per-symbol `rank`: 1-based rank of the symbol's 5-day momentum among all
    symbols in `histories` (1 = strongest), and the count.
- `_prompt` inserts, when `market_context` is true, immediately after the
  symbol feature line and before the lessons block:

  ```
  Market today: SPY_1d=+1.68% SPY_5d=+2.10% breadth_5d=0.72 (share of 65 names with positive 5d momentum)
  This name: momentum_5d rank 12/65 (1 = strongest)
  ```

  When false, the prompt is byte-identical to today's.
- `decide()` keeps its signature; the block is passed to it as an optional
  keyword so per-symbol calls from `decide_all` carry it and direct
  `decide()` calls (tests, papertrade) get none.

### `src/rhagent/forward.py`

- New CLI flag `--market-context` (store_true). Passed through to both
  `AgentEngine(...)` construction sites (`_positions` and `tick_and_reflect`).
  Nothing else in the tick path changes; `agent_ctx` is just another
  `--eval-id` with its own directory, `pos_*.csv`, `decisions.jsonl`,
  `returns.csv`, `run.json`, anchored on first run like every record.
- `run.json` records `market_context: true|false` so the record is
  self-describing.

### `scripts/paper_cron.sh`

Inside the existing `NVIDIA_API_KEY` guard, after the control agent tick,
non-fatal like the others:

```bash
python -m rhagent.forward --engine agent --eval-id agent_ctx --market-context \
  || echo "!! agent_ctx tick failed -- other records still persisted" >&2
```

### `README.md`

Add `agent_ctx` to the records table and a bar under "The bar, written
down": judged on the first scheduled run on or after **2027-03-02**;
passes only if all four hold:

1. at least 90 scored days;
2. cumulative net on the days it and `mean_reversion_real` both scored is at
   least `mean_reversion_real`'s;
3. fewer than 10% of candidate days excluded for failed decisions;
4. cumulative net on the days it and the control `agent` both scored is
   above the control's. **This is the hypothesis.**

Outcomes: fails 4 → market context did not help; the next experiment is
news, and this record's tick is removed. Passes 4, fails 2 → context helped
but the agent is still not fundable; keep running as the new control. Passes
all → it replaces the control agent in the trust ladder.

## What does not change

- The three existing records, their ticks, flags, and bars.
- The model, `max_tokens`, `allow_short`, `use_lessons` (off), reflection
  (the new record does not reflect: `tick_and_reflect`'s reflection step is
  keyed on the control's memory file and is skipped when `--market-context`
  is set, so the control's memory is not written by two records).
- The dashboard. It hardcodes three legs; the new record is judged from
  `returns.csv` with the README's judge command, extended with `agent_ctx`.
- Cost: ~65 more calls per day under the existing pacer, a few minutes.

## Tests

- `market_context=False`: prompt for a fixed history is exactly the current
  string (regression pin).
- `market_context=True`: with a small synthetic `histories` dict including
  SPY, the block renders the expected SPY returns, breadth, and rank; the
  per-symbol call from `decide_all` receives it; a history without SPY
  renders `n/a` and does not raise.
- `forward` CLI: `--market-context` reaches `AgentEngine` (patch the class,
  assert the kwarg) and lands in `run.json`.
- Before shipping: one live local `decide_all` on four symbols with
  `market_context=True`, all `status=ok`.
