# Agent Market-Context Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a fourth forward paper-trade record, `agent_ctx`, identical to the control `agent` record except that every per-symbol prompt carries a two-line cross-sectional market block (SPY 1d/5d return, breadth, the symbol's momentum rank).

**Architecture:** `AgentEngine` gets a `market_context` flag; `decide_all` computes the block once per bar from a full-universe `context_histories` dict and passes a rendered string into each per-symbol `decide()` → `_prompt()`. `forward.py` gets a `--market-context` CLI flag, builds the engine through one shared helper, hands `_agent_positions` the full-universe histories, records the flag in `run.json`, and skips reflection for the new record. `paper_cron.sh` adds one non-fatal tick; the README adds the record and its bar.

**Tech Stack:** Python 3.10+, pandas, pytest (run as `PYTHONPATH=src .venv/bin/python -m pytest -q`). No new dependencies.

## Global Constraints

- With `market_context=False` (the default) the prompt string must be **byte-identical** to today's. The control record must not change.
- The block is exactly two lines, in this shape (values vary):
  `Market today: SPY_1d=+1.68% SPY_5d=+2.10% breadth_5d=0.72 (share of 65 names with positive 5d momentum)` /
  `This name: momentum_5d rank 12/65 (1 = strongest)`
- Missing SPY renders `n/a`, never raises.
- Model, `max_tokens`, `allow_short`, `use_lessons` stay whatever `config.yaml` says; nothing in this plan edits `config.yaml`.
- New record id is `agent_ctx`; it runs inside the existing `NVIDIA_API_KEY` guard in `scripts/paper_cron.sh`, non-fatal.
- Spec: `docs/superpowers/specs/2026-09-03-agent-market-context-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/rhagent/engine.py` (modify) | `_mom5`, `market_block`, `render_context` (pure); `AgentEngine(market_context=)`; `decide(context=)`; `decide_all(context_histories=)` |
| `src/rhagent/forward.py` (modify) | `_build_agent(cfg, *, market_context, memory_text)` helper; `_agent_positions` passes `context_histories`; `--market-context` flag; `run.json["market_context"]`; reflection skipped when set |
| `scripts/paper_cron.sh` (modify) | fourth tick |
| `README.md` (modify) | record table row + bar |
| `tests/test_agent_context.py` (create) | engine-level tests |
| `tests/test_forward_market_context.py` (create) | forward-level tests |

---

### Task 1: The market block and prompt seam in `engine.py`

**Files:**
- Modify: `src/rhagent/engine.py` (`_features` ~line 186, `_prompt` ~205, `decide` ~233, `decide_all` ~251)
- Test: `tests/test_agent_context.py`

**Interfaces:**
- Produces: `market_block(histories: dict[str, pd.DataFrame]) -> dict` with keys `spy_1d: float|None`, `spy_5d: float|None`, `breadth: float|None`, `rank: dict[str, int]`, `n: int`.
- Produces: `render_context(block: dict, symbol: str) -> str` (two lines, each ending `\n`).
- Produces: `AgentEngine.__init__(..., market_context: bool = False)`; `AgentEngine.decide(symbol, history, current_pos, context: str = "")`; `AgentEngine.decide_all(symbols, histories, current_pos, context_histories: dict | None = None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_context.py`:

```python
"""Market-context block for the agent_ctx record: pure block math, prompt
insertion, and the regression pin that market_context=False changes nothing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rhagent.engine import AgentEngine, market_block, render_context


def _hist(closes):
    closes = list(closes)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


def _universe():
    # SPY up 1% on the last bar and ~+2% over 5; AAA strongest, CCC weakest.
    return {
        "SPY": _hist([100, 100.5, 101, 101.5, 101.0, 102.0]),
        "AAA": _hist([10, 10, 10, 10, 10, 12]),      # +20% 5d
        "BBB": _hist([10, 10, 10, 10, 10, 10.5]),    # +5% 5d
        "CCC": _hist([10, 10, 10, 10, 10, 9]),       # -10% 5d
    }


def test_market_block_values():
    b = market_block(_universe())
    assert abs(b["spy_1d"] - (102.0 / 101.0 - 1)) < 1e-12
    assert abs(b["spy_5d"] - (102.0 / 100.0 - 1)) < 1e-12
    assert b["n"] == 4
    # SPY, AAA, BBB positive; CCC negative -> 3/4
    assert abs(b["breadth"] - 0.75) < 1e-12
    assert b["rank"]["AAA"] == 1 and b["rank"]["CCC"] == 4


def test_market_block_without_spy_is_na_not_error():
    u = _universe()
    del u["SPY"]
    b = market_block(u)
    assert b["spy_1d"] is None and b["spy_5d"] is None
    assert b["n"] == 3
    txt = render_context(b, "AAA")
    assert "SPY_1d=n/a SPY_5d=n/a" in txt
    assert "rank 1/3" in txt


def test_render_context_shape():
    txt = render_context(market_block(_universe()), "BBB")
    lines = txt.split("\n")
    assert lines[0].startswith("Market today: SPY_1d=+0.99% SPY_5d=+2.00% breadth_5d=0.75 ")
    assert lines[0].endswith("(share of 4 names with positive 5d momentum)")
    assert lines[1] == "This name: momentum_5d rank 3/4 (1 = strongest)"
    assert txt.endswith("\n")


def test_market_context_false_prompt_is_unchanged():
    """Regression pin: the control record's prompt must not move."""
    h = _hist([10, 11, 12, 13, 14, 15, 16])
    eng = AgentEngine(complete=lambda p: "{}")
    before = eng._prompt("NVDA", h, 0.0)
    after = AgentEngine(complete=lambda p: "{}", market_context=False)._prompt("NVDA", h, 0.0)
    assert before == after
    assert "Market today" not in before
    # and the exact text a production prompt has today
    assert before.startswith("You are a trading agent deciding today's position in NVDA.\n"
                             "NVDA: last_close=16.00 momentum_5d=+0.4545 vol_20d=")
    assert "Respond with ONLY this JSON object" in before


def test_decide_all_passes_block_into_every_prompt():
    u = _universe()
    seen = {}

    def fake(prompt):
        sym = prompt.split("position in ")[1].split(".")[0]
        seen[sym] = prompt
        return json.dumps({"target": 1, "reason": "up"})

    eng = AgentEngine(complete=fake, market_context=True)
    # decide only two names, but give the whole universe as context
    out = eng.decide_all(["AAA", "CCC"], {s: u[s] for s in ("AAA", "CCC")},
                         {"AAA": 0.0, "CCC": 0.0}, context_histories=u)
    assert set(seen) == {"AAA", "CCC"}
    for sym, p in seen.items():
        assert "Market today: SPY_1d=+0.99% SPY_5d=+2.00% breadth_5d=0.75" in p
        assert "Respond with ONLY this JSON object" in p
        # block sits after the feature line and before the JSON instruction
        assert p.index(f"{sym}: last_close=") < p.index("Market today") < p.index("Respond with ONLY")
    assert "rank 1/4" in seen["AAA"] and "rank 4/4" in seen["CCC"]
    assert out["AAA"].status == "ok" and out["CCC"].status == "ok"


def test_decide_all_without_context_histories_uses_histories():
    u = _universe()
    seen = []
    eng = AgentEngine(complete=lambda p: (seen.append(p), '{"target": 0, "reason": "x"}')[1],
                      market_context=True)
    eng.decide_all(list(u), u, {s: 0.0 for s in u})
    assert all("breadth_5d=0.75" in p for p in seen)


def test_market_context_false_decide_all_sends_no_block():
    u = _universe()
    seen = []
    eng = AgentEngine(complete=lambda p: (seen.append(p), '{"target": 0, "reason": "x"}')[1])
    eng.decide_all(list(u), u, {s: 0.0 for s in u}, context_histories=u)
    assert seen and not any("Market today" in p for p in seen)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_context.py -q`
Expected: FAIL at import with `ImportError: cannot import name 'market_block'`.

- [ ] **Step 3: Implement in `src/rhagent/engine.py`**

Add these module-level functions directly above `class AgentEngine` (after `_fail_reason` is also fine; they must be importable from `rhagent.engine`):

```python
def _mom5(close: pd.Series) -> float:
    """5-bar momentum with the same short-history fallback _features uses."""
    k = min(5, len(close) - 1)
    return float(close.iloc[-1] / close.iloc[-1 - k] - 1.0) if k >= 1 else 0.0


def market_block(histories: dict[str, pd.DataFrame]) -> dict:
    """Cross-sectional context for ONE bar, from every symbol's history up to
    that bar. Pure; no I/O. Missing SPY -> None fields, never an exception."""
    moms = {s: _mom5(h["close"].astype(float)) for s, h in histories.items() if len(h)}
    spy = histories.get("SPY")
    if spy is not None and len(spy) >= 2:
        c = spy["close"].astype(float)
        spy_1d = float(c.iloc[-1] / c.iloc[-2] - 1.0)
        spy_5d = _mom5(c)
    else:
        spy_1d = spy_5d = None
    n = len(moms)
    breadth = (sum(m > 0 for m in moms.values()) / n) if n else None
    order = sorted(moms, key=lambda s: moms[s], reverse=True)
    return {"spy_1d": spy_1d, "spy_5d": spy_5d, "breadth": breadth,
            "rank": {s: i + 1 for i, s in enumerate(order)}, "n": n}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def render_context(block: dict, symbol: str) -> str:
    """The two prompt lines for `symbol`. Ends with a newline."""
    n = block["n"]
    breadth = "n/a" if block["breadth"] is None else f"{block['breadth']:.2f}"
    rank = block["rank"].get(symbol)
    rank_txt = f"{rank}/{n}" if rank is not None else f"n/a/{n}"
    return (f"Market today: SPY_1d={_pct(block['spy_1d'])} SPY_5d={_pct(block['spy_5d'])} "
            f"breadth_5d={breadth} (share of {n} names with positive 5d momentum)\n"
            f"This name: momentum_5d rank {rank_txt} (1 = strongest)\n")
```

In `AgentEngine.__init__`, add the kwarg and attribute:

```python
        allow_short: bool = False,
        max_tokens: int | None = None,
        market_context: bool = False,
    ) -> None:
        ...
        self.max_tokens = max_tokens
        self.market_context = market_context
```

In `_features`, replace the two momentum lines with the helper so the two paths cannot drift:

```python
        mom5 = _mom5(close)
```

(delete the `k = min(5, ...)` line and the old `mom5 = ...` line; output is identical.)

Change `_prompt` to take the context and insert it between the feature line and the lessons block:

```python
    def _prompt(self, symbol: str, history: pd.DataFrame, current_pos: float,
                context: str = "") -> str:
        return (
            f"You are a trading agent deciding today's position in {symbol}.\n"
            f"{self._features(symbol, history, current_pos)}\n"
            f"{context}"
            f"{self._lessons_block()}"
            "Respond with ONLY this JSON object and nothing else -- no "
            "reasoning, no markdown fences, no text before or after it: "
            '{"target": -1 | 0 | 1, "reason": "<=15 words"} where target is '
            "the desired position (-1 short, 0 flat, 1 long)."
        )
```

Change `decide` to accept and forward it:

```python
    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float,
        context: str = "",
    ) -> Decision:
        if self.complete is None:
            self.complete = self._default_complete()
        status = "ok"
        try:
            obj = self._extract(self.complete(
                self._prompt(symbol, history, current_pos, context)))
```

Change `decide_all` (keep its docstring; append one paragraph):

```python
    def decide_all(self, symbols: list[str], histories: dict[str, pd.DataFrame],
                   current_pos: dict[str, float],
                   context_histories: dict[str, pd.DataFrame] | None = None
                   ) -> dict[str, Decision]:
        """...existing docstring...

        `context_histories` is the FULL universe's history up to this bar (the
        caller may be deciding only a subset). Used only when market_context is
        set: the block is computed once here and handed to every per-symbol
        prompt. Falls back to `histories` when not given.
        """
        if self.complete is None:
            self.complete = self._default_complete()
        ctx: dict[str, str] = {}
        if self.market_context:
            block = market_block(context_histories if context_histories is not None
                                 else histories)
            ctx = {s: render_context(block, s) for s in symbols}
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            futures = {s: pool.submit(self.decide, s, histories[s], current_pos[s],
                                      ctx.get(s, ""))
                       for s in symbols}
        return {s: f.result() for s, f in futures.items()}
```

- [ ] **Step 4: Run the new tests and the whole suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_context.py -q`
Expected: 7 passed.

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: all pass (200 existing + 7). Any failure in `tests/test_agent_engine.py` or `tests/test_forward_*.py` means the default-path prompt or `decide` signature changed; fix before continuing.

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/engine.py tests/test_agent_context.py
git commit -m "feat(engine): optional cross-sectional market block in the agent prompt

market_context=False (default) is byte-identical to today. decide_all takes
context_histories (the full universe up to the bar) and renders one block
per bar into every per-symbol prompt when the flag is on."
```

---

### Task 2: Wire the flag through `forward.py`

**Files:**
- Modify: `src/rhagent/forward.py` (`_agent_positions` ~line 70 and the `decide_all` call ~153; `_positions` ~215; `tick_and_reflect` ~380; `main` ~549)
- Test: `tests/test_forward_market_context.py`

**Interfaces:**
- Consumes (Task 1): `AgentEngine(market_context=bool)`, `decide_all(..., context_histories=)`.
- Produces: `forward._build_agent(cfg, *, market_context: bool = False, memory_text: str | None = None) -> AgentEngine`.
- Produces: CLI flag `--market-context`; `run.json["market_context"]: bool` on agent records.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forward_market_context.py`:

```python
"""--market-context must reach the AgentEngine a production tick constructs,
the per-bar decide_all call must carry the full universe as context, run.json
must say which kind of record this is, and the ctx record must not reflect."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rhagent import forward


def _bars(symbols, n=30):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    out = {}
    for i, s in enumerate(symbols):
        close = pd.Series(np.linspace(100, 100 + i, n), index=idx)
        out[s] = pd.DataFrame({"open": close, "high": close, "low": close,
                               "close": close, "volume": 1e6}, index=idx)
    return out


def _cfg(universe):
    return SimpleNamespace(
        strategy=SimpleNamespace(name="agent", params={}, universe=list(universe),
                                 overlay="none", cost_bps=7.0, fill_mode="next_open"),
        agent=SimpleNamespace(model="", max_tokens=None, allow_short=True,
                              use_lessons=False),
    )


def test_build_agent_passes_flag():
    cfg = _cfg(["AAA"])
    assert forward._build_agent(cfg).market_context is False
    assert forward._build_agent(cfg, market_context=True).market_context is True
    assert forward._build_agent(cfg, market_context=True).allow_short is True


def test_agent_positions_gives_decide_all_the_full_universe(monkeypatch, tmp_path):
    bars = _bars(["AAA", "BBB", "SPY"])
    calls = []

    class Fake:
        market_context = True

        def decide_all(self, symbols, histories, current_pos, context_histories=None):
            calls.append((tuple(symbols), None if context_histories is None
                          else tuple(sorted(context_histories))))
            from rhagent.engine import Decision
            return {s: Decision(target=0.0, reason="x", status="ok") for s in symbols}

    forward._agent_positions(tmp_path, bars, Fake())
    assert calls, "decide_all never called"
    for symbols, ctx in calls:
        assert ctx == ("AAA", "BBB", "SPY")   # every bar sees the whole universe


def test_positions_uses_market_context_flag(monkeypatch, tmp_path):
    bars = _bars(["AAA", "SPY"])
    seen = []

    def fake_complete(prompt):
        seen.append(prompt)
        return '{"target": 1, "reason": "long"}'

    monkeypatch.setattr("rhagent.engine.AgentEngine._default_complete",
                        lambda self: fake_complete)
    cfg = _cfg(["AAA", "SPY"])
    forward._positions(cfg, "agent", bars, tmp_path,
                       agent=forward._build_agent(cfg, market_context=True))
    assert seen and all("Market today: SPY_1d=" in p for p in seen)


def test_tick_and_reflect_records_flag_and_skips_reflection(monkeypatch, tmp_path):
    bars = _bars(["AAA", "SPY"], n=40)
    cfg = _cfg(["AAA", "SPY"])

    def fake_complete(prompt):
        return '{"target": 1, "reason": "long"}'

    monkeypatch.setattr("rhagent.engine.AgentEngine._default_complete",
                        lambda self: fake_complete)
    reflected = []
    monkeypatch.setattr("rhagent.memory.reflect",
                        lambda *a, **k: (reflected.append(1), "x")[1])

    def fake_fetch(*a, **k):
        raise AssertionError("must not fetch")

    cache = tmp_path / "data"
    cache.mkdir()
    for s, f in bars.items():
        f.to_csv(cache / f"{s}.csv", index_label="date")

    ed = tmp_path / "agent_ctx"
    agent = forward._build_agent(cfg, market_context=True)
    res = forward.tick_and_reflect(cfg, ed, 7.0, engine="agent", fill="next_open",
                                   today=date(2025, 2, 12), cache_dir=cache, agent=agent,
                                   memory_path=str(tmp_path / "mem.md"))
    assert res["appended"] == 1
    meta = json.loads((ed / "run.json").read_text())
    assert meta["market_context"] is True
    assert meta["reflected"] is False
    assert reflected == []          # ctx record never touches the shared memory

    ed2 = tmp_path / "agent"
    res2 = forward.tick_and_reflect(cfg, ed2, 7.0, engine="agent", fill="next_open",
                                    today=date(2025, 2, 12), cache_dir=cache,
                                    agent=forward._build_agent(cfg),
                                    memory_path=str(tmp_path / "mem.md"))
    assert res2["appended"] == 1
    assert json.loads((ed2 / "run.json").read_text())["market_context"] is False


def test_cli_flag_reaches_engine(monkeypatch, tmp_path):
    captured = {}

    def fake_tick_and_reflect(cfg, eval_dir, cost_bps, **kw):
        captured["agent"] = kw.get("agent")
        captured["eval_dir"] = eval_dir
        return {"appended": 0, "total_days": 0, "meta": {}}

    monkeypatch.setattr(forward, "tick_and_reflect", fake_tick_and_reflect)
    monkeypatch.setattr(forward, "_report", lambda *a, **k: None)
    forward.main(["--engine", "agent", "--eval-id", "agent_ctx", "--market-context",
                  "--out-dir", str(tmp_path)])
    assert captured["agent"] is not None
    assert captured["agent"].market_context is True
    assert captured["eval_dir"] == Path(tmp_path) / "agent_ctx"

    forward.main(["--engine", "agent", "--out-dir", str(tmp_path)])
    assert captured["agent"] is None   # control path unchanged: engine built inside
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_forward_market_context.py -q`
Expected: FAIL, first with `AttributeError: module 'rhagent.forward' has no attribute '_build_agent'`.

- [ ] **Step 3: Implement in `src/rhagent/forward.py`**

Add the helper above `_positions`:

```python
def _build_agent(cfg, *, market_context: bool = False, memory_text: str | None = None):
    """The one place a production AgentEngine is constructed, so every knob in
    config.yaml reaches every call site. `memory_text` is the already-read
    agent memory (tick_and_reflect reads it once); None means read it here."""
    from .engine import AgentEngine
    from .learn import lessons_from_runs
    from .memory import read_memory

    if cfg.agent.use_lessons:
        mem = read_memory() if memory_text is None else memory_text
        lessons = mem + "\n" + lessons_from_runs()
    else:
        lessons = ""
    return AgentEngine(lessons=lessons, allow_short=cfg.agent.allow_short,
                       market_context=market_context)
```

In `_positions`, replace the whole `if agent is None:` block with:

```python
    if engine == "agent":
        if agent is None:
            agent = _build_agent(cfg)
        return _agent_positions(eval_dir, {s: bars[s] for s in cfg.strategy.universe},
                                agent)
```

In `_agent_positions`, change the `decide_all` call so every bar passes the full universe up to that bar:

```python
        if todo:
            ds = agent.decide_all(todo, {s: bars[s].loc[:ts] for s in todo},
                                  {s: cur[s] for s in todo},
                                  context_histories={s: bars[s].loc[:ts] for s in syms
                                                     if ts in bars[s].index})
```

In `tick_and_reflect`, replace the `if agent is None:` block, gate the reflection, and record the flag:

```python
    memory_text = read_memory(memory_path)
    if agent is None:
        agent = _build_agent(cfg, memory_text=memory_text)
    ctx = bool(getattr(agent, "market_context", False))

    res = tick(cfg, eval_dir, cost_bps, engine=engine, fill=fill, fetch=fetch,
              today=today, cache_dir=cache_dir, agent=agent)

    reflected = False
    # The ctx record shares nothing with the control: it never writes the
    # control's memory file (lessons are off for both, so it loses nothing).
    if res["appended"] >= 1 and not ctx:
        try:
            ...existing reflection body unchanged...
```

and, just before `(eval_dir / "run.json").write_text(...)` at the end of `tick_and_reflect`:

```python
    meta["market_context"] = ctx
```

The now-unused imports in `tick_and_reflect` (`AgentEngine`, `lessons_from_runs`) can be removed from that function's local import line; keep `nvidia_complete`, `read_memory`, `recent_outcomes`, `reflect`.

In `main`, add the flag and build the engine only when it is set (the control path stays exactly as it is, constructing inside `tick_and_reflect`):

```python
    p.add_argument("--market-context", action="store_true",
                   help="agent only: prepend the cross-sectional market block "
                        "(SPY 1d/5d, breadth, momentum rank) to every prompt. "
                        "Use with its own --eval-id (agent_ctx); never on the "
                        "control agent record.")
```

```python
    if not args.report:
        agent = (_build_agent(cfg, market_context=True)
                 if args.market_context else None)
        res = tick_and_reflect(cfg, eval_dir, args.cost_bps, engine=engine,
                               fill=args.fill_mode, agent=agent)
```

- [ ] **Step 4: Run the new tests and the whole suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_forward_market_context.py -q`
Expected: 5 passed.

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: all pass. `tests/test_forward_allow_short.py` and `tests/test_memory.py` exercise the paths you touched; a failure there means `_build_agent` dropped a knob or the reflection gate is wrong.

- [ ] **Step 5: Commit**

```bash
git add src/rhagent/forward.py tests/test_forward_market_context.py
git commit -m "feat(forward): --market-context flag, full-universe context per bar, run.json flag, no reflection for ctx record"
```

---

### Task 3: Schedule it, document the bar, smoke it live

**Files:**
- Modify: `scripts/paper_cron.sh` (inside the `NVIDIA_API_KEY` guard, after the agent tick)
- Modify: `README.md` (records table near the top; "The bar, written down" section)

**Interfaces:**
- Consumes (Task 2): `python -m rhagent.forward --engine agent --eval-id agent_ctx --market-context`.

- [ ] **Step 1: Add the tick to `scripts/paper_cron.sh`**

Inside the existing `if [ -n "${NVIDIA_API_KEY:-}" ]; then` block, after the control agent line and before `else`:

```bash
  # Fourth record: the same agent with a two-line market block in every
  # prompt (spec: docs/superpowers/specs/2026-09-03-agent-market-context-design.md).
  # Own record dir, no reflection, non-fatal like the rest.
  python -m rhagent.forward --engine agent --eval-id agent_ctx --market-context \
    || echo "!! agent_ctx tick failed -- other records still persisted" >&2
```

Check: `bash -n scripts/paper_cron.sh` prints nothing.

- [ ] **Step 2: README records table**

Add a row after the `agent` row:

```markdown
| `agent_ctx` | config | config | The same LLM with a two-line market block (SPY 1d/5d, breadth, momentum rank) in every prompt. Started 2026-09-04; see the bar below. |
```

- [ ] **Step 3: README bar**

In "The bar, written down (2026-09-03)", after the `agent` block and before "The model behind the agent changed", add:

```markdown
**`agent_ctx` (added 2026-09-03, judged on the first scheduled run on or
after 2027-03-02) passes only if all four hold:**

1. At least 90 scored days.
2. On the days both it and `mean_reversion_real` scored, its cumulative net
   return is at least `mean_reversion_real`'s.
3. Fewer than 10% of candidate days excluded for failed decisions.
4. On the days both it and the control `agent` scored, its cumulative net
   return is above the control's. **This is the hypothesis**: that seeing
   the market's own move stops the agent shorting into broad rebounds like
   2026-07-29.

Fails 4: market context did not help; its tick is removed and the next
experiment is news. Passes 4 but fails 2: context helped, the agent is still
not fundable; it becomes the new control. Passes all four: it replaces the
control agent in the ladder above.
```

And extend the judge command's list: `for r in ['mean_reversion_real', 'agent', 'agent_ctx']:`.

- [ ] **Step 4: Live smoke test (needs `.env` with `NVIDIA_API_KEY`)**

```bash
cd "/Users/adijain/robinhood agentic trading" && set -a && . ./.env && set +a && \
PYTHONPATH=src .venv/bin/python - <<'EOF'
from rhagent.config import load
from rhagent.engine import AgentEngine
from rhagent.data import get_bars
cfg = load()
syms = ["AAPL", "JPM", "NVDA", "PFE", "SPY"]
bars = get_bars(syms, "2026-06-01", "2026-09-02", cache_dir="data")
eng = AgentEngine(model=cfg.agent.model, allow_short=True, max_tokens=cfg.agent.max_tokens,
                  market_context=True)
out = eng.decide_all(syms[:4], {s: bars[s] for s in syms[:4]}, {s: 0.0 for s in syms[:4]},
                     context_histories=bars)
for s, d in out.items():
    print(s, d.status, d.target, d.reason[:90])
assert all(d.status == "ok" for d in out.values())
EOF
```

Expected: four `ok` lines. If any is `failed`, read the reason before touching anything; do not ship.

- [ ] **Step 5: Full suite, commit, PR**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
git add scripts/paper_cron.sh README.md
git commit -m "feat: schedule the agent_ctx record and write down its bar (judged 2027-03-02)"
git push -u origin feat/agent-market-context
gh pr create --title "feat: agent_ctx forward record (market-context experiment)" --body "..."
```

PR body must say: what the block is, that the control record's prompt is byte-identical (regression test), the new tick is non-fatal, the bar and its date, and "Not the paper-state banner PR."
