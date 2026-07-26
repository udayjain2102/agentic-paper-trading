"""Per-symbol LLM calls in the forward tick dispatch concurrently under a
shared rate limit, while bars *within* one symbol stay strictly sequential
(each bar's decision feeds the next bar's current_pos -- see forward.py's
_decide_new_agent_rows docstring)."""

import json
import re
import threading
import time

import numpy as np
import pandas as pd

from rhagent import forward
from rhagent.engine import AgentEngine


def _bars(n=5, seed=0):
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return pd.DataFrame({"open": close, "close": close})


def _symbol_of(prompt: str) -> str:
    m = re.search(r"position in (\w+)\.", prompt)
    return m.group(1)


def test_symbols_dispatch_concurrently_but_bars_stay_ordered(tmp_path):
    universe = ["AAA", "BBB", "CCC"]
    bars = {s: _bars(5, seed=i) for i, s in enumerate(universe)}
    # Pre-seed each symbol's cache with the first 2 bars already decided, so
    # 3 bars remain new per symbol (a fresh cache would anchor to only the
    # latest bar -- see _read_prev_positions -- giving just 1 call/symbol,
    # not enough to exercise in-symbol ordering).
    for s in universe:
        seeded = pd.Series(0.0, index=bars[s].index[:2])
        seeded.rename_axis("date").rename("pos").to_csv(tmp_path / f"pos_{s}.csv")

    events = []
    lock = threading.Lock()
    seen_pos = {s: [] for s in universe}

    def complete(prompt):
        symbol = _symbol_of(prompt)
        start = time.monotonic()
        time.sleep(0.03)  # hold the "call" open long enough to observe overlap
        end = time.monotonic()
        with lock:
            events.append((symbol, start, end))
        return '{"target": 1, "reason": "x"}'

    agent = AgentEngine(complete=complete)
    # Wrap decide() to record the current_pos each call saw, per symbol, so we
    # can confirm bars were fed in order (no bar-level parallelism).
    real_decide = agent.decide
    def decide(symbol, history, current_pos):
        seen_pos[symbol].append((len(history), current_pos))
        return real_decide(symbol, history, current_pos)
    agent.decide = decide

    forward._agent_positions_parallel(tmp_path, universe, bars, agent,
                                      rate_limit_per_min=1_000_000)

    assert len(events) == len(universe) * 3  # 3 uncached bars/symbol

    # Concurrency: at least one pair of calls from *different* symbols overlap.
    overlapped = False
    for i, (sym_a, sa, ea) in enumerate(events):
        for sym_b, sb, eb in events[i + 1:]:
            if sym_a != sym_b and sa < eb and sb < ea:
                overlapped = True
    assert overlapped, "expected calls from different symbols to overlap in time"

    # Ordering: within a symbol, history length strictly increases call over
    # call (i.e. bars were never processed out of order or in parallel).
    for s in universe:
        lengths = [n for n, _ in seen_pos[s]]
        assert lengths == sorted(lengths) == [3, 4, 5]


def test_rate_limiter_caps_call_spacing():
    clock = {"t": 0.0}
    sleeps = []

    def time_func():
        return clock["t"]

    def sleep_func(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    limiter = forward.RateLimiter(60, time_func=time_func, sleep_func=sleep_func)
    # 60/min => one call every 1.0s. Fire 5 calls back-to-back on a clock that
    # never advances on its own; the limiter must insert the spacing itself.
    for _ in range(5):
        limiter.wait()

    # First call needs no wait; each of the remaining 4 is spaced by
    # min_interval on the (otherwise frozen) fake clock.
    assert len(sleeps) == 4
    assert all(s == 1.0 for s in sleeps)
    assert clock["t"] == 4.0


def test_failing_symbol_does_not_corrupt_or_block_others(tmp_path):
    universe = ["GOOD1", "BAD", "GOOD2"]
    bars = {s: _bars(3, seed=i) for i, s in enumerate(universe)}

    def complete(prompt):
        symbol = _symbol_of(prompt)
        if symbol == "BAD":
            raise RuntimeError("simulated model outage")
        return '{"target": 1, "reason": "ok"}'

    agent = AgentEngine(complete=complete)
    results = forward._agent_positions_parallel(tmp_path, universe, bars, agent,
                                                rate_limit_per_min=1_000_000)

    assert set(results) == set(universe)
    # 3-bar frame with a fresh cache: only the latest bar is a genuine
    # decision (see _read_prev_positions' anchor), so check that one.
    assert results["GOOD1"].iloc[-1] == 1.0
    assert results["GOOD2"].iloc[-1] == 1.0
    # BAD held its (flat) starting position rather than raising/blocking.
    assert results["BAD"].iloc[-1] == 0.0

    rows = [json.loads(l) for l in
            (tmp_path / "decisions.jsonl").read_text().splitlines()]
    by_symbol = {}
    for r in rows:
        by_symbol.setdefault(r["symbol"], []).append(r)
    assert all(r["status"] == "ok" for r in by_symbol["GOOD1"])
    assert all(r["status"] == "failed" for r in by_symbol["BAD"])
    assert "simulated model outage" in by_symbol["BAD"][0]["reason"]

    # decisions.jsonl is written once, sorted by (date, symbol) -- deterministic.
    keys = [(r["date"], r["symbol"]) for r in rows]
    assert keys == sorted(keys)


def test_decisions_log_deterministic_across_symbols(tmp_path):
    universe = ["ZZZ", "AAA", "MMM"]
    bars = {s: _bars(3, seed=i) for i, s in enumerate(universe)}

    def complete(prompt):
        return '{"target": 1, "reason": "x"}'

    agent = AgentEngine(complete=complete)
    forward._agent_positions_parallel(tmp_path, universe, bars, agent,
                                      rate_limit_per_min=1_000_000)
    rows = [json.loads(l) for l in
            (tmp_path / "decisions.jsonl").read_text().splitlines()]
    keys = [(r["date"], r["symbol"]) for r in rows]
    assert keys == sorted(keys)
    for s in universe:
        cache = tmp_path / f"pos_{s}.csv"
        assert cache.exists()
