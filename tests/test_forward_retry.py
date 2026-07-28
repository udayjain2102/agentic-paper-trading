"""A status="failed" cached decision must be retried on a later tick; a
status="ok" one must stay frozen forever; legacy (no-status) rows must never
be back-decided; and retries must be bounded so a long outage can't trigger
an unbounded catch-up burst.

See forward._agent_positions's RETRY RULE docstring for the design.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rhagent import forward
from rhagent.engine import AgentEngine


def _cfg(universe):
    return SimpleNamespace(strategy=SimpleNamespace(
        name="agent", params={}, universe=universe, overlay="none"),
        agent=SimpleNamespace(use_lessons=False, allow_short=True))


def _bars(idx, seed):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e6}, index=idx)


def _agent(target=1.0):
    def complete(_prompt):
        return "{" + ", ".join(f'"{s}": {target:.0f}' for s in _SYMS) + "}"
    return AgentEngine(complete=complete)


_SYMS = ["AAA", "BBB"]


def test_failed_decision_is_retried_and_ok_stays_frozen():
    idx = pd.date_range("2026-01-01", periods=10, freq="B")
    bars = {"AAA": _bars(idx, 1), "BBB": _bars(idx, 2)}
    eval_dir = Path("/tmp/rhagent_test_retry_ok_vs_failed")
    for p in eval_dir.glob("*"):
        p.unlink()
    eval_dir.mkdir(exist_ok=True)

    # Seed a cache: bar[-2] decided "ok", bar[-1] decided "failed" for AAA;
    # everything before is legacy (no status column).
    ok_ts, failed_ts = idx[-2], idx[-1]
    prev = pd.DataFrame({
        "date": idx[:-1],
        "pos": [0.0] * (len(idx) - 2) + [1.0],
        "status": [None] * (len(idx) - 2) + ["ok"],
    })
    prev.loc[prev["date"] == ok_ts, "status"] = "ok"
    prev.to_csv(eval_dir / "pos_AAA.csv", index=False)
    # BBB: same shape, all ok, so it never blocks AAA's retry via decide_all
    # batching (decide_all is called per-symbol-subset "todo" here anyway).
    prevb = prev.copy()
    prevb.loc[prevb["date"] == ok_ts, "status"] = "ok"
    prevb.to_csv(eval_dir / "pos_BBB.csv", index=False)
    # Append a failed row on disk for the last bar for both symbols directly
    # (simulating what a real failed tick would have appended).
    for s, f in (("AAA", "pos_AAA.csv"), ("BBB", "pos_BBB.csv")):
        df = pd.read_csv(eval_dir / f, parse_dates=["date"])
        df = pd.concat([df, pd.DataFrame(
            {"date": [failed_ts], "pos": [df["pos"].iloc[-1]], "status": ["failed"]})],
            ignore_index=True)
        df.to_csv(eval_dir / f, index=False)

    calls = {"n": 0, "syms": None}
    def complete(_prompt):
        calls["n"] += 1
        return '{"AAA": 1, "BBB": 1}'
    agent = AgentEngine(complete=complete)

    out, excluded = forward._agent_positions(eval_dir, bars, agent)

    assert calls["n"] == 1, "exactly one retry call for the failed bar"
    # the failed bar is now decided ok, target flipped to 1.0 per the fake model
    assert out["AAA"].loc[failed_ts] == 1.0
    status_after = pd.read_csv(eval_dir / "pos_AAA.csv", parse_dates=["date"]
                                ).set_index("date")["status"]
    assert status_after.loc[failed_ts] == "ok"
    # the earlier ok row is untouched -- still frozen at its original value
    assert status_after.loc[ok_ts] == "ok"

    for p in eval_dir.glob("*"):
        p.unlink()
    eval_dir.rmdir()


def test_legacy_rows_are_never_back_decided():
    idx = pd.date_range("2026-01-01", periods=10, freq="B")
    bars = {"AAA": _bars(idx, 1), "BBB": _bars(idx, 2)}
    eval_dir = Path("/tmp/rhagent_test_retry_legacy")
    eval_dir.mkdir(exist_ok=True)

    # Legacy cache: no status column at all, covering every bar but the last.
    for s in _SYMS:
        pd.DataFrame({"date": idx[:-1], "pos": 0.0}).to_csv(
            eval_dir / f"pos_{s}.csv", index=False)

    calls = {"n": 0}
    def complete(_prompt):
        calls["n"] += 1
        return '{"AAA": 1, "BBB": 1}'
    agent = AgentEngine(complete=complete)

    out, excluded = forward._agent_positions(eval_dir, bars, agent)

    # Only the genuinely new last bar is decided -- one call, not one per
    # legacy bar. Legacy history must not be re-decided.
    assert calls["n"] == 1, calls
    status = pd.read_csv(eval_dir / "pos_AAA.csv", parse_dates=["date"]
                          ).set_index("date")["status"]
    assert (status.loc[idx[:-1]] == "legacy").all()

    for p in eval_dir.glob("*"):
        p.unlink()
    eval_dir.rmdir()


def test_retry_is_bounded():
    idx = pd.date_range("2026-01-01", periods=20, freq="B")
    bars = {"AAA": _bars(idx, 1)}
    eval_dir = Path("/tmp/rhagent_test_retry_bound")
    eval_dir.mkdir(exist_ok=True)

    # 8 consecutive failed bars at the tail -- more than RETRY_BOUND (5).
    n_failed = forward.RETRY_BOUND + 3
    failed_idx = idx[-n_failed:]
    df = pd.DataFrame({"date": idx[:-1], "pos": 0.0, "status": "ok"})
    df.loc[df["date"].isin(failed_idx[:-1]), "status"] = "failed"
    df.to_csv(eval_dir / "pos_AAA.csv", index=False)

    calls = {"n": 0}
    def complete(_prompt):
        calls["n"] += 1
        return '{"AAA": 1}'
    agent = AgentEngine(complete=complete)

    out, excluded = forward._agent_positions(eval_dir, bars, agent)

    # RETRY_BOUND failed bars retried, plus the genuinely-new last bar.
    assert calls["n"] == forward.RETRY_BOUND + 1, calls
    status = pd.read_csv(eval_dir / "pos_AAA.csv", parse_dates=["date"]
                          ).set_index("date")["status"]
    still_failed = (status == "failed").sum()
    assert still_failed == n_failed - 1 - forward.RETRY_BOUND, still_failed

    for p in eval_dir.glob("*"):
        p.unlink()
    eval_dir.rmdir()


def test_failed_bar_holds_its_successor_out_but_legacy_does_not():
    """SUCCESSOR RULE: net[T] depends on pos[T-1] via turnover, so the day after
    an unsettled day can't be scored yet. Legacy rows are never retried, so
    their positions can't change and their successors are safe."""
    idx = pd.date_range("2026-01-01", periods=10, freq="B")
    bars = {"AAA": _bars(idx, 1)}
    eval_dir = Path("/tmp/rhagent_test_successor_rule")
    eval_dir.mkdir(exist_ok=True)

    failed_ts, next_ts = idx[3], idx[4]
    # Everything decided ok except one failed bar in the middle. Bound the
    # retry out of the way so the failed row stays failed for this assertion.
    df = pd.DataFrame({"date": idx, "pos": 0.0, "status": "ok"})
    df.loc[df["date"] == failed_ts, "status"] = "failed"
    df.to_csv(eval_dir / "pos_AAA.csv", index=False)

    def boom(_prompt):
        raise ValueError("still down")
    _, excluded = forward._agent_positions(eval_dir, bars, AgentEngine(complete=boom))
    assert failed_ts in excluded
    assert next_ts in excluded, "the day after an unsettled day must be held out"

    # Same shape, but the row is legacy rather than failed -> successor is fine.
    df = pd.DataFrame({"date": idx, "pos": 0.0, "status": "ok"})
    df.loc[df["date"] == failed_ts, "status"] = "legacy"
    df.to_csv(eval_dir / "pos_AAA.csv", index=False)
    _, excluded = forward._agent_positions(eval_dir, bars, AgentEngine(complete=boom))
    assert failed_ts in excluded
    assert next_ts not in excluded, "legacy never changes, so it can't stale its successor"

    for p in eval_dir.glob("*"):
        p.unlink()
    eval_dir.rmdir()
