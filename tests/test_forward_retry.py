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
        return f'{{"target": {target:.0f}, "reason": "x"}}'
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
    # BBB: same shape, all ok, so it never blocks AAA's retry -- decide_all
    # fans out one call per symbol still in the "todo" set for a bar.
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

    calls = []
    def complete(_prompt):
        calls.append(_prompt)
        return '{"target": 1, "reason": "x"}'
    agent = AgentEngine(complete=complete)

    out, excluded = forward._agent_positions(eval_dir, bars, agent)

    assert len(calls) == 2, "one retry call per symbol for the one failed bar"
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

    calls = []
    def complete(_prompt):
        calls.append(_prompt)
        return '{"target": 1, "reason": "x"}'
    agent = AgentEngine(complete=complete)

    out, excluded = forward._agent_positions(eval_dir, bars, agent)

    # Only the genuinely new last bar is decided -- one call per symbol for
    # that bar (2), not one per legacy bar. Legacy history must not be
    # re-decided.
    assert len(calls) == 2, calls
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

    calls = []
    def complete(_prompt):
        calls.append(_prompt)
        return '{"target": 1, "reason": "x"}'
    agent = AgentEngine(complete=complete)

    out, excluded = forward._agent_positions(eval_dir, bars, agent)

    # RETRY_BOUND failed bars retried, plus the genuinely-new last bar. One
    # symbol here, so one call per bar.
    assert len(calls) == forward.RETRY_BOUND + 1, calls
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


def test_drift_report_catches_moved_and_vanished_days():
    """A recorded day must stay reproducible. Two ways it stops: its value moves
    (a retry healed the positions under it, or the price cache was revised), or
    it drops out of the recomputed series entirely. A value comparison alone
    cannot see the second, which is the likelier one."""
    idx = pd.date_range("2026-01-01", periods=4, freq="B")
    prev = pd.DataFrame({"date": idx, "net": [0.01, 0.02, 0.03, 0.04]})

    assert forward._report_drift(
        prev, pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)) == 0
    assert forward._report_drift(
        prev, pd.Series([0.01, 0.99, 0.03, 0.04], index=idx)) == 1
    assert forward._report_drift(
        prev, pd.Series([0.01, 0.02, 0.03], index=idx[:3])) == 1
    assert forward._report_drift(
        pd.DataFrame(columns=["date", "net"]), pd.Series(dtype=float)) == 0


def test_drift_report_renders_distinct_strings_for_a_close_pair(capsys):
    """A pair that differs only below the old 5-decimal print precision (e.g.
    -0.00016 vs -0.000160001) must still render as two distinct strings in the
    warning -- the reader needs to see the drift, not just be told it exists."""
    idx = pd.date_range("2026-01-01", periods=1, freq="B")
    prev = pd.DataFrame({"date": idx, "net": [-0.00016]})
    recomputed = pd.Series([-0.00016 + 2e-9], index=idx)

    assert forward._report_drift(prev, recomputed) == 1
    err = capsys.readouterr().err
    line = [l for l in err.splitlines() if "recorded" in l][0]
    recorded_str, recomputed_str = line.split("=")[1].split("but recomputes to")
    assert recorded_str.strip() != recomputed_str.split("(")[0].strip()


def test_retry_serves_the_newest_failed_dates_not_the_oldest():
    """Selection order only bites above RETRY_BOUND simultaneous failures, which
    nothing else reaches -- the selfcheck's failing-call cases exploit ascending
    bar PROCESSING order, which is independent of selection. Pins newest-first:
    oldest-first head-of-line blocks on an unhealable date and starves every
    later one, an absorbing state; newest-first evicts it."""
    import json

    idx = pd.date_range("2026-01-01", periods=12, freq="B")
    bars = {s: _bars(idx, i) for i, s in enumerate(_SYMS)}
    failed = list(idx[2:10])  # 8 > RETRY_BOUND
    eval_dir = Path("/tmp/rhagent_test_retry_order")
    eval_dir.mkdir(exist_ok=True)
    for p in eval_dir.glob("*"):
        p.unlink()
    for s in _SYMS:
        pd.DataFrame({
            "date": idx, "pos": 0.0,
            "status": ["failed" if t in failed else "ok" for t in idx],
        }).to_csv(eval_dir / f"pos_{s}.csv", index=False)

    forward._agent_positions(eval_dir, bars, _agent())

    rows = [json.loads(l) for l
            in (eval_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]
    retried = sorted({r["date"] for r in rows})
    assert len(retried) == forward.RETRY_BOUND, retried
    expected = sorted(str(t.date()) for t in failed[-forward.RETRY_BOUND:])
    assert retried == expected, (
        f"retry must serve the NEWEST {forward.RETRY_BOUND} failed dates; "
        f"got {retried}, want {expected}")

    for p in eval_dir.glob("*"):
        p.unlink()
    eval_dir.rmdir()
