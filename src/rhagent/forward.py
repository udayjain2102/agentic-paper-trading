"""Forward paper-trading tick: one call per trading day, P&L that accumulates.

The batch harness (papertrade.py) re-replays a whole window; the runner has no
P&L. Neither gives a *forward* track record. This does: each weekday after close
it computes the configured strategy's net return for the newly-realized day and
appends it to a single growing record under journal/forward/<eval_id>/, in the
same format evaluate.py / the dashboard already read.

Anchored at first run so the curve reflects the go-forward period, not backfilled
history. Reuses backtest.net_returns (the exact math
compare.py ranks with), so forward numbers match the backtest path.

Usage (cache must already be refreshed for today -- see rhagent.refresh):
    PYTHONPATH=src python -m rhagent.forward            # tick + report
    PYTHONPATH=src python -m rhagent.forward --report   # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .backtest import net_returns
from .config import load
from .data import get_bars

_LEGACY = "legacy"

# ponytail: bound on how many distinct failed bars get retried in one tick,
# newest-first. Each retried bar is a fresh agent.decide_all call for every
# symbol still failed on it -- an unbounded catch-up after a week-long outage
# would blow the GitHub Actions job budget. 5 bars (~a trading week) drains a
# normal-length outage in a few ticks; raise if outages start regularly
# exceeding it.
#
# Per-bar cost: one bar is 65 per-symbol calls (engine.py's decide_all), run
# MAX_WORKERS-wide and paced at RATE_LIMIT_PER_MIN=18/min at the client seam.
# The rate limit sets a hard floor of ~65/18 ~= 3.6min per bar, but the
# binding constraint measured 2026-07-31 was worker concurrency, not the
# pacer: a live 13-symbol decide_all took 79s, i.e. ~9.9 calls/min effective
# at MAX_WORKERS=8 against ~48s median latency. That extrapolates to ~6.6min
# per 65-symbol bar and ~40min for a full RETRY_BOUND=5 catch-up (well inside
# the 6h GitHub Actions job budget, and against the 64min the 2026-07-30 run
# burned to record ONE day). If catch-up runtime becomes the problem, raise
# MAX_WORKERS toward the pacer's ceiling (~14 in flight saturates 18/min at
# that latency) before touching RETRY_BOUND -- the pacer, not the pool size,
# is what protects the rate limit.
#
# No per-date attempt cap, deliberately. Newest-first ordering already retires
# an unhealable date structurally, and a cap is measurably WORSE at the rate
# actually observed (it retires transient failures that would have healed).
# Trigger to revisit: one (date, symbol) pair in decisions.jsonl with 3+
# attempts still failing. That log is append-only, so the evidence accrues
# without new code -- it's a grep, not a judgement call.
RETRY_BOUND = 5

# A recorded day recomputed from unchanged positions and unchanged prices is
# bit-identical, so any nonzero difference is a real input change, not roundoff
# -- this is a "not exactly equal" guard with slack for float replay, not a
# materiality threshold. One constant, because the warning prints the tolerance
# it used: two literals would eventually disagree and the report would lie.
_DRIFT_TOL = 1e-9


def _agent_positions(eval_dir: Path, bars: dict[str, pd.DataFrame],
                     agent) -> tuple[dict[str, pd.Series], set]:
    """Target-position series per symbol, plus the dates to exclude from returns.

    One model call per symbol per uncached bar (agent.decide_all), run
    concurrently through decide_all's worker pool and paced under the
    endpoint's ~18/min bucket. The original 65-call design was slow because it
    was strictly SERIAL and unpaced, not because it was per-symbol.

    Agent decisions are non-deterministic and cost an API call, so past verdicts
    are frozen to disk (eval_dir/pos_<sym>.csv, columns date,pos,status) and only
    new bars are decided -- a second tick the same day makes zero calls.

    RETRY RULE: a cached status=="ok" row is frozen forever -- a genuine verdict
    is never re-decided. A cached status=="failed" row (timeout/parse/API error)
    is NOT frozen: it is dropped back into "not yet decided" so the next tick
    retries it, bounded to the newest RETRY_BOUND failed dates per tick so one
    long outage can't trigger an unbounded catch-up burst. Newest-first so an
    unhealable date can't hold a slot forever -- see the ordering comment below.
    Rows with no status
    at all (pos_*.csv written before the column existed, and the seeded anchor
    bars) are legacy -- excluded from returns like a failure, but never
    retried, or the first tick after this change would try to back-decide a
    year of seeded history at real API cost.

    EXCLUSION RULE: a date is dropped from the net-return series if ANY symbol's
    decision for it is not status "ok". A failed call holds the prior position,
    so its P&L is the P&L of an outage, not of a decision; and the net series is
    the equal-weight basket mean, so a basket with one leg's verdict missing
    isn't a basket decision at all. Rows with no status (pos_*.csv written before
    the column existed, and the seeded anchor bars) are unknown/legacy, NOT "ok",
    so they are excluded too -- the forward record starts at the first day the
    agent actually answered for every name.
    """
    syms = list(bars)
    decided: dict[str, dict] = {}
    stat: dict[str, dict] = {}
    for s in syms:
        cache = eval_dir / f"pos_{s}.csv"
        if cache.exists():
            prev = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
            decided[s] = dict(prev["pos"])
            stat[s] = dict(prev["status"]) if "status" in prev else \
                {ts: _LEGACY for ts in prev.index}
        else:
            # Anchor: don't back-decide a year of history (that's ~N API calls
            # and isn't "forward"). Seed all but the latest bar flat; decide only
            # new. Seeded bars are legacy, not "ok" -- nobody decided them.
            decided[s] = {ts: 0.0 for ts in bars[s].index[:-1]}
            stat[s] = {ts: _LEGACY for ts in bars[s].index[:-1]}

    # Un-freeze the newest RETRY_BOUND failed dates (across the whole
    # universe) so this tick's todo-selection below picks them back up.
    # status=="failed" only -- legacy (no status) is left alone.
    # Newest-first. NOT because old dates are unrecordable -- tick()'s
    # interior-gap filter makes any interior date recordable again -- but
    # because oldest-first head-of-line blocks: a date no retry can fix (a
    # prompt that always overflows, a dead symbol) sits at the queue head
    # forever holding a slot, and five of them starve every later failure
    # permanently. That is an absorbing state, reached at any nonzero rate of
    # unhealable failures -- modelled, oldest-first collapses from 5.0 to 0.5
    # recorded days/week once ~5% of failures are unhealable, and past that
    # point the rate stops mattering at all. Newest-first evicts an unhealable
    # date automatically once five newer failures exist, and its own risk (a
    # transient old failure starved during a burst) is self-limiting: the retry
    # set is recomputed from scratch every tick, so a skipped date returns as
    # soon as the backlog drains. That needs arrival to exceed service --
    # ~0.67 failed dates/day against ~4/day of capacity -- and only bites above
    # a ~70% per-chunk failure rate, 3.5x anything observed.
    failed_dates = sorted({ts for s in syms for ts, st in stat[s].items()
                           if st == "failed"}, reverse=True)
    retry = set(failed_dates[:RETRY_BOUND])
    for s in syms:
        for ts in retry:
            if stat[s].get(ts) == "failed":
                del decided[s][ts]
                del stat[s][ts]

    cur = {s: 0.0 for s in syms}
    new_rows = []
    for ts in sorted(set().union(*(b.index for b in bars.values()))):
        todo = [s for s in syms if ts in bars[s].index and ts not in decided[s]]
        if todo:
            ds = agent.decide_all(todo, {s: bars[s].loc[:ts] for s in todo},
                                  {s: cur[s] for s in todo})
            for s in todo:
                d = ds[s]
                decided[s][ts] = d.target
                stat[s][ts] = getattr(d, "status", "ok")
                new_rows.append({"date": str(ts.date()), "symbol": s,
                                 "target": d.target, "reason": d.reason,
                                 "status": stat[s][ts]})
        for s in syms:
            if ts in decided[s]:
                cur[s] = decided[s][ts]

    out, excluded, pending = {}, set(), set()
    for s in syms:
        pos = pd.Series(decided[s]).reindex(bars[s].index).astype(float)
        st = pd.Series(stat[s], dtype=object).reindex(bars[s].index).fillna(_LEGACY)
        pd.DataFrame({"pos": pos, "status": st}).rename_axis("date").to_csv(
            eval_dir / f"pos_{s}.csv")
        out[s] = pos
        excluded |= set(st.index[st != "ok"])
        pending |= set(st.index[st == "failed"])
    # SUCCESSOR RULE: net[T] is a function of pos[T-1] as well as pos[T] --
    # net_returns takes turnover from pos.diff(), and in next_open mode that
    # turnover also selects close-to-close vs open-to-close fills. So scoring
    # the day AFTER a failed day against the held-over position bakes in a
    # number the retry will later invalidate: measured 106bp of drift at
    # next_open (7bp at close, which is only the cost term). Hold the successor
    # out until its predecessor settles -- a hole, not a contaminated value,
    # same principle as the exclusion rule itself. From "failed" only: legacy
    # rows are never retried (see the un-freeze above), so their positions
    # cannot change and their successors cannot go stale.
    if pending:
        idx = pd.Index(sorted(set().union(*(b.index for b in bars.values()))))
        nxt = idx.get_indexer(sorted(pending)) + 1
        excluded |= set(idx[nxt[nxt < len(idx)]])
    # Append-only decisions log. `status` ("ok" vs "failed") makes a genuine
    # verdict distinguishable from a parse-fail/timeout/API-error fallback
    # without sniffing the reason string; agent performance metrics should
    # filter to status == "ok" rather than counting a failed tick as a real
    # flat decision. Rows written before this field existed have no "status"
    # key -- readers should treat a missing key as unknown/legacy, not "ok".
    _append_decisions(eval_dir, new_rows)
    return out, excluded


def _append_decisions(eval_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    if ok_rows and all(r["target"] == 0.0 for r in ok_rows):
        # This is exactly the failure mode that wasted ~260 calls: a config
        # artifact (e.g. allow_short=False silently disabling half the
        # decision space) can zero out every verdict without ever raising.
        print(f"!! all {len(ok_rows)} genuine decisions this tick are flat "
              f"(target=0.0) -- check for a config artifact (e.g. allow_short) "
              f"suppressing non-flat decisions", file=sys.stderr)
    with (eval_dir / "decisions.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _positions(cfg, engine: str, bars: dict[str, pd.DataFrame],
               eval_dir: Path, agent=None) -> tuple[dict[str, pd.Series], set]:
    """Per-symbol target-position series for the chosen engine, plus the dates
    to exclude from the return series (agent-only; rule-based engines never
    fail to decide, so their exclusion set is empty)."""
    if engine == "agent":
        if agent is None:
            from .engine import AgentEngine
            from .learn import lessons_from_runs
            from .memory import read_memory

            lessons = (read_memory() + "\n" + lessons_from_runs()
                       if cfg.agent.use_lessons else "")
            agent = AgentEngine(lessons=lessons, allow_short=cfg.agent.allow_short)
        return _agent_positions(eval_dir, {s: bars[s] for s in cfg.strategy.universe},
                                agent)
    from .strategies import build

    strat = build(engine, cfg.strategy.params)
    pos = {s: strat.positions(bars[s]) for s in cfg.strategy.universe}
    # Apply the configured decision overlay. Only the conviction gate is wired
    # into the forward path (it is a pure function of the signal, so it has an
    # exact vectorized twin); other overlays need the bar-by-bar papertrade loop.
    overlay = getattr(cfg.strategy, "overlay", "none")
    if overlay == "conviction":
        from .overlay import apply_conviction

        pos = {s: apply_conviction(pos[s], strat.signal(bars[s])) for s in pos}
    elif overlay not in ("none", ""):
        raise SystemExit(
            f"overlay {overlay!r} is not wired into the forward path "
            "(only 'conviction' is; use papertrade.py for the others)"
        )
    return pos, set()


def _net_series(cfg, engine: str, bars: dict[str, pd.DataFrame], cost_bps: float,
                eval_dir: Path, agent=None, fill: str = "close") -> pd.Series:
    """Fully-realized daily net-return series for the chosen engine.

    net_returns records a day's return at its *entry* date, so a day is only
    trustworthy once the next trading bar exists for every leg. Keep only days
    with full coverage: ticking mid-update otherwise appends a thin partial-day
    mean (e.g. 3 of 66 names realized), which misrepresents the basket.

    Days whose decisions weren't real decisions are dropped too -- see
    _agent_positions for the exclusion rule.

    ponytail: strict full coverage means one chronically-missing name (a symbol
    the feed stops updating) freezes the whole record; upgrade by dropping dead
    names from the universe or switching to a coverage threshold.
    """
    pos, excluded = _positions(cfg, engine, bars, eval_dir, agent)
    legs = {s: net_returns(bars[s], pos[s], cost_bps, fill) for s in pos}
    df = pd.concat(legs, axis=1)
    full = df.notna().sum(axis=1) == len(df.columns)
    return df[full & ~df.index.isin(excluded)].mean(axis=1)


def _report_drift(prev: pd.DataFrame, net: pd.Series) -> int:
    """Warn when an already-recorded day no longer reproduces. Returns the count.

    Two ways a recorded day can stop matching: its value moved (positions healed
    after a retry, or the price cache was revised under it), or it fell out of
    the recomputed series entirely (newly excluded, or newly failing the
    full-coverage filter because a refresh dropped a symbol's bar). The second
    is the likelier one and a value comparison alone cannot see it.
    """
    if not len(prev):
        return 0
    recorded = prev.set_index("date")["net"]
    recorded = recorded[~recorded.index.duplicated()]
    n = 0
    both = recorded.reindex(net.index).dropna()
    diff = (net.reindex(both.index) - both).abs()
    for d in diff[diff > _DRIFT_TOL].index:
        print(f"!! recorded {d.date()} = {both[d]:+.9f} but recomputes to "
              f"{net[d]:+.9f} (diff {diff[d]:.2e} > tol {_DRIFT_TOL:.0e}) -- "
              f"record kept, investigate", file=sys.stderr)
        n += 1
    for d in sorted(set(recorded.index) - set(net.index)):
        print(f"!! recorded {d.date()} is no longer in the recomputed series "
              f"-- record kept, investigate", file=sys.stderr)
        n += 1
    return n


def tick(cfg, eval_dir: Path, cost_bps: float | None = None, *, engine: str | None = None,
         fill: str | None = None, fetch=None, today=None, cache_dir="data",
         agent=None) -> dict:
    """Append newly-realized days to eval_dir/returns.csv. Returns the meta dict.

    cost_bps/fill default to cfg.strategy's fields (an explicit argument, e.g.
    from --cost-bps/--fill-mode, wins). getattr fallbacks (1.0/"close") keep
    lightweight test configs -- a bare SimpleNamespace with no cost_bps/fill_mode
    -- working unchanged.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    engine = engine or cfg.strategy.name
    if cost_bps is None:
        cost_bps = getattr(cfg.strategy, "cost_bps", 1.0)
    if fill is None:
        fill = getattr(cfg.strategy, "fill_mode", "close")
    today = today or date.today()
    start = (today - timedelta(days=400)).isoformat()
    bars = get_bars(cfg.strategy.universe, start, today.isoformat(), fetch=fetch,
                    cache_dir=cache_dir)
    net = _net_series(cfg, engine, bars, cost_bps, eval_dir, agent, fill)

    ret_path = eval_dir / "returns.csv"
    prev = (pd.read_csv(ret_path, parse_dates=["date"]) if ret_path.exists()
            else pd.DataFrame(columns=["date", "net"]))
    if len(prev):
        # Newer than everything recorded, PLUS interior gaps -- dates inside the
        # recorded range that aren't in prev. A date excluded for a failed
        # decision heals on a later tick (see _agent_positions' RETRY RULE), and
        # a plain `> max()` filter would silently drop it the moment any later
        # date recorded first: the retry window would be one tick, not
        # RETRY_BOUND. Interior-only is what keeps that safe for the rule-based
        # engines, whose net has no exclusions and carries a full year of
        # history -- backfilling before prev's first date would turn a year of
        # backtest into "forward record" on the second tick.
        seen = set(prev["date"])
        new = net[(net.index > prev["date"].max())
                  | ((net.index > prev["date"].min()) & ~net.index.isin(seen))]
    else:
        # Anchor: first tick records only the latest realized day, so the curve
        # starts now rather than backfilling a year of history as "forward".
        # `len(prev)` not `exists()`: an agent run whose every candidate day was
        # excluded writes an empty returns.csv, and max() of nothing is NaT --
        # comparing against which drops every future day forever.
        new = net.tail(1)

    rows = pd.DataFrame({"date": new.index, "net": new.values})
    combined = pd.concat([prev, rows], ignore_index=True).drop_duplicates("date")
    combined = combined.sort_values("date")
    combined.to_csv(ret_path, index=False)

    # AFTER the write, and swallowed: a recorded day should stay reproducible
    # from the positions and prices it came from, but this is a report line and
    # a report line must never cost a day of the forward record (see the
    # paper_cron persist-last ordering). Report drift, never rewrite it --
    # cost_bps/fill are CLI-overridable, so a --cost-bps debugging tick that
    # silently re-priced the overlap would be a worse integrity hole than the
    # one being caught.
    try:
        _report_drift(prev, net)
    except Exception as e:  # noqa: BLE001 -- reporting must not break the tick
        print(f"!! drift check failed (non-fatal): {e}", file=sys.stderr)

    meta = {
        "run_id": eval_dir.name,
        "engine": engine,
        "symbols": list(cfg.strategy.universe),
        "cost_bps": cost_bps,
        "fill_mode": fill,
        "notional": 10_000.0,
        "start": str(combined["date"].iloc[0]) if len(combined) else "",
        "end": str(combined["date"].iloc[-1]) if len(combined) else "",
    }
    (eval_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (eval_dir / "trades.jsonl").touch()  # evaluate.load_run expects the file
    return {"meta": meta, "appended": len(rows), "total_days": len(combined)}


def tick_and_reflect(cfg, eval_dir: Path, cost_bps: float | None = None, *,
                     engine: str | None = None, fill: str | None = None,
                     fetch=None, today=None, cache_dir="data", agent=None,
                     reflect_complete=None,
                     memory_path: str = "journal/agent_memory.md") -> dict:
    """Agent-only wrapper around tick(): feeds prior memory into the day's
    decisions, then -- if the tick actually appended a new day -- writes a
    self-reflection over recent outcomes. Non-agent engines just tick().

    Records `memory_chars`/`reflected` into run.json so each run's meta is an
    audit trail of what education it got. Reflection is best-effort: any
    failure (model, data) is swallowed so it never breaks the tick.
    """
    engine = engine or cfg.strategy.name
    if engine != "agent":
        return tick(cfg, eval_dir, cost_bps, engine=engine, fill=fill, fetch=fetch,
                    today=today, cache_dir=cache_dir, agent=agent)

    from .engine import AgentEngine, nvidia_complete
    from .learn import lessons_from_runs
    from .memory import read_memory, recent_outcomes, reflect

    memory_text = read_memory(memory_path)
    if agent is None:
        agent = AgentEngine(
            lessons=(memory_text + "\n" + lessons_from_runs()
                     if cfg.agent.use_lessons else ""),
            allow_short=cfg.agent.allow_short)

    res = tick(cfg, eval_dir, cost_bps, engine=engine, fill=fill, fetch=fetch,
              today=today, cache_dir=cache_dir, agent=agent)

    reflected = False
    if res["appended"] >= 1:
        try:
            today_d = today or date.today()
            start = (today_d - timedelta(days=400)).isoformat()
            bars = get_bars(cfg.strategy.universe, start, today_d.isoformat(),
                            fetch=fetch, cache_dir=cache_dir)
            outcomes = recent_outcomes(eval_dir, bars)
            # 600 was below the floor: nemotron-super spends 585-826 tokens on
            # chain-of-thought before writing anything (measured live), so the
            # cap was inside the truncation range and the reflection died in
            # its own reasoning -- swallowed by the except below as a stderr
            # line. Budget reasoning + the ~450-token reflection it must emit.
            complete = reflect_complete or nvidia_complete(max_tokens=2000)
            reflected = bool(reflect(complete, memory_path, outcomes, today_d.isoformat()))
        except Exception as e:
            print(f"!! reflection failed (non-fatal): {e}", file=sys.stderr)

    meta = res["meta"]
    # What the engine actually held, not what was merely read off disk. With
    # use_lessons false (config.yaml explains why) the lessons string is empty,
    # and an injected agent never saw memory_path at all -- reporting the file's
    # length either way credits the agent with an education it did not get.
    meta["memory_chars"] = len(getattr(agent, "lessons", ""))
    meta["reflected"] = reflected
    (eval_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return res


def _report(eval_dir: Path) -> None:
    from .evaluate import MIN_RETURN_DAYS_FOR_SHARPE, aggregate, load_run

    meta, trades, net = load_run(eval_dir)
    a = aggregate(trades, net)
    print(f"forward record: {meta['engine']} {','.join(meta['symbols'])}  "
          f"{meta['start'][:10]} -> {meta['end'][:10]}  ({a['n_trades'] or len(net)} days)")
    # sharpe is None until the sample supports it (evaluate.MIN_RETURN_DAYS_FOR_SHARPE);
    # print why rather than a number the sample can't carry -- and never crash the
    # tick over a report line.
    sharpe = (f"{a['sharpe']:.2f}" if a["sharpe"] is not None
              else f"n/a (needs {MIN_RETURN_DAYS_FOR_SHARPE} return-days, have {a['n_return_days']})")
    print(f"  total_return   {a['total_return']:+.2%}")
    print(f"  sharpe         {sharpe}")
    print(f"  max_drawdown   {a['max_drawdown']:.2%}")
    _report_decision_quality(eval_dir)
    _report_record_consistency(eval_dir)


def _report_decision_quality(eval_dir: Path) -> None:
    """Print the share of ticks that were API/parse failures, not verdicts.

    A failed tick is a forced hold, not a decision, so its date is excluded from
    returns.csv entirely (see _agent_positions). This number is therefore the
    cost of the outage in *coverage*, not in P&L: a high failure rate means the
    forward record has holes, not that it is contaminated. Rows predating the
    `status` field are counted as unknown.

    decisions.jsonl is append-only, and a failed (date, symbol) is now
    retryable (see _agent_positions), so a later tick can append a second,
    resolved row for the same (date, symbol). Tally only the latest row per
    (date, symbol) -- otherwise a since-resolved failure keeps inflating the
    failure count forever.
    """
    latest = _latest_decisions(eval_dir)
    if not latest:
        return
    ok = failed = unknown = 0
    for row in latest.values():
        status = row.get("status")
        if status == "ok":
            ok += 1
        elif status is None:
            unknown += 1
        else:
            failed += 1
    total = ok + failed + unknown
    print(f"  decisions      {ok}/{total} genuine verdicts"
          f"  ({failed} failed, {unknown} legacy/unknown)")
    if ok < total:
        print(f"  !! {(total - ok) / total:.0%} of ticks were not real decisions -- "
              f"those days are excluded from the returns above (gaps, not P&L)")


def _latest_decisions(eval_dir: Path) -> dict[tuple[str, str], dict]:
    """Latest decisions.jsonl row per (date, symbol) -- see the append-only /
    retry note on _report_decision_quality. Empty dict if there's no log yet."""
    log = eval_dir / "decisions.jsonl"
    if not log.exists():
        return {}
    latest: dict[tuple[str, str], dict] = {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[(r["date"], r["symbol"])] = r
    return latest


def _report_record_consistency(eval_dir: Path) -> None:
    """Compare decisions.jsonl (the append-only log) against pos_<SYM>.csv
    (the cache the code actually reads) and print any (date, symbol) where
    they disagree. This is the check that would have caught #41: the two
    silently diverged for weeks because nothing ever cross-checked them.

    Seeded/legacy bars (no decisions.jsonl row at all) are expected and NOT a
    divergence -- nobody decided them. Only rows that both sides claim to
    know about, but describe differently, are reported.
    """
    latest = _latest_decisions(eval_dir)
    problems = []
    for cache in sorted(eval_dir.glob("pos_*.csv")):
        sym = cache.name[len("pos_"):-len(".csv")]
        df = pd.read_csv(cache, dtype={"date": str})
        for _, row in df.iterrows():
            key = (row["date"], sym)
            dec = latest.pop(key, None)
            csv_status = row["status"] if pd.notna(row.get("status")) else _LEGACY
            if dec is None:
                if csv_status != _LEGACY:
                    problems.append(f"{key}: pos_{sym}.csv has status={csv_status} "
                                     f"but decisions.jsonl has no row")
                continue
            if dec.get("status", "ok") != csv_status:
                problems.append(f"{key}: status differs -- decisions.jsonl="
                                 f"{dec.get('status', 'ok')} pos_{sym}.csv={csv_status}")
            elif dec["target"] != row["pos"]:
                problems.append(f"{key}: target/pos differ -- decisions.jsonl="
                                 f"{dec['target']} pos_{sym}.csv={row['pos']}")
    # anything left in `latest` is a decision with no matching pos_<SYM>.csv row
    problems += [f"{key}: decisions.jsonl has a row but pos_{key[1]}.csv has none"
                 for key in latest]
    if problems:
        print(f"  !! decisions.jsonl / pos_*.csv disagree on {len(problems)} row(s):")
        for p in problems[:20]:
            print(f"     {p}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rhagent.forward")
    p.add_argument("--out-dir", default="journal/forward")
    p.add_argument("--eval-id", help="record dir name (default: engine name)")
    p.add_argument("--engine", help="momentum|mean_reversion|agent "
                                    "(default: config strategy)")
    p.add_argument("--cost-bps", type=float, default=None,
                   help="per-side cost in bps (default: config.yaml strategy.cost_bps)")
    p.add_argument("--fill-mode", default=None, choices=["close", "next_open"],
                   help="'close' fills at the same bar's close the signal was decided "
                        "from (not really tradable); 'next_open' fills at the following "
                        "bar's open instead (default: config.yaml strategy.fill_mode)")
    p.add_argument("--report", action="store_true", help="report only, no tick")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = load()
    if cfg.strategy is None:
        raise SystemExit("no `strategy:` block in config.yaml")
    engine = args.engine or cfg.strategy.name
    eval_dir = Path(args.out_dir) / (args.eval_id or engine)
    if not args.report:
        res = tick_and_reflect(cfg, eval_dir, args.cost_bps, engine=engine,
                               fill=args.fill_mode)
        print(f"tick: appended {res['appended']} day(s), {res['total_days']} total")
    # The tick above already wrote run.json/trades.jsonl/returns.csv to disk --
    # _report only prints a summary of what's already persisted. A bug in this
    # cosmetic print path must never turn into a nonzero exit: paper_cron.sh
    # runs this call unguarded (set -euo pipefail) and only pushes the
    # refreshed state to the paper-state branch at the very end of the script,
    # so a crash here would silently drop a whole day of forward record even
    # though the tick itself succeeded (see GH Actions run 30246319323).
    try:
        _report(eval_dir)
    except Exception as e:
        print(f"!! report failed (non-fatal, tick already persisted): {e}", file=sys.stderr)
    return 0


def _selfcheck() -> None:
    import tempfile
    from types import SimpleNamespace

    import numpy as np

    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    def frame(seed):
        r = np.random.default_rng(seed).normal(0, 0.01, len(idx))
        close = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame({"open": close, "high": close, "low": close,
                             "close": close, "volume": 1e6}, index=idx)
    bars = {"AAA": frame(1), "BBB": frame(2)}
    cfg = SimpleNamespace(strategy=SimpleNamespace(name="mean_reversion", params={},
                                                   universe=["AAA", "BBB"],
                                                   overlay="none"))
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "cache"
        cache.mkdir()
        for s, f in bars.items():
            f.to_csv(cache / f"{s}.csv", index_label="date")
        ed = Path(d) / "mr"
        r1 = tick(cfg, ed, today=date(2026, 3, 20), cache_dir=cache)
        assert r1["appended"] == 1, r1          # first tick anchors to 1 day
        r2 = tick(cfg, ed, today=date(2026, 3, 20), cache_dir=cache)
        assert r2["appended"] == 0, r2          # idempotent same day

        # agent path: injected complete() = no API. One call per *bar* for the
        # whole universe (not per symbol), decisions cached to disk so a second
        # tick decides zero new bars.
        from .engine import AgentEngine
        calls = {"n": 0}
        def complete(_prompt):
            calls["n"] += 1
            return '{"AAA": 1, "BBB": 1}'
        agent = AgentEngine(complete=complete)
        eda = Path(d) / "agent"

        def seed(n):  # pretend only the first n bars have printed
            for s, f in bars.items():
                f.iloc[:n].to_csv(cache / f"{s}.csv", index_label="date")

        seed(50)
        ta = tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
                  engine="agent", agent=agent)
        # Anchor bars are seeded, not decided: nothing realized+decided yet.
        assert ta["appended"] == 0, ta
        assert calls["n"] == 1, calls  # 1 bar decided, 2 symbols, ONE call
        tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
             engine="agent", agent=agent)
        assert calls["n"] == 1, "cached bars must not re-call model"

        seed(51)  # a new bar prints -> one more call, and bar 50 is now realized
        t2 = tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
                  engine="agent", agent=agent)
        assert calls["n"] == 2, calls
        assert t2["appended"] == 1, t2

        # a failed batched call must EXCLUDE its day, not book it as a flat hold
        def boom(_prompt):
            calls["n"] += 1
            raise ValueError("model down")
        agent.complete = boom
        seed(52)  # bar 51 decided by a failing call
        tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
             engine="agent", agent=agent)
        assert calls["n"] == 3, calls
        assert list(pd.read_csv(eda / "pos_AAA.csv")["status"])[-1] == "failed"

        agent.complete = complete
        seed(53)  # bar 52 decided ok -> bar 51 is now realized, but it failed
        t4 = tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
                  engine="agent", agent=agent)
        # bar 51's failure is retried this tick (one call) alongside the
        # genuinely new bar 52 (another call) -> both resolve ok, so bar 51
        # is no longer excluded and enters returns.csv.
        assert calls["n"] == 5, calls
        assert t4["appended"] == 1, t4
        assert (pd.read_csv(eda / "returns.csv", parse_dates=["date"])["date"]
                == bars["AAA"].index[51]).any()
        assert list(pd.read_csv(eda / "pos_AAA.csv")["status"])[51] == "ok"

        # A bar that fails its retry TOO must still land once it finally
        # resolves -- even though a later bar recorded ahead of it in the
        # meantime. This is the case a plain `> prev.max()` append filter drops
        # forever, making RETRY_BOUND a lie (retry window = 1 tick, not 5).
        start_before = pd.read_csv(eda / "returns.csv",
                                   parse_dates=["date"])["date"].min()
        agent.complete = boom
        seed(54)  # bar 53 decided by a failing call
        tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
             engine="agent", agent=agent)
        assert list(pd.read_csv(eda / "pos_AAA.csv")["status"])[53] == "failed"

        # The precondition for the drop: bar 53's RETRY keeps failing while
        # newer bars succeed and record, moving prev["date"].max() past 53 and
        # stranding it. Bars are decided in ascending date order, so the oldest
        # outstanding bar (53) is always the first call of a tick.
        def fail_oldest(prompt):
            calls["n"] += 1
            if fail_oldest.first:
                fail_oldest.first = False
                raise ValueError("retry down")
            return complete(prompt)

        # Three bars, not two: 53 stays failed, so the SUCCESSOR RULE also holds
        # 54 out, and it takes one more bar before anything records ahead of 53.
        for n in (55, 56, 57):
            fail_oldest.first = True
            agent.complete = fail_oldest
            seed(n)
            tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
                 engine="agent", agent=agent)
        st = list(pd.read_csv(eda / "pos_AAA.csv")["status"])
        assert st[53] == "failed" and st[54] == "ok", st[53:55]
        recorded = pd.read_csv(eda / "returns.csv", parse_dates=["date"])["date"]
        assert recorded.max() > bars["AAA"].index[53], (
            "test precondition: a later bar must record ahead of stranded 53")

        agent.complete = complete
        seed(58)  # bar 53 finally resolves, now behind the record's high-water mark
        tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
             engine="agent", agent=agent)
        rec = set(pd.read_csv(eda / "returns.csv", parse_dates=["date"])["date"])
        assert bars["AAA"].index[53] in rec, (
            "a bar that failed twice must still enter returns.csv once it "
            f"resolves; got {sorted(rec)}")
        # ...and so must its successor, held out by the SUCCESSOR RULE until
        # bar 53 settled. Both land together, computed from final positions.
        assert bars["AAA"].index[54] in rec, sorted(rec)
        assert list(pd.read_csv(eda / "pos_AAA.csv")["status"])[53] == "ok"

        # ...but never backfill BEFORE the record's first date: the anchor bars
        # are legacy, and a rule-based net carries a full year of history.
        first = pd.read_csv(eda / "returns.csv", parse_dates=["date"])["date"].min()
        assert first == start_before, f"record start moved back to {first}"

        # a status=="ok" row must stay frozen forever: re-ticking makes no
        # further calls even though the same bars are still cached.
        n_before = calls["n"]
        tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
             engine="agent", agent=agent)
        assert calls["n"] == n_before, "an ok verdict must never be re-decided"
    print("forward selfcheck ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selfcheck"]:
        _selfcheck()
    else:
        sys.exit(main())
