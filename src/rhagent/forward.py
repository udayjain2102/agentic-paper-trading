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


def _agent_positions(eval_dir: Path, symbol: str, bars: pd.DataFrame,
                     agent) -> pd.Series:
    """Target-position series for `symbol`, deciding only bars not yet cached.

    Agent decisions are non-deterministic and cost an API call each, so past
    verdicts are frozen to disk (eval_dir/pos_<sym>.csv) and only new bars are
    decided. Keeps a daily forward tick at ~1 call/symbol/day.
    ponytail: one call per uncached bar; catch-up after a long gap costs N calls.
    """
    cache = eval_dir / f"pos_{symbol}.csv"
    if cache.exists():
        prev = pd.read_csv(cache, parse_dates=["date"]).set_index("date")["pos"]
    else:
        # Anchor: don't back-decide a year of history (that's ~N API calls and
        # isn't "forward"). Seed all but the latest bar as flat; decide only new.
        prev = pd.Series(0.0, index=bars.index[:-1])
    pos = float(prev.iloc[-1]) if len(prev) else 0.0
    decided = dict(prev)
    new_rows = []
    for ts in bars.index:
        if ts in decided:
            pos = decided[ts]
            continue
        history = bars.loc[:ts]
        d = agent.decide(symbol, history, pos)
        pos = d.target
        decided[ts] = pos
        new_rows.append({"date": str(ts.date()), "symbol": symbol,
                         "target": pos, "reason": d.reason,
                         "status": getattr(d, "status", "ok")})
    out = pd.Series(decided).reindex(bars.index).astype(float)
    out.rename_axis("date").rename("pos").to_csv(cache)
    # Append-only decisions log. `status` ("ok" vs "failed") makes a genuine
    # verdict distinguishable from a parse-fail/timeout/API-error fallback
    # without sniffing the reason string; agent performance metrics should
    # filter to status == "ok" rather than counting a failed tick as a real
    # flat decision. Rows written before this field existed have no "status"
    # key -- readers should treat a missing key as unknown/legacy, not "ok".
    if new_rows:
        with (eval_dir / "decisions.jsonl").open("a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
    return out


def _positions(cfg, engine: str, bars: dict[str, pd.DataFrame],
               eval_dir: Path, agent=None) -> dict[str, pd.Series]:
    """Per-symbol target-position series for the chosen engine."""
    if engine == "agent":
        if agent is None:
            from .engine import AgentEngine
            from .learn import lessons_from_runs
            from .memory import read_memory

            lessons = read_memory() + "\n" + lessons_from_runs()
            agent = AgentEngine(lessons=lessons)
        return {s: _agent_positions(eval_dir, s, bars[s], agent) for s in cfg.strategy.universe}
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
    return pos


def _net_series(cfg, engine: str, bars: dict[str, pd.DataFrame], cost_bps: float,
                eval_dir: Path, agent=None, fill: str = "close") -> pd.Series:
    """Fully-realized daily net-return series for the chosen engine.

    net_returns records a day's return at its *entry* date, so a day is only
    trustworthy once the next trading bar exists for every leg. Keep only days
    with full coverage: ticking mid-update otherwise appends a thin partial-day
    mean (e.g. 3 of 66 names realized), which misrepresents the basket.

    ponytail: strict full coverage means one chronically-missing name (a symbol
    the feed stops updating) freezes the whole record; upgrade by dropping dead
    names from the universe or switching to a coverage threshold.
    """
    pos = _positions(cfg, engine, bars, eval_dir, agent)
    legs = {s: net_returns(bars[s], pos[s], cost_bps, fill) for s in pos}
    df = pd.concat(legs, axis=1)
    full = df.notna().sum(axis=1) == len(df.columns)
    return df[full].mean(axis=1)


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
    if ret_path.exists():
        prev = pd.read_csv(ret_path, parse_dates=["date"])
        last = prev["date"].max()
        new = net[net.index > last]
    else:
        # Anchor: first tick records only the latest realized day, so the curve
        # starts now rather than backfilling a year of history as "forward".
        prev = pd.DataFrame(columns=["date", "net"])
        new = net.tail(1)

    rows = pd.DataFrame({"date": new.index, "net": new.values})
    combined = pd.concat([prev, rows], ignore_index=True).drop_duplicates("date")
    combined = combined.sort_values("date")
    combined.to_csv(ret_path, index=False)

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
        agent = AgentEngine(lessons=memory_text + "\n" + lessons_from_runs())

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
    meta["memory_chars"] = len(memory_text)
    meta["reflected"] = reflected
    (eval_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return res


def _report(eval_dir: Path) -> None:
    from .evaluate import aggregate, load_run

    meta, trades, net = load_run(eval_dir)
    a = aggregate(trades, net)
    print(f"forward record: {meta['engine']} {','.join(meta['symbols'])}  "
          f"{meta['start'][:10]} -> {meta['end'][:10]}  ({a['n_trades'] or len(net)} days)")
    print(f"  total_return   {a['total_return']:+.2%}")
    print(f"  sharpe         {a['sharpe']:.2f}")
    print(f"  max_drawdown   {a['max_drawdown']:.2%}")
    _report_decision_quality(eval_dir)


def _report_decision_quality(eval_dir: Path) -> None:
    """Print the share of ticks that were API/parse failures, not verdicts.

    A failed tick holds the prior position, so its day still lands in the
    return series -- the P&L is real, but it is the P&L of an outage, not of a
    decision. Printing the failure rate next to the return keeps the headline
    from being read as "the agent chose flat" when it means "the agent never
    answered". Rows predating the `status` field are counted as unknown.
    """
    log = eval_dir / "decisions.jsonl"
    if not log.exists():
        return
    ok = failed = unknown = 0
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        status = json.loads(line).get("status")
        if status == "ok":
            ok += 1
        elif status is None:
            unknown += 1
        else:
            failed += 1
    total = ok + failed + unknown
    if not total:
        return
    print(f"  decisions      {ok}/{total} genuine verdicts"
          f"  ({failed} failed, {unknown} legacy/unknown)")
    if ok < total:
        print(f"  !! {(total - ok) / total:.0%} of ticks were not real decisions -- "
              f"returns above include days the model never answered")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rhagent.forward")
    p.add_argument("--out-dir", default="journal/forward")
    p.add_argument("--eval-id", help="record dir name (default: engine name)")
    p.add_argument("--engine", help="momentum|linreg|mean_reversion|agent "
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
    _report(eval_dir)
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

        # agent path: injected complete() = no API; decisions cached to disk so
        # a second tick decides zero new bars.
        from .engine import AgentEngine
        calls = {"n": 0}
        def complete(_prompt):
            calls["n"] += 1
            return '{"target": 1, "reason": "test"}'
        agent = AgentEngine(complete=complete)
        eda = Path(d) / "agent"
        ta = tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
                  engine="agent", agent=agent)
        assert ta["appended"] == 1, ta
        n_after_first = calls["n"]
        assert n_after_first > 0, "agent should have called the model"
        tick(cfg, eda, today=date(2026, 3, 20), cache_dir=cache,
             engine="agent", agent=agent)
        assert calls["n"] == n_after_first, "cached bars must not re-call model"
    print("forward selfcheck ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selfcheck"]:
        _selfcheck()
    else:
        sys.exit(main())
