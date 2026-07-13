"""Forward paper-trading tick: one call per trading day, P&L that accumulates.

The batch harness (papertrade.py) re-replays a whole window; the runner has no
P&L. Neither gives a *forward* track record. This does: each weekday after close
it computes the configured strategy's net return for the newly-realized day and
appends it to a single growing record under journal/forward/<eval_id>/, in the
same format evaluate.py / the dashboard already read.

Anchored at first run so the curve reflects the go-forward period, not backfilled
history. Reuses Pairs.positions_pair + backtest.net_returns (the exact math
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
    for ts in bars.index:
        if ts in decided:
            pos = decided[ts]
            continue
        history = bars.loc[:ts]
        pos = agent.decide(symbol, history, pos).target
        decided[ts] = pos
    out = pd.Series(decided).reindex(bars.index).astype(float)
    out.rename_axis("date").rename("pos").to_csv(cache)
    return out


def _positions(cfg, engine: str, bars: dict[str, pd.DataFrame],
               eval_dir: Path, agent=None) -> dict[str, pd.Series]:
    """Per-symbol target-position series for the chosen engine."""
    if engine == "agent":
        if agent is None:
            from .engine import AgentEngine
            from .learn import lessons_from_runs

            agent = AgentEngine(lessons=lessons_from_runs())
        return {s: _agent_positions(eval_dir, s, bars[s], agent) for s in cfg.strategy.universe}
    if engine == "pairs":
        from .strategies.pairs import Pairs

        a, b = cfg.strategy.universe
        pa, pb = Pairs(**cfg.strategy.params).positions_pair(bars[a], bars[b])
        return {a: pa, b: pb}
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
                eval_dir: Path, agent=None) -> pd.Series:
    """Realized daily net-return series for the chosen engine."""
    pos = _positions(cfg, engine, bars, eval_dir, agent)
    legs = [net_returns(bars[s], pos[s], cost_bps) for s in pos]
    return pd.concat(legs, axis=1).mean(axis=1).dropna()


def tick(cfg, eval_dir: Path, cost_bps: float = 1.0, *, engine: str | None = None,
         fetch=None, today=None, cache_dir="data", agent=None) -> dict:
    """Append newly-realized days to eval_dir/returns.csv. Returns the meta dict."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    engine = engine or cfg.strategy.name
    today = today or date.today()
    start = (today - timedelta(days=400)).isoformat()
    bars = get_bars(cfg.strategy.universe, start, today.isoformat(), fetch=fetch,
                    cache_dir=cache_dir)
    net = _net_series(cfg, engine, bars, cost_bps, eval_dir, agent)

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
        "notional": 10_000.0,
        "start": str(combined["date"].iloc[0]) if len(combined) else "",
        "end": str(combined["date"].iloc[-1]) if len(combined) else "",
    }
    (eval_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (eval_dir / "trades.jsonl").touch()  # evaluate.load_run expects the file
    return {"meta": meta, "appended": len(rows), "total_days": len(combined)}


def _report(eval_dir: Path) -> None:
    from .evaluate import aggregate, load_run

    meta, trades, net = load_run(eval_dir)
    a = aggregate(trades, net)
    print(f"forward record: {meta['engine']} {','.join(meta['symbols'])}  "
          f"{meta['start'][:10]} -> {meta['end'][:10]}  ({a['n_trades'] or len(net)} days)")
    print(f"  total_return   {a['total_return']:+.2%}")
    print(f"  sharpe         {a['sharpe']:.2f}")
    print(f"  max_drawdown   {a['max_drawdown']:.2%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rhagent.forward")
    p.add_argument("--out-dir", default="journal/forward")
    p.add_argument("--eval-id", help="record dir name (default: engine name)")
    p.add_argument("--engine", help="pairs|momentum|linreg|mean_reversion|agent "
                                    "(default: config strategy)")
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--report", action="store_true", help="report only, no tick")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = load()
    if cfg.strategy is None:
        raise SystemExit("no `strategy:` block in config.yaml")
    engine = args.engine or cfg.strategy.name
    eval_dir = Path(args.out_dir) / (args.eval_id or engine)
    if not args.report:
        res = tick(cfg, eval_dir, args.cost_bps, engine=engine)
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
    cfg = SimpleNamespace(strategy=SimpleNamespace(name="pairs", params={},
                                                   universe=["AAA", "BBB"]))
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "cache"
        cache.mkdir()
        for s, f in bars.items():
            f.to_csv(cache / f"{s}.csv", index_label="date")
        ed = Path(d) / "pairs"
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
