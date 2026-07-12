"""CLI: run the coarse-to-fine strategy search for one strategy, in-sample only.

    python -m rhagent.search --strategy mean_reversion [--horizon 5] [--rounds 4]
        [--icir-floor 0.3] [--half-life-floor 5] [--min-names 10] [--oos-frac 0.25]
        [--days 400] [--symbols A,B,...]

Prints the per-round log, the ranked surviving configs, and the total number of
configs tested (which the sub-project-3 gate corrects for). The out-of-sample
slice (dates >= the cutoff) is never read.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from ..factor.split import oos_cutoff
from ..factor.universe import UNIVERSE, load_universe
from ..strategies import REGISTRY
from .loop import Gates, run_search


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="rhagent.search")
    p.add_argument("--strategy", required=True, choices=sorted(REGISTRY))
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--icir-floor", type=float, default=0.3)
    p.add_argument("--half-life-floor", type=int, default=5)
    p.add_argument("--min-names", type=int, default=10)
    p.add_argument("--oos-frac", type=float, default=0.25)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--symbols", help="comma-separated override of the default universe")
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args(argv)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else UNIVERSE
    )
    end = date.today()
    start = end - timedelta(days=args.days)
    bars_by_symbol, close = load_universe(
        symbols, start.isoformat(), end.isoformat(), cache_dir=args.cache_dir
    )
    cutoff = oos_cutoff(close.index, args.oos_frac)
    close_is = close.loc[close.index < cutoff]
    if len(close_is) == 0:
        p.error("no in-sample days after the out-of-sample split")

    gates = Gates(icir_floor=args.icir_floor, half_life_floor=args.half_life_floor)
    result = run_search(
        args.strategy, bars_by_symbol, close_is,
        horizon=args.horizon, min_names=args.min_names,
        max_rounds=args.rounds, gates=gates,
    )

    print(f"strategy: {result.strategy}   universe: {len(bars_by_symbol)} names   "
          f"in-sample days: {len(close_is)}")
    for rl in result.rounds:
        print(f"\nround {rl.round}: {rl.n_scored} scored, {len(rl.survivors)} survived")
        for s in rl.survivors[:5]:
            print(f"  survive  {s.params}  ICIR={s.icir:+.3f}  half_life={s.half_life}  "
                  f"signs={s.subperiod_ic_signs}")
        for params, gate in rl.rejected[:5]:
            print(f"  reject   {params}  ({gate})")
    print("\ntop survivors (ranked by ICIR):")
    if not result.survivors:
        print("  none passed all gates")
    for s in result.survivors[:10]:
        print(f"  {s.params}  ICIR={s.icir:+.3f}  half_life={s.half_life}")
    print(f"\nconfigs tested: {result.n_tested}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
