"""CLI: run the full out-of-sample gate for one strategy.

    python -m rhagent.gate --strategy mean_reversion [--horizon 5] [--min-names 10]
        [--oos-frac 0.25] [--rounds 4] [--icir-floor 0.3] [--half-life-floor 5]
        [--alpha 0.05] [--dsr-threshold 0.95] [--days 400] [--symbols A,B,...]

Runs the in-sample search, tests survivors on the locked out-of-sample slice with
the Bonferroni and Deflated-Sharpe corrections, and prints the verdict table plus
the count of viable configs. On a thin universe expect zero viable — the gate is
telling you there isn't enough evidence.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from ..factor.universe import UNIVERSE, load_universe
from ..strategies import REGISTRY
from .gate import run_gate


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="rhagent.gate")
    p.add_argument("--strategy", required=True, choices=sorted(REGISTRY))
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--min-names", type=int, default=10)
    p.add_argument("--oos-frac", type=float, default=0.25)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--icir-floor", type=float, default=0.3)
    p.add_argument("--half-life-floor", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dsr-threshold", type=float, default=0.95)
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
    result = run_gate(
        args.strategy, bars_by_symbol, close,
        horizon=args.horizon, min_names=args.min_names, oos_frac=args.oos_frac,
        rounds=args.rounds, icir_floor=args.icir_floor,
        half_life_floor=args.half_life_floor, alpha=args.alpha,
        dsr_threshold=args.dsr_threshold,
    )

    print(f"strategy: {result.strategy}   universe: {len(bars_by_symbol)} names   "
          f"configs tested: {result.n_tested}")
    print(f"corrections: Bonferroni alpha/N, Deflated-Sharpe > {args.dsr_threshold}")
    if not result.rows:
        print("  no survivors from the in-sample search")
    for r in result.rows:
        tag = "VIABLE" if r.viable else r.reason
        print(f"  {r.params}  IS_ICIR={r.is_icir:+.3f}  OOS_ICIR={r.oos_icir:+.3f}  "
              f"half_life={r.oos_half_life}  bonf_p={r.bonf_p:.2e}<{r.bonf_threshold:.2e}?"
              f"{r.bonf_pass}  DSR={r.dsr:.3f}  -> {tag}")
    print(f"\nviable: {len(result.viable)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
