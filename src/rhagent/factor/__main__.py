"""CLI: evaluate a strategy's in-sample cross-sectional ICIR and decay.

    python -m rhagent.factor --strategy momentum [--horizon 5] [--oos-frac 0.25]
                             [--days 400] [--min-names 10] [--symbols A,B,...]

Loads the universe, builds the strategy's signal panel, restricts to the locked
in-sample slice, and prints ICIR (with interpretation bands) and the IC decay
curve. The out-of-sample slice is never touched here.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from ..strategies import REGISTRY, build
from .ic import ic_decay, ic_series, half_life, icir
from .signals import signal_panel
from .split import in_sample_mask, oos_cutoff
from .universe import UNIVERSE, load_universe


def _band(x: float) -> str:
    a = abs(x)
    if a > 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    return "likely noise"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="rhagent.factor")
    p.add_argument("--strategy", required=True, choices=sorted(REGISTRY))
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--oos-frac", type=float, default=0.25)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--min-names", type=int, default=10)
    p.add_argument("--symbols", help="comma-separated override of the default universe")
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args(argv)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else UNIVERSE
    )
    strat = build(args.strategy, {})

    end = date.today()
    start = end - timedelta(days=args.days)
    bars_by_symbol, close = load_universe(
        symbols, start.isoformat(), end.isoformat(), cache_dir=args.cache_dir
    )
    try:
        panel = signal_panel(strat, bars_by_symbol, close.index)
    except NotImplementedError:
        p.error(f"strategy {args.strategy!r} does not implement signal()")

    cutoff = oos_cutoff(close.index, args.oos_frac)
    mask = in_sample_mask(close.index, cutoff, args.horizon)
    is_days = close.index[mask.to_numpy()]
    if len(is_days) == 0:
        p.error("no in-sample days after applying the out-of-sample split")

    sig_is = panel.loc[is_days]
    close_is = close.loc[close.index < cutoff]

    ic = ic_series(sig_is, close_is, args.horizon, args.min_names)
    score = icir(ic)
    decay = ic_decay(sig_is, close_is, min_names=args.min_names)
    hl = half_life(decay)

    approx_indep_obs = len(ic) // args.horizon if args.horizon > 0 else len(ic)

    print(f"strategy: {args.strategy}   universe: {len(bars_by_symbol)} names")
    print(
        f"in-sample days: {len(is_days)}   IC observations: {len(ic)}"
        f"   approx. non-overlapping observations: {approx_indep_obs}"
    )
    print(
        "caveat: for horizon h>1 the daily IC observations use overlapping "
        "forward windows, so consecutive ICs are autocorrelated; the effective "
        "independent sample is roughly (in-sample days / horizon), and the "
        "ICIR band below overstates statistical evidence."
    )
    if len(ic) == 0:
        print(f"\nICIR (h={args.horizon}): insufficient data (no day had >= {args.min_names} valid names)")
    else:
        print(f"\nICIR (h={args.horizon}): {score:+.3f}  [{_band(score)}]")
    print(f"mean IC (h={args.horizon}): {ic.mean() if len(ic) else float('nan'):+.4f}")
    print("\nIC decay (mean IC by horizon):")
    for h, v in decay.items():
        print(f"  h={h:<3} {v:+.4f}")
    print(f"half-life: {hl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
