"""Walk-forward bake-off for the price-only model engine.

Scores three strategies on the same bars, same cost basis (7 bps per unit of
turnover), same fill convention (next_open — the mean_reversion_real basis):

  model            price_model.walkforward_positions (monthly refit)
  buy_and_hold     flat 1.0 every day, every symbol
  mean_reversion   config strategy + its configured overlay, exactly as
                   forward._positions builds it for the mean_reversion_real
                   record

All three are reduced to the equal-weight basket mean of per-symbol
net_returns — the same aggregation forward._net_series uses — so the numbers
are comparable to the live records.

    PYTHONPATH=src python scripts/eval_model.py [--threshold 0.5] [--top-k N]
        [--cost-bps 7] [--years 10] [--min-train-rows 20000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rhagent.backtest import net_returns, result_from_returns  # noqa: E402
from rhagent.config import load as load_config  # noqa: E402
from rhagent.data import get_bars  # noqa: E402
from rhagent.overlay import apply_conviction  # noqa: E402
from rhagent.price_model import walkforward_positions  # noqa: E402
from rhagent.strategies import build  # noqa: E402


def _basket(pos_by_sym: dict[str, pd.Series], bars, cost_bps: float,
            fill: str) -> pd.Series:
    legs = {s: net_returns(bars[s], pos_by_sym[s], cost_bps, fill)
            for s in pos_by_sym}
    df = pd.concat(legs, axis=1)
    full = df.notna().sum(axis=1) == len(df.columns)
    return df[full].mean(axis=1)


def _row(name: str, net: pd.Series) -> dict:
    r = result_from_returns(net)
    return {"strategy": name, "total_return": r.total_return, "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown, "hit_rate": r.hit_rate,
            "n_days": r.n_days}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval_model")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--cost-bps", type=float, default=7.0)
    p.add_argument("--years", type=float, default=10.0)
    p.add_argument("--min-train-rows", type=int, default=20_000)
    p.add_argument("--cache-dir", default="data")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = load_config()
    universe = list(cfg.strategy.universe)
    bars = get_bars(universe, "2016-01-01", "2027-01-01",
                    cache_dir=args.cache_dir)

    print(f"model: threshold={args.threshold} top_k={args.top_k} "
          f"cost={args.cost_bps}bps fill=next_open")
    pos_model = walkforward_positions(
        bars, cost_bps=args.cost_bps, threshold=args.threshold,
        top_k=args.top_k, min_train_rows=args.min_train_rows)

    pos_bh = {s: pd.Series(1.0, index=df.index) for s, df in bars.items()}

    strat = build(cfg.strategy.name, cfg.strategy.params)
    pos_mr = {}
    for s in universe:
        pos = strat.positions(bars[s])
        pos_mr[s] = apply_conviction(pos, strat.signal(bars[s])) \
            if getattr(cfg.strategy, "overlay", "none") == "conviction" else pos

    nets = {
        "model": _basket(pos_model, bars, args.cost_bps, "next_open"),
        "buy_and_hold": _basket(pos_bh, bars, args.cost_bps, "next_open"),
        "mean_reversion_real": _basket(pos_mr, bars, args.cost_bps, "next_open"),
    }

    # Exposure diagnostic: mean number of names held per day.
    held = pd.DataFrame({s: pos_model[s] for s in pos_model})
    print(f"model avg names held/day: {held.sum(axis=1).mean():.1f} "
          f"(of {len(universe)})")

    rows = [_row(name, net) for name, net in nets.items()]
    df = pd.DataFrame(rows).set_index("strategy")
    print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

    # Same comparison over the most recent 3 years — regime drift matters
    # more than the full-window number for a go-forward decision.
    cutoff = nets["model"].index.max() - pd.DateOffset(years=3)
    recent = {k: v[v.index >= cutoff] for k, v in nets.items()}
    print("\nlast 3 years:")
    print(pd.DataFrame([_row(k, v) for k, v in recent.items()])
          .set_index("strategy").to_string(float_format=lambda x: f"{x:+.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
