"""Experiment: does the locked mean-reversion rule work on stocks it was never
chosen on? Fresh evidence without waiting for forward days.

Two baskets from the public defeatbeta/yahoo-finance-data dataset, neither of
which touched the strategy's selection:
  us_other  the 200 most-traded US stocks outside the 65-name universe
  hk        the 60 most-traded Hong Kong stocks
Names are picked by median dollar volume in 2015-2019 and must have history
back to 2005. Rule and parameters are exactly the locked preset; nothing is
re-tuned. Hong Kong is also scored at 20bp a side (stamp duty is ~10bp alone).

    PYTHONPATH=src uv run --with-requirements requirements.txt --with duckdb --with scikit-learn python experiments/other_markets.py data_long
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402
from trade_filter import basket  # noqa: E402
from rhagent.config import load  # noqa: E402
from rhagent.data import _read_csv  # noqa: E402
from rhagent.overlay import apply_conviction  # noqa: E402
from rhagent.strategies import build  # noqa: E402

BASE = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data"
MARKETS = {"us_other": ("US", 200, (7.0,)), "hk": ("HK", 60, (7.0, 20.0))}
WINDOW = slice("2006-01-01", "2025-05-31")


def fetch(market: str, region: str, n: int, out: Path, exclude: list[str]) -> None:
    import duckdb

    out.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    src = f"'{BASE}/{region}/stock_prices.parquet'"
    c.sql(f"""create table picks as
        select symbol from {src}
        where close > 0 and volume > 0
        group by symbol
        having min(report_date) <= '2005-01-03' and max(report_date) >= '2025-05-30'
           and count(*) filter (where report_date between '2015-01-01' and '2019-12-31') > 1100
           and symbol not in {tuple(exclude)}
        order by median(close * volume) filter (where report_date between '2015-01-01' and '2019-12-31') desc
        limit {n}""")
    c.sql(f"""create table p as select symbol, report_date as date, open::double open, high::double high,
        low::double low, close::double close, volume::double volume
        from {src} where symbol in (select symbol from picks) and close > 0 and open > 0""")
    for (s,) in c.sql("select symbol from picks").fetchall():
        c.sql(f"copy (select date,open,high,low,close,volume from p where symbol='{s}' order by date) "
              f"to '{out / (s + '.csv')}' (header)")


def main(data_dir: str) -> None:
    universe = list(load().strategy.universe)
    strat = build("mean_reversion", {})
    for market, (region, n, costs) in MARKETS.items():
        d = Path(data_dir) / market
        if not d.exists():
            fetch(market, region, n, d, universe)
        bars = {p.stem: _read_csv(p) for p in sorted(d.glob("*.csv"))}
        raw = {s: strat.positions(b) for s, b in bars.items()}
        conv = {s: apply_conviction(raw[s], strat.signal(bars[s])) for s in bars}
        print(f"\n== {market}: {len(bars)} names, e.g. {', '.join(sorted(bars)[:8])}")
        for cost in costs:
            for name, pos in (("raw", raw), ("conviction", conv)):
                net = basket(bars, pos, cost=cost).loc[WINDOW]
                row = ledger.record("other_markets", f"{name}.{market}.{cost:g}bp", net,
                                    {"market": market, "names": len(bars), "cost_bps": cost,
                                     "fill": "next_open", "picked_by": "median dollar volume 2015-2019"},
                                    note="names still listed in 2025 (survivorship); dividends excluded")
                print(f"  {name:<11}{cost:>3g}bp  sharpe {row['sharpe']:.2f} [{row['sharpe_ci_lo']:.2f}, "
                      f"{row['sharpe_ci_hi']:.2f}]  ann {row['ann_return']:.2%}  maxDD {row['max_drawdown']:.2%}  n={row['n_days']}")


if __name__ == "__main__":
    main(sys.argv[1])
