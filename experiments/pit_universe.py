"""Experiment: how much of the backtest edge is hindsight in the stock list?

The 65-name universe was written down in 2026, so a backtest on it trades 2005
knowing which companies would still be large twenty years on. Here the list is
rebuilt every January from the previous year's data only: the N companies with
the highest median dollar volume, priced above $5. A name is traded only in the
years it qualifies. Rule and parameters are the locked preset, untouched.

This removes the hindsight in CHOOSING names. It does not remove survivorship:
the source only carries stocks still listed around 2024-2026, so companies that
went bankrupt or were bought are still missing. Read the result as an upper
bound that is tighter than the fixed list, not as the true number.

    PYTHONPATH=src uv run --with-requirements requirements.txt --with duckdb --with scikit-learn python experiments/pit_universe.py data_long
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402
from rhagent.backtest import net_returns  # noqa: E402
from rhagent.config import load  # noqa: E402
from rhagent.data import _read_csv  # noqa: E402
from rhagent.overlay import apply_conviction  # noqa: E402
from rhagent.strategies import build  # noqa: E402

BASE = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/US"
YEARS = range(2003, 2026)
SIZES = (65, 200)
WINDOW = slice("2003-01-01", "2025-05-31")
COST_BPS, FILL = 7.0, "next_open"


def fetch(out: Path) -> None:
    import duckdb

    out.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    # Companies only: funds (SPY, QQQ, leveraged ETFs) have no sector in the profile table.
    c.sql(f"""create table px as select symbol, report_date as date, open::double open, high::double high,
        low::double low, close::double close, volume::double volume
        from '{BASE}/stock_prices.parquet'
        where close > 0 and open > 0 and report_date >= '2001-01-01'
          and symbol in (select symbol from '{BASE}/stock_profile.parquet' where coalesce(sector, '') <> '')""")
    c.sql(f"""create table members as
        select year, symbol, rk from (
            select cast(substr(date, 1, 4) as int) + 1 as year, symbol,
                   row_number() over (partition by substr(date, 1, 4)
                                      order by median(close * volume) desc) as rk
            from px group by substr(date, 1, 4), symbol
            having count(*) >= 200 and min(close) >= 5)
        where rk <= {max(SIZES)} and year between {YEARS[0]} and {YEARS[-1]}""")
    members = c.sql("select year, symbol, rk from members order by year, rk").df()
    members.to_csv(out / "_members.csv", index=False)
    for s in members["symbol"].unique():
        c.sql(f"copy (select date,open,high,low,close,volume from px where symbol='{s}' order by date) "
              f"to '{out / (s + '.csv')}' (header)")


def main(data_dir: str) -> None:
    d = Path(data_dir) / "pit"
    if not (d / "_members.csv").exists():
        fetch(d)
    members = pd.read_csv(d / "_members.csv")
    bars = {p.stem: _read_csv(p) for p in sorted(d.glob("*.csv")) if not p.stem.startswith("_")}
    strat = build("mean_reversion", {})
    raw = {s: strat.positions(b) for s, b in bars.items()}
    conv = {s: apply_conviction(raw[s], strat.signal(bars[s])) for s in bars}
    fixed = set(load().strategy.universe)

    def basket(pos: dict, n: int) -> pd.Series:
        """Equal-weight over that year's members; a name is flat outside its member years."""
        mem = members[members["rk"] <= n]
        years_of = mem.groupby("symbol")["year"].apply(set)
        legs = {}
        for s, yrs in years_of.items():
            p = pos[s].where(pos[s].index.year.isin(yrs), 0)
            net = net_returns(bars[s], p, COST_BPS, FILL)
            legs[s] = net.where(net.index.year.isin(yrs))   # NaN outside member years: not in the mean
        return pd.concat(legs, axis=1).mean(axis=1).loc[WINDOW]

    print(f"{len(bars)} distinct names ever in a top-{max(SIZES)} list")
    for n in SIZES:
        mem = members[members["rk"] <= n]
        first, last = set(mem[mem.year == YEARS[0]].symbol), set(mem[mem.year == YEARS[-1]].symbol)
        print(f"\n== top {n} each year: {mem.symbol.nunique()} distinct names; "
              f"{len(first & last)} of the {YEARS[0]} list still in the {YEARS[-1]} list; "
              f"{len(last & fixed)} of the {YEARS[-1]} list are in the fixed 65")
        for name, pos in (("raw", raw), ("conviction", conv)):
            row = ledger.record("pit_universe", f"{name}.top{n}", basket(pos, n),
                                {"n": n, "cost_bps": COST_BPS, "fill": FILL,
                                 "picked_by": "prior-year median dollar volume, min close >= 5, companies only"},
                                note="selection hindsight removed; survivorship (delisted names missing) remains")
            print(f"  {name:<11} sharpe {row['sharpe']:.2f} [{row['sharpe_ci_lo']:.2f}, {row['sharpe_ci_hi']:.2f}]"
                  f"  ann {row['ann_return']:.2%}  maxDD {row['max_drawdown']:.2%}  n={row['n_days']}")


if __name__ == "__main__":
    main(sys.argv[1])
