"""Append-only ledger of research experiments, so any two runs can be compared.

Every recorded variant keeps its full daily return series (runs/<id>/returns.csv)
and one summary row in results.jsonl. Keeping the series, not just the Sharpe,
is what lets a later run be tested against an earlier one on the same days
(`compare`) instead of eyeballing two numbers.

    PYTHONPATH=src python experiments/ledger.py          # print the table
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rhagent.backtest import result_from_returns
from rhagent.evaluate_robust import bootstrap_sharpe_ci

HERE = Path(__file__).resolve().parent
LEDGER, RUNS = HERE / "results.jsonl", HERE / "runs"


def metrics(net: pd.Series) -> dict:
    r = result_from_returns(net)
    lo, hi = bootstrap_sharpe_ci(net)
    return {
        "sharpe": round(r.sharpe, 4), "sharpe_ci_lo": lo and round(lo, 4),
        "sharpe_ci_hi": hi and round(hi, 4),
        "ann_return": round((1 + r.total_return) ** (252 / max(len(net), 1)) - 1, 6),
        "max_drawdown": round(r.max_drawdown, 6), "n_days": len(net),
        "start": str(net.index[0].date()), "end": str(net.index[-1].date()),
    }


def compare(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0) -> dict:
    """Sharpe(a) - Sharpe(b) on their common days, with a paired bootstrap CI.
    Paired: the same resampled days go to both, so shared market noise cancels."""
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    x = df.to_numpy()
    sr = lambda m: m.mean(0) / m.std(0, ddof=1) * np.sqrt(252)  # noqa: E731
    rng = np.random.default_rng(seed)
    d = np.array([np.subtract(*sr(x[rng.integers(0, len(x), len(x))])) for _ in range(n)])
    return {"diff": round(float(np.subtract(*sr(x))), 4),
            "ci_lo": round(float(np.percentile(d, 2.5)), 4),
            "ci_hi": round(float(np.percentile(d, 97.5)), 4), "n_days": len(df)}


def record(experiment: str, variant: str, net: pd.Series, config: dict, note: str = "") -> dict:
    now = datetime.now(timezone.utc)
    run_id = f"{now:%Y-%m-%dT%H-%M-%SZ}-{experiment}-{variant}"
    (RUNS / run_id).mkdir(parents=True)
    net.rename("net").rename_axis("date").to_csv(RUNS / run_id / "returns.csv")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                         capture_output=True, text=True).stdout.strip()
    row = {"run_id": run_id, "created": now.isoformat(timespec="seconds"), "git": sha,
           "experiment": experiment, "variant": variant, **metrics(net),
           "config": config, "note": note}
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load(run_id: str) -> pd.Series:
    return pd.read_csv(RUNS / run_id / "returns.csv", parse_dates=["date"]).set_index("date")["net"]


def table() -> pd.DataFrame:
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    cols = ["created", "experiment", "variant", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
            "ann_return", "max_drawdown", "start", "end", "git"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    print(table().to_string(index=False))
