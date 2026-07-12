"""The locked in-sample / out-of-sample date boundary.

The out-of-sample slice is fixed up front and must never be read during signal
development or the search loop — it is reserved for the final gate. in_sample_mask
also trims the boundary so that no in-sample day's forward-return window peeks
across the cutoff into out-of-sample data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def oos_cutoff(dates, oos_frac: float = 0.25) -> pd.Timestamp:
    uniq = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    if len(uniq) == 0:
        raise ValueError("no dates to split")
    idx = int(np.floor(len(uniq) * (1.0 - oos_frac)))
    idx = min(max(idx, 1), len(uniq) - 1)
    return uniq[idx]


def in_sample_mask(index, cutoff, horizon: int) -> pd.Series:
    index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing:
        raise ValueError("in_sample_mask requires a chronologically sorted index")
    n = len(index)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        j = i + horizon
        ok[i] = j < n and index[j] < cutoff
    return pd.Series(ok, index=index)
