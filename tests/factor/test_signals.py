import pandas as pd

from rhagent.factor.signals import signal_panel
from rhagent.strategies import build


def _bars(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", name="date")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


def test_signal_panel_shape_and_alignment():
    bars = {"AAA": _bars([100.0 + i for i in range(50)]),
            "BBB": _bars([200.0 - i for i in range(50)])}
    idx = bars["AAA"].index
    panel = signal_panel(build("momentum", {}), bars, idx)
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel.index.equals(idx)
    # momentum(40) signal is defined (non-NaN) once past warmup
    assert not panel.iloc[-1].isna().any()
