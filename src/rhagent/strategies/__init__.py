"""Rule-based trading strategies derived from the Quant Bible.

Each strategy is a pure function of a price-bar DataFrame; it produces a target
position series and never performs I/O. See ``base.Strategy``.
"""

from .base import Strategy
from .linreg import LinReg
from .mean_reversion import MeanReversion
from .momentum import Momentum

# Single-symbol strategies only. Pairs is two-symbol and handled separately.
REGISTRY: dict[str, type] = {
    MeanReversion.name: MeanReversion,
    Momentum.name: Momentum,
    LinReg.name: LinReg,
}


def build(name: str, params: dict) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))
