"""Rule-based trading strategies derived from the Quant Bible.

Each strategy is a pure function of a price-bar DataFrame; it produces a target
position series and never performs I/O. See ``base.Strategy``.
"""
