import pandas as pd
from rhagent.overlay import IdentityOverlay, build_overlay
from rhagent.engine import Decision


def test_identity_overlay_passes_target_through():
    ov = IdentityOverlay()
    d = Decision(target=1.0, reason="x", conviction=0.5)
    out = ov.adjust("NVDA", pd.DataFrame({"close": [1, 2]}), d, pd.DataFrame())
    assert out == 1.0
    assert ov.name == "none"


def test_build_overlay_none():
    assert build_overlay("none").name == "none"
