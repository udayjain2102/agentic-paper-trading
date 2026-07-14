import numpy as np
from rhagent.overlay import _fit_logit, _predict_logit

def test_logit_recovers_monotone_separation():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    # P(win) increases in x; label accordingly with noise
    p = 1 / (1 + np.exp(-3 * x))
    y = (rng.uniform(size=n) < p).astype(float)
    X = np.column_stack([np.ones(n), x])          # bias + feature
    beta = _fit_logit(X, y)
    # predictions should be monotone in x: high-x row > low-x row
    lo = _predict_logit(beta, np.array([[1.0, -2.0]]))[0]
    hi = _predict_logit(beta, np.array([[1.0, 2.0]]))[0]
    assert hi > 0.5 > lo
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

def test_logit_handles_separable_without_blowup():
    # perfectly separable data would send unregularized weights to infinity
    X = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    beta = _fit_logit(X, y, l2=1.0)
    assert np.all(np.isfinite(beta))
    p = _predict_logit(beta, X)
    assert np.all((p >= 0.0) & (p <= 1.0))

def test_logit_all_one_class_returns_finite():
    X = np.array([[1.0, 0.3], [1.0, -0.2], [1.0, 0.5]])
    y = np.array([1.0, 1.0, 1.0])
    beta = _fit_logit(X, y)
    assert np.all(np.isfinite(beta))
