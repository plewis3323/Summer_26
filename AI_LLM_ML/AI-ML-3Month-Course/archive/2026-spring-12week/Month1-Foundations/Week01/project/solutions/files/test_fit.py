"""E7 -- the pytest drill, against week01.fit.  SOLUTION.

Run:  uv run pytest -q tests/test_fit.py

The three tests the exercise asks for, plus one that writing them made obvious.

abs(fit["mu"] - 0.135) < 3 * fit["mu_err"] is a statistical claim, not
hand-waving: even with an unbiased fit and a correct covariance it still fails
about 3 times in 1000. That is why every test below pins its seed -- a test
that fails one run in three hundred is worse than no test at all.
"""
import numpy as np
import pytest

from week01.data import make_pi0_toy
from week01.fit import fit_pi0_peak

BIN_EDGES = np.linspace(0.05, 0.25, 80)
MU_TRUE = 0.135


def test_fit_recovers_mu():
    toy = make_pi0_toy(n=50000, signal_frac=0.20, seed=0)
    fit = fit_pi0_peak(toy, BIN_EDGES)
    assert abs(fit["mu"] - MU_TRUE) < 3 * fit["mu_err"]


def test_fit_positive_sigma():
    toy = make_pi0_toy(n=50000, signal_frac=0.20, seed=1)
    fit = fit_pi0_peak(toy, BIN_EDGES)
    assert fit["sigma"] > 0


def test_empty_input_raises():
    with pytest.raises(ValueError):
        fit_pi0_peak(np.array([]), BIN_EDGES)


def test_sigma_is_positive_on_many_seeds():
    """The sign flip depends on the seed, so one seed proves nothing."""
    for seed in range(10):
        toy = make_pi0_toy(n=20000, signal_frac=0.20, seed=seed)
        fit = fit_pi0_peak(toy, BIN_EDGES)
        assert fit["sigma"] > 0, f"negative sigma at seed={seed}"
