"""E7 -- the pytest drill, against week01.fit.

Run:  uv run pytest -q tests/test_fit.py

fit_pi0_peak returns a dictionary, so the results come out as fit["mu"],
fit["mu_err"] and so on. (exercises.md writes these as fit.mu and fit.mu_err.)
"""
import numpy as np
import pytest

from week01.data import make_pi0_toy
from week01.fit import fit_pi0_peak

BIN_EDGES = np.linspace(0.05, 0.25, 80)


def test_fit_recovers_mu():
    # TODO: make a toy with a known mu of 0.135, fit it, and check that
    #       abs(fit["mu"] - 0.135) < 3 * fit["mu_err"]
    raise NotImplementedError


def test_fit_positive_sigma():
    # TODO: fit another toy (a different seed) and check fit["sigma"] > 0.
    #       Have a think about why curve_fit can hand you a negative sigma.
    raise NotImplementedError


def test_empty_input_raises():
    # TODO: this one fails until you add the guard to fit_pi0_peak.
    with pytest.raises(ValueError):
        fit_pi0_peak(np.array([]), BIN_EDGES)
