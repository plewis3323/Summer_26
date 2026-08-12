"""Sanity checks on the loaders we were given. These pass as they are.

Run:  uv run pytest -q tests/test_data.py
"""
import numpy as np

from week01.data import make_pi0_toy


def test_toy_has_requested_length():
    assert len(make_pi0_toy(n=1000, seed=0)) == 1000


def test_toy_is_in_range():
    x = make_pi0_toy(n=5000, seed=0)
    assert x.min() >= 0.05
    assert x.max() <= 0.25


def test_toy_is_reproducible():
    first = make_pi0_toy(n=500, seed=7)
    second = make_pi0_toy(n=500, seed=7)
    assert np.array_equal(first, second)


def test_toy_peak_is_where_we_put_it():
    x = make_pi0_toy(n=200000, signal_frac=0.5, seed=1)
    counts, edges = np.histogram(x, bins=100)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = centers[counts.argmax()]
    assert abs(peak - 0.135) < 0.01
