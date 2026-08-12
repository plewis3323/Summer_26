"""E2 -- tests that catch the missing-keepdims bug.

Run:  uv run pytest -q tests/test_center.py
"""
import numpy as np

from week01.center import center_cols, center_rows

SHAPES = [(3, 5), (5, 3), (1, 7)]


def make(shape):
    """The same test data every run, and deliberately not symmetric."""
    rng = np.random.default_rng(shape[0] * 100 + shape[1])
    return rng.standard_normal(shape) * 10 + 3


def test_center_rows():
    # TODO: for each shape in SHAPES, check that every ROW of
    #       center_rows(make(shape)) has a mean of about 0.
    #       np.testing.assert_allclose(..., atol=1e-12) is the tool here
    #       (atol, not rtol: we are comparing against 0, and a relative
    #       tolerance against 0 means nothing).
    raise NotImplementedError


def test_center_cols():
    # TODO: the same idea, but for columns.
    raise NotImplementedError


def test_shape_is_preserved():
    # TODO: centering must not change the shape of the input.
    raise NotImplementedError


def test_does_not_mutate_input():
    # TODO: centering returns a new array and leaves the caller's x alone.
    #       (This is the bug you get from writing x -= x.mean(...).)
    raise NotImplementedError
