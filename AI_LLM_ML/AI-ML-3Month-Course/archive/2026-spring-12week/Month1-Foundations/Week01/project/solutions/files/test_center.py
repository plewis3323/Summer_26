"""E2 -- tests that catch the missing-keepdims bug.  SOLUTION.

Run:  uv run pytest -q tests/test_center.py

The shapes are on purpose not square. A (5, 5) test passes even against the
buggy version, because on a square array the wrong broadcast is only a
transpose -- a legal shape, finite numbers, wrong answer. (3, 5) and (5, 3) are
what make the bug either raise or show up. (1, 7) is the degenerate case.

atol, not rtol: the number we are aiming at is 0, and any relative tolerance
against 0 is meaningless.
"""
import numpy as np
import pytest

from week01.center import center_cols, center_rows

SHAPES = [(3, 5), (5, 3), (1, 7)]


def make(shape):
    """The same test data every run, and deliberately not symmetric."""
    rng = np.random.default_rng(shape[0] * 100 + shape[1])
    return rng.standard_normal(shape) * 10 + 3


def test_center_rows():
    for shape in SHAPES:
        out = center_rows(make(shape))
        np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-12)


def test_center_cols():
    for shape in SHAPES:
        out = center_cols(make(shape))
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)


def test_shape_is_preserved():
    for shape in SHAPES:
        x = make(shape)
        assert center_rows(x).shape == shape
        assert center_cols(x).shape == shape


def test_does_not_mutate_input():
    x = make((3, 5))
    before = x.copy()
    center_rows(x)
    center_cols(x)
    np.testing.assert_array_equal(x, before)


def test_dropping_keepdims_is_a_bug():
    """The bug this file exists to catch, written out as a test."""
    x = make((3, 5))
    with pytest.raises(ValueError):
        x - x.mean(axis=1)          # not square, so NumPy catches it for us

    square = make((5, 5))
    wrong = square - square.mean(axis=1)    # square, so NumPy does not
    assert wrong.shape == square.shape
    assert not np.allclose(wrong.mean(axis=1), 0, atol=1e-12)
