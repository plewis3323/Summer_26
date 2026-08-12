"""E2 -- centering helpers.  SOLUTION.

The whole exercise is one keyword:

    x.mean(axis=1)                  -> shape (3,)   the axis is gone
    x.mean(axis=1, keepdims=True)   -> shape (3, 1) the axis is kept, as length 1

NumPy lines shapes up from the right, so for an x of shape (3, 5):

    x - x.mean(axis=1)                 (3,5) - (3,)   5 against 3 -> ValueError
    x - x.mean(axis=1, keepdims=True)  (3,5) - (3,1)  correct
    x - x.mean(axis=0)                 (3,5) - (5,)   correct, but by luck
    x - x.mean(axis=0, keepdims=True)  (3,5) - (1,5)  correct, and it says so

Writing both of them with keepdims means we are not relying on that luck --
which is the same luck that turns into a silent transpose the day somebody
hands the function a square array.
"""
import numpy as np


def center_rows(x):
    """Subtract each row's mean, so every row of the answer sums to about 0."""
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=1, keepdims=True)


def center_cols(x):
    """Subtract each column's mean, so every column sums to about 0."""
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0, keepdims=True)
