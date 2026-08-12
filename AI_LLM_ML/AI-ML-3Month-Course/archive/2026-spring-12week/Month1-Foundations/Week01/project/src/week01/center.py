"""E2 -- centering helpers.

Both functions have to work for any 2D shape: (3, 5), (5, 3) and (1, 7).
The bug you are guarding against is dropping keepdims=True, which quietly
broadcasts along the wrong axis when the array is not square.
"""
import numpy as np


def center_rows(x):
    """Subtract each row's mean, so every row of the answer sums to about 0."""
    raise NotImplementedError("E2: implement me")


def center_cols(x):
    """Subtract each column's mean, so every column sums to about 0."""
    raise NotImplementedError("E2: implement me")
