#!/usr/bin/env python3
"""E9 -- the reproducibility drill.

What this should do when you are finished:

    uv run python run.py --seed 0 --n-events 50000 --output-dir results

writes results/run_<timestamp>.json holding mu_hat, sigma_hat, log_likelihood,
git_sha and wall_clock. Running it twice with the same seed gives you identical
JSON apart from wall_clock (and the timestamp in the file name).

Checklist:
  [ ] an argparse CLI with --seed, --n-events and --output-dir
  [ ] every random number traces back to --seed (no bare np.random.* calls)
  [ ] git_sha from subprocess, marked dirty if the tree has uncommitted changes
  [ ] wall_clock from time.perf_counter(), not time.time()
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

from week01.data import make_pi0_toy
from week01.fit import fit_pi0_peak, signal_plus_bg


def git_sha():
    """The current commit, with '-dirty' added if the tree has been edited."""
    # TODO (E9.1): subprocess.check_output(["git", "rev-parse", "HEAD"]).
    #   Decide what to return when git is missing or this is not a repo --
    #   quietly returning "unknown" is a fair choice; crashing is not.
    raise NotImplementedError


def log_likelihood(counts, expected):
    """Poisson log-likelihood of the fitted model given the binned data."""
    # TODO (E9.1): sum over the bins of  k*log(lam) - lam - log(k!).
    #   scipy.stats.poisson.logpmf does that whole expression in one call.
    #   fit["popt"] holds the five fitted parameters, so signal_plus_bg with
    #   those parameters is the expected count in each bin.
    raise NotImplementedError


def main():
    # TODO (E9.4): build the parser, then:
    #   1. make a toy with make_pi0_toy(n=args.n_events, seed=args.seed)
    #   2. fit it with fit_pi0_peak
    #   3. build the record dictionary
    #   4. write results/run_<timestamp>.json
    raise NotImplementedError


if __name__ == "__main__":
    main()
