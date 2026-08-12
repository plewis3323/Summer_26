#!/usr/bin/env python3
"""E9 -- the reproducibility drill.  SOLUTION.

    uv run python run.py --seed 0 --n-events 50000 --output-dir results

writes results/run_<timestamp>.json holding mu_hat, sigma_hat, log_likelihood,
git_sha and wall_clock. Two runs at the same seed give identical JSON apart
from wall_clock; two runs at different seeds differ in every field that depends
on the data.

Two rules are what make that true.

Every random number traces back to --seed. One stray np.random.uniform() --
in a starting point, in a bootstrap, in a shuffle -- breaks the guarantee in
the worst possible way, by giving you results that are *nearly* identical.

A git SHA of a commit you have since edited is worse than no SHA at all,
because it claims a reproducibility you do not actually have. Hence -dirty.
"""
import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
from scipy.stats import poisson

from week01.data import make_pi0_toy
from week01.fit import fit_pi0_peak, signal_plus_bg

HERE = os.path.dirname(os.path.abspath(__file__))


def git_sha():
    """The current commit, '-dirty' if the tree is edited, 'unknown' outside a repo.

    Crashing here would make the script useless inside a container or a tarball,
    which is exactly where you most want the rest of the record.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                      stderr=subprocess.DEVNULL, text=True)
        changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=HERE,
                                          stderr=subprocess.DEVNULL, text=True)
    except Exception:  # noqa: BLE001 - no git on the box, or this is not a repo
        return "unknown"

    sha = sha.strip()
    if changes.strip():
        return sha + "-dirty"
    return sha


def log_likelihood(counts, expected):
    """Poisson log-likelihood: sum over bins of k*log(lam) - lam - log(k!).

    poisson.logpmf is that expression, worked out through gammaln so the
    factorial never overflows. Clip lam away from zero first: the background is
    a straight line and is free to go negative where no data holds it up, and a
    single nan poisons the whole sum.
    """
    expected = np.asarray(expected, dtype=float)
    expected = np.clip(expected, 1e-12, None)
    return float(np.sum(poisson.logpmf(counts, expected)))


def main():
    parser = argparse.ArgumentParser(description="Fit one pi0 toy and log the result.")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--n-events", type=int, default=50000, help="events in the toy")
    parser.add_argument("--output-dir", default="results",
                        help="where run_<timestamp>.json is written")
    args = parser.parse_args()

    # perf_counter for measuring how long something took: it only ever counts
    # forward, and a clock adjustment cannot make it jump
    start = time.perf_counter()

    bin_edges = np.linspace(0.05, 0.25, 80)
    m_gg = make_pi0_toy(n=args.n_events, signal_frac=0.20, seed=args.seed)
    fit = fit_pi0_peak(m_gg, bin_edges)

    counts, _edges = np.histogram(m_gg, bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    popt = fit["popt"]
    expected = signal_plus_bg(centers, popt[0], popt[1], popt[2], popt[3], popt[4])

    record = {
        "mu_hat": float(fit["mu"]),
        "mu_err": float(fit["mu_err"]),
        "sigma_hat": float(fit["sigma"]),
        "sigma_err": float(fit["sigma_err"]),
        "signal_count": float(fit["signal_count"]),
        "log_likelihood": log_likelihood(counts, expected),
        "n_events": int(args.n_events),
        "seed": int(args.seed),
        "git_sha": git_sha(),
        "wall_clock": time.perf_counter() - start,
        # "same code, same seed" only reproduces on the same libraries too
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    # seconds are not enough: two runs back to back would overwrite each other
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"_{int(time.time() % 1 * 1e6):06d}"
    out_path = os.path.join(args.output_dir, "run_" + stamp + ".json")
    with open(out_path, "w") as out_file:
        json.dump(record, out_file, indent=2, sort_keys=True)
        out_file.write("\n")

    print("wrote", out_path)
    print(f"  mu_hat = {record['mu_hat']:.6f} +/- {record['mu_err']:.6f}")
    print(f"  sigma_hat = {record['sigma_hat']:.6f}")
    print(f"  logL = {record['log_likelihood']:.3f}")
    print("  git_sha =", record["git_sha"])
    print(f"  wall_clock = {record['wall_clock']:.3f} s")


if __name__ == "__main__":
    main()
