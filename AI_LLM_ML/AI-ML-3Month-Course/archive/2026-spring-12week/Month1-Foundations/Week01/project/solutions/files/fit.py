"""Fitting a gaussian peak on a straight-line background.  SOLUTION (E7 step 3).

Four changes from the scaffold, each one driven by something that was wrong:

1. ValueError on empty input -- test_empty_input_raises asked for it.
   np.histogram([]) does not raise, it hands back a row of zeros, so without
   the guard curve_fit gets a nicely shaped problem with no information in it
   and dies somewhere deep inside MINPACK. Better to fail at the door.
2. abs(sigma). The model only ever sees sigma**2, so -sigma fits exactly as
   well as +sigma; which one you get depends on which side of the valley the
   optimiser happened to wander into.
3. absolute_sigma=True. Poisson bin errors are real errors, not relative
   weights, so curve_fit's default rescaling of the covariance by the reduced
   chi-square is answering a question nobody asked. This is the line that makes
   E5's pull width mean something.
4. background_count divided by the bin width. README section 8 multiplies
   counts-per-bin by a width in GeV, which gives counts*GeV/bin -- a units bug.
   Check the units of every yield you report.
"""
import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, mu, sigma, norm):
    return norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def background(x, a, b):
    return a + b * x


def signal_plus_bg(x, mu, sigma, norm, a, b):
    return gaussian(x, mu, sigma, norm) + background(x, a, b)


def fit_pi0_peak(m_gg, bin_edges):
    """Fit a gaussian peak on a straight line to binned photon-pair masses.

    Returns a dictionary with keys mu, mu_err, sigma, sigma_err, signal_count,
    background_count, popt and pcov.

    Raises ValueError if the sample is empty, if bin_edges cannot make a bin,
    or if no event lands inside the binning.
    """
    m_gg = np.asarray(m_gg, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if m_gg.size == 0:
        raise ValueError("fit_pi0_peak: empty sample, there is nothing to fit")
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("fit_pi0_peak: bin_edges must make at least one bin")

    counts, _edges = np.histogram(m_gg, bins=bin_edges)
    if counts.sum() == 0:
        raise ValueError("fit_pi0_peak: no events land inside bin_edges")

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # starting guesses: peak near 0.135 GeV, about 10 MeV wide, flat background
    p0 = [0.135, 0.010, counts.max(), 1.0, 0.0]
    # sqrt(counts + 1) is the Poisson error on each bin, with the +1 so an
    # empty bin gets weight 1 instead of infinite weight
    bin_errors = np.sqrt(counts + 1)
    popt, pcov = curve_fit(signal_plus_bg, centers, counts, p0=p0,
                           sigma=bin_errors, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    mu = popt[0]
    sigma = abs(popt[1])    # only sigma**2 is in the model, so pick the physical sign
    norm = popt[2]
    a = popt[3]
    b = popt[4]

    bin_width = bin_edges[1] - bin_edges[0]     # assumes even bins, which E5 and E7 use
    # the area under a gaussian is norm * sigma * sqrt(2 pi); dividing by the
    # bin width turns "counts per bin" into a number of events
    signal_count = norm * sigma * np.sqrt(2 * np.pi) / bin_width
    background_count = (a + b * mu) * (6 * sigma) / bin_width   # plus/minus 3 sigma

    return {
        "mu": mu,
        "mu_err": perr[0],
        "sigma": sigma,
        "sigma_err": perr[1],
        "signal_count": signal_count,
        "background_count": background_count,
        "popt": popt,
        "pcov": pcov,
    }
