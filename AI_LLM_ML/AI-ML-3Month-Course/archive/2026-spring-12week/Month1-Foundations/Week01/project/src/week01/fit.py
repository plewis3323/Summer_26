"""Fitting a gaussian peak on a straight-line background.

E7 step 1 ("drop the worked example into src/week01/fit.py") is already done
for you -- this is README.md section 8. The drill is writing the tests, and one
of them (test_empty_input_raises) will fail against the code below. That is
deliberate: let the test tell you what to add.
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
    """Fit the peak and return the results in a dictionary.

    Keys: mu, mu_err, sigma, sigma_err, signal_count, background_count, popt,
    pcov. E9 needs popt to rebuild the fitted curve, so it is kept here too.
    """
    # TODO (E7.3): this happily tries to fit an empty array. Make it raise
    # ValueError on zero-length input instead -- but write the test first.
    counts, _edges = np.histogram(m_gg, bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # starting guesses: peak near 0.135 GeV, about 10 MeV wide, flat background
    p0 = [0.135, 0.010, counts.max(), 1.0, 0.0]
    # sqrt(counts + 1) is the Poisson error on each bin, with the +1 so an
    # empty bin gets weight 1 instead of infinite weight
    bin_errors = np.sqrt(counts + 1)
    popt, pcov = curve_fit(signal_plus_bg, centers, counts, p0=p0, sigma=bin_errors)
    perr = np.sqrt(np.diag(pcov))

    mu = popt[0]
    sigma = popt[1]
    norm = popt[2]
    a = popt[3]
    b = popt[4]

    bin_width = bin_edges[1] - bin_edges[0]
    # the area under a gaussian is norm * sigma * sqrt(2 pi); dividing by the
    # bin width turns "counts per bin" into a number of events
    signal_count = norm * sigma * np.sqrt(2 * np.pi) / bin_width
    background_count = (a + b * mu) * (6 * sigma)   # a plus/minus 3 sigma window

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
