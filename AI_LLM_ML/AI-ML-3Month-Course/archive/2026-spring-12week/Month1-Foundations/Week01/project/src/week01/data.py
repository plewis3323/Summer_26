"""Data loading for Week 01.

This file is plumbing, not one of the exercises. Downloads are cached in
project/data/ so the network only gets hit once.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

# this file lives at project/src/week01/data.py, so the project folder is
# two levels up from the folder this file is in
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))


def data_dir():
    """Path to project/data, created if it is not there yet."""
    folder = os.path.join(PROJECT_DIR, "data")
    os.makedirs(folder, exist_ok=True)
    return folder


def results_dir():
    """Path to project/results, created if it is not there yet."""
    folder = os.path.join(PROJECT_DIR, "results")
    os.makedirs(folder, exist_ok=True)
    return folder


def download_file(urls, dest):
    """Download the first url in the list that works, save it to dest.

    Returns dest. If the file is already there it does nothing, so re-running
    a cell costs nothing.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    failures = []
    for url in urls:
        print("downloading " + url)
        try:
            # write to a .part file first, so a half-finished download never
            # gets mistaken for a cached one
            urllib.request.urlretrieve(url, dest + ".part")
        except Exception as err:  # noqa: BLE001 - any failure means "try the next mirror"
            failures.append("  " + url + "\n    " + str(err))
            continue
        os.rename(dest + ".part", dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  cached {dest} ({size_mb:.1f} MB)")
        return dest

    message = "could not download the file. Tried:\n"
    for line in failures:
        message = message + line + "\n"
    raise RuntimeError(message)


# --- CMS dimuon data, used by E3 and E4 -------------------------------------
# The link printed in exercises.md (cms-opendata-workshop/workshop2023-...) is
# dead. These two mirrors carry the same 21-column file.
DIMUON_URLS = [
    ("https://raw.githubusercontent.com/cms-opendata-education/"
     "cms-jupyter-materials-english/master/Data/DoubleMuRun2011A.csv"),
    "https://opendata.cern.ch/record/545/files/Dimuon_DoubleMu.csv",
]

DIMUON_COLUMNS = [
    "Run", "Event",
    "Type1", "E1", "px1", "py1", "pz1", "pt1", "eta1", "phi1", "Q1",
    "Type2", "E2", "px2", "py2", "pz2", "pt2", "eta2", "phi2", "Q2",
    "M",
]


def dimuon_path():
    """Path to the cached dimuon CSV (about 72 MB, downloaded once)."""
    return download_file(DIMUON_URLS, os.path.join(data_dir(), "DoubleMuRun2011A.csv"))


def load_dimuon(nrows=None):
    """CMS DoubleMuParked 2011 dimuon events as a DataFrame.

    Columns are Run, Event, then the same nine per muon (Type, E, px, py, pz,
    pt, eta, phi, Q), then M, which is the invariant mass CMS computed. Do not
    take M on trust: E3 asks you to compute your own and compare.
    """
    df = pd.read_csv(dimuon_path(), nrows=nrows)

    # some mirrors ship the header with stray spaces
    clean_names = []
    for name in df.columns:
        clean_names.append(name.strip())
    df.columns = clean_names

    # and the two mirrors disagree about capitalising the type columns
    df = df.rename(columns={"type1": "Type1", "type2": "Type2"})
    return df


# --- a small ROOT file, used by E6 ------------------------------------------
ZMUMU_ROOT_URLS = [
    ("https://raw.githubusercontent.com/scikit-hep/scikit-hep-testdata/"
     "main/src/skhep_testdata/data/uproot-Zmumu.root"),
]


def zmumu_root_path():
    """Path to a small (about 180 kB) real ROOT file holding dimuon events."""
    return download_file(ZMUMU_ROOT_URLS, os.path.join(data_dir(), "uproot-Zmumu.root"))


# --- fake photon pairs, used by E5, E7, E9 and E10 --------------------------
def make_pi0_toy(n=50000, signal_frac=0.20, mu=0.135, sigma=0.008,
                 lo=0.05, hi=0.25, seed=None):
    """Photon-pair masses: a gaussian pi0 peak sitting on a flat background.

    Provided so E5 and E10 start from the same generator. Pass a seed to get
    the same numbers every time.
    """
    rng = np.random.default_rng(seed)

    n_signal = rng.binomial(n, signal_frac)
    signal = rng.normal(mu, sigma, n_signal)
    background = rng.uniform(lo, hi, n - n_signal)
    masses = np.concatenate([signal, background])

    # a handful of signal events land outside (lo, hi); redraw those so the
    # array we hand back always has exactly n entries
    outside = (masses < lo) | (masses > hi)
    while outside.any():
        masses[outside] = rng.normal(mu, sigma, int(outside.sum()))
        outside = (masses < lo) | (masses > hi)

    rng.shuffle(masses)
    return masses
