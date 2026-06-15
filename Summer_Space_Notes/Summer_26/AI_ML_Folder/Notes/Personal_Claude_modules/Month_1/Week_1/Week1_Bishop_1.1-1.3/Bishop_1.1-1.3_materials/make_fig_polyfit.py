"""Reproduce Bishop Fig. 1.4-style polynomial curve fitting / over-fitting panel."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N = 10
x = np.linspace(0, 1, N)
t = np.sin(2 * np.pi * x) + rng.normal(0, 0.18, N)

xs = np.linspace(0, 1, 400)
truth = np.sin(2 * np.pi * xs)

def polyfit_predict(x, t, M, xs):
    X = np.vander(x, M + 1, increasing=True)
    w, *_ = np.linalg.lstsq(X, t, rcond=None)
    Xs = np.vander(xs, M + 1, increasing=True)
    return Xs @ w

orders = [0, 1, 3, 9]
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, M in zip(axes.ravel(), orders):
    yhat = polyfit_predict(x, t, M, xs)
    ax.plot(xs, truth, "g-", lw=1.5, label=r"$\sin(2\pi x)$")
    ax.plot(xs, yhat, "r-", lw=1.8, label=f"fit M={M}")
    ax.scatter(x, t, facecolors="none", edgecolors="b", s=55, label="data")
    ax.set_title(f"M = {M}")
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.legend(fontsize=7, loc="upper right")
fig.suptitle("Polynomial curve fitting: under-fit (M=0,1), good (M=3), over-fit (M=9)")
fig.tight_layout()
fig.savefig("img_polyfit_overfitting.png", dpi=130)
print("saved img_polyfit_overfitting.png")
