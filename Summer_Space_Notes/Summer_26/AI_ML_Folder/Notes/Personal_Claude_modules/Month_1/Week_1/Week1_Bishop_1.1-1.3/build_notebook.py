"""Assemble Bishop 1.1-1.3 notes + figures into a single readable, runnable notebook.

Figures are embedded as base64 attachments so the .ipynb is fully self-contained
(it renders even if the materials folder is moved or absent).
"""
import json, os, base64

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = "Bishop_1.1-1.3_materials"  # source location of the note/image files


_counter = [0]
def _next_id():
    _counter[0] += 1
    return f"cell-{_counter[0]:02d}"


def md(text):
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": _next_id(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def read_note(name):
    with open(os.path.join(HERE, MAT, name), encoding="utf-8") as f:
        return f.read()


def md_image(filename, caption):
    """Markdown cell with the image embedded as a base64 attachment (self-contained)."""
    with open(os.path.join(HERE, MAT, filename), "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    cell = md(f"![{caption}](attachment:{filename})\n\n*{caption}*\n")
    cell["attachments"] = {filename: {"image/png": b64}}
    return cell


cells = []

# --- Title / overview ---
cells.append(md(
"""# Bishop, *Pattern Recognition and Machine Learning* — §1.1–1.3

**Reading session notes — Week 1, Month 1.**

This notebook compiles a reading session over Bishop's §1.1 (Polynomial Curve Fitting),
§1.2 (Probability Theory), and §1.3 (Model Selection). It contains:

- **Grad-student summaries** of each section, with key derivations and worked exercises.
- **Figures** reproduced programmatically (shown statically, and regenerable via the code cells).
- A **Q&A log** of questions worked through during the session.

*How to use:* read top-to-bottom as notes; run the code cells to regenerate the figures.
Requires `numpy` and `matplotlib`.
"""))

# --- 1.1 ---
cells.append(md(read_note("notes_01_polynomial_curve_fitting.md")))
cells.append(md_image("img_polyfit_overfitting.png",
                      "Polynomial curve fitting: under-fit (M=0,1), good (M=3), over-fit (M=9). "
                      "Run the next cell to regenerate."))
cells.append(code(
"""# Figure: polynomial curve fitting / over-fitting (Bishop Fig. 1.4-style)
import numpy as np
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
    return np.vander(xs, M + 1, increasing=True) @ w

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, M in zip(axes.ravel(), [0, 1, 3, 9]):
    ax.plot(xs, truth, "g-", lw=1.5, label=r"$\\sin(2\\pi x)$")
    ax.plot(xs, polyfit_predict(x, t, M, xs), "r-", lw=1.8, label=f"fit M={M}")
    ax.scatter(x, t, facecolors="none", edgecolors="b", s=55, label="data")
    ax.set_title(f"M = {M}"); ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.legend(fontsize=7, loc="upper right")
fig.suptitle("Under-fit (M=0,1), good (M=3), over-fit (M=9)")
fig.tight_layout(); plt.show()
"""))

# --- 1.2 ---
cells.append(md(read_note("notes_02_probability_theory.md")))
cells.append(md_image("img_bayes_fruit.png",
                      "Prior to posterior for the fruit-box example. Run the next cell to regenerate."))
cells.append(code(
"""# Figure: Bayesian update for the fruit-box example
import numpy as np
import matplotlib.pyplot as plt

labels = ["Red box", "Blue box"]
prior = [0.4, 0.6]
post  = [2/3, 1/3]
xpos = np.arange(2); w = 0.35
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(xpos - w/2, prior, w, label="prior  p(B)", color="#bbbbbb")
ax.bar(xpos + w/2, post,  w, label="posterior  p(B | orange)", color=["#d62728", "#1f77b4"])
for i, (a, b) in enumerate(zip(prior, post)):
    ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.8); ax.set_ylabel("probability")
ax.set_title("Bayesian update after observing an orange"); ax.legend()
fig.tight_layout(); plt.show()
"""))

# --- 1.3 ---
cells.append(md(read_note("notes_03_model_selection.md")))
cells.append(md_image("img_cross_validation.png",
                      "S-fold cross-validation rotation (S=4). Run the next cell to regenerate."))
cells.append(code(
"""# Figure: S-fold cross-validation diagram
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

S = 4
fig, ax = plt.subplots(figsize=(7, 3.2))
for run in range(S):
    for fold in range(S):
        color = "#ff7f0e" if fold == run else "#9ecae1"
        ax.add_patch(Rectangle((fold, S - 1 - run), 1, 1, facecolor=color, edgecolor="white"))
    ax.text(-0.15, S - 0.5 - run, f"run {run+1}", ha="right", va="center", fontsize=9)
ax.set_xlim(-1.2, S); ax.set_ylim(0, S)
ax.set_xticks(np.arange(S) + 0.5); ax.set_xticklabels([f"fold {i+1}" for i in range(S)])
ax.set_yticks([])
ax.set_title("S-fold cross-validation (S=4): orange = validation, blue = training")
for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(); plt.show()
"""))

# --- Q&A log ---
cells.append(md(read_note("notes_04_qa_log.md")))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = os.path.join(HERE, "Bishop_1.1-1.3_Notes.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")

# Execute in place so the code-cell figures are rendered/embedded in the saved notebook.
rc = os.system(f'jupyter nbconvert --to notebook --execute --inplace "{out}"')
print("executed in place" if rc == 0 else f"(execution skipped/failed, rc={rc}) — "
      "static attachment images still render; run the code cells manually if desired")
