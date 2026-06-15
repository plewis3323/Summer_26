"""Figures for Bishop 1.2 (Bayes update) and 1.3 (S-fold cross-validation)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- Fig 1: Bayes fruit-box prior -> posterior ----
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Red box", "Blue box"]
prior = [0.4, 0.6]
post = [2/3, 1/3]
xpos = np.arange(2)
w = 0.35
ax.bar(xpos - w/2, prior, w, label="prior  p(B)", color="#bbbbbb")
ax.bar(xpos + w/2, post, w, label="posterior  p(B | orange)",
       color=["#d62728", "#1f77b4"])
for i, (a, b) in enumerate(zip(prior, post)):
    ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.8); ax.set_ylabel("probability")
ax.set_title("Bayesian update after observing an orange")
ax.legend()
fig.tight_layout()
fig.savefig("img_bayes_fruit.png", dpi=130)
print("saved img_bayes_fruit.png")

# ---- Fig 2: S-fold cross-validation diagram ----
S = 4
fig, ax = plt.subplots(figsize=(7, 3.2))
for run in range(S):
    for fold in range(S):
        is_val = (fold == run)
        color = "#ff7f0e" if is_val else "#9ecae1"
        ax.add_patch(Rectangle((fold, S - 1 - run), 1, 1,
                               facecolor=color, edgecolor="white"))
    ax.text(-0.15, S - 0.5 - run, f"run {run+1}", ha="right", va="center", fontsize=9)
ax.set_xlim(-1.2, S); ax.set_ylim(0, S)
ax.set_xticks(np.arange(S) + 0.5)
ax.set_xticklabels([f"fold {i+1}" for i in range(S)])
ax.set_yticks([])
ax.set_title("S-fold cross-validation (S=4): orange = validation, blue = training")
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig("img_cross_validation.png", dpi=130)
print("saved img_cross_validation.png")
