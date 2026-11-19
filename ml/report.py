"""Plots for the README / the in-app 'Optimisation report' page."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN, LEAF, AMBER, GREY = "#2f7d4f", "#7cb342", "#e0a030", "#8a948c"


def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def make_plots(out_dir, woa_hist, rs_hist, finals, cm, classes, trace):
    _style()
    out = Path(out_dir)

    # convergence --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    it = [h["iter"] for h in woa_hist]
    ax.plot(it, [h["best"] for h in woa_hist], color=GREEN, lw=2.2, marker="o", ms=4, label="WOA best")
    ax.plot(it, [h["mean"] for h in woa_hist], color=LEAF, lw=1.4, ls="--", label="WOA population mean")
    if rs_hist:
        n_per_iter = max(len(rs_hist) // max(len(woa_hist), 1), 1)
        xs = np.linspace(0, len(woa_hist) - 1, len(rs_hist))
        ax.plot(xs, rs_hist, color=AMBER, lw=1.6, label="Random search best (same budget)")
    ax.axhline(finals["baseline"]["val_fitness"], color=GREY, ls=":", label="Hand-tuned baseline")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Fitness (val accuracy - cost penalty)")
    ax.set_title("Whale Optimization convergence"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "woa_convergence.png"); plt.close(fig)

    # method comparison -----------------------------------------------------------------
    names = ["baseline", "random_search", "woa"]
    labels = ["Hand-tuned\nbaseline", "Random\nsearch", "Whale Opt.\n(WOA)"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, key, title, fmt in ((axs[0], "test_acc", "Test accuracy", "{:.1%}"),
                                (axs[1], "latency_ms", "Latency per image (ms)", "{:.1f}"),
                                (axs[2], "size_kb", "Model size (KB)", "{:.0f}")):
        vals = [finals[n][key] for n in names]
        bars = ax.bar(labels, vals, color=[GREY, AMBER, GREEN])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(title); ax.grid(axis="x", visible=False)
        if key == "test_acc":
            ax.set_ylim(max(0, min(vals) - 0.1), min(1.0, max(vals) + 0.05))
    fig.tight_layout(); fig.savefig(out / "method_comparison.png"); plt.close(fig)

    # accuracy vs latency scatter of every evaluated config ---------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = {"svm": GREEN, "mlp": AMBER, "rf": "#5b7db1"}
    for c in ("svm", "mlp", "rf"):
        pts = [(r["latency_ms"], r["val_acc"]) for r in trace if r["params"]["classifier"] == c]
        if pts:
            ax.scatter(*zip(*pts), s=22, alpha=0.7, color=colors[c], label=c.upper())
    ax.set_xlabel("Latency per image (ms, log)"); ax.set_ylabel("Validation accuracy"); ax.set_xscale("log")
    ax.set_title("Every configuration explored: accuracy vs. compute"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "accuracy_vs_latency.png"); plt.close(fig)

    # confusion matrix ----------------------------------------------------------------------
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    short = [c.replace("___", " - ").replace("_", " ")[:28] for c in classes]
    ax.set_xticklabels(short, rotation=90, fontsize=6); ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
    ax.set_title("Normalised confusion matrix (test split, WOA model)")
    fig.colorbar(im, fraction=0.04); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png"); plt.close(fig)
