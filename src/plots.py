"""Grafik hasil pelatihan & evaluasi (dibuat dari file di outputs/results)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve)

from config import FIG_DIR, RESULT_DIR

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
LABEL = {"S1_baseline_bce": "S1 Baseline BCE", "S2_weighted_bce": "S2 Weighted BCE",
         "S3_focal": "S3 Focal Loss"}
COL = {"S1_baseline_bce": "#7F8C8D", "S2_weighted_bce": "#E67E22", "S3_focal": "#2E86AB"}


def load(name):
    return json.load(open(RESULT_DIR / name))


def curves():
    for tag in LABEL:
        h = load(f"history_{tag}.json")
        ep = np.arange(1, len(h["loss"]) + 1)
        best = int(np.argmax(h["val_pr_auc"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
        for ax, key, ttl in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy"),
                             (axes[2], "pr_auc", "PR-AUC")):
            ax.plot(ep, h[key], label="Training", color="#2E86AB")
            ax.plot(ep, h[f"val_{key}"], label="Validation", color="#D7263D")
            ax.axvline(best, ls="--", color="gray", lw=.8)
            ax.set_title(f"{ttl} — {LABEL[tag]}"); ax.set_xlabel("Epoch")
        axes[0].legend()
        axes[2].text(best, axes[2].get_ylim()[0], f" best ep={best}", fontsize=7,
                     color="gray", va="bottom")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"res_curves_{tag}.png"); plt.close(fig)


FIN = "S1_baseline_bce"   # diisi ulang di main() dari final_selection.json
SHORT = {"S1_baseline_bce": "S1 BCE", "S2_weighted_bce": "S2 W-BCE", "S3_focal": "S3 Focal"}


def confusion_matrices(final_thr):
    y = np.load(RESULT_DIR / "y_test.npy")
    p1 = np.load(RESULT_DIR / "prob_test_S1_baseline_bce.npy")
    p3 = np.load(RESULT_DIR / "prob_test_S3_focal.npy")
    pf = np.load(RESULT_DIR / f"prob_test_{FIN}.npy")
    items = [("S1 BCE, thr default 0.50", p1, 0.5),
             ("S3 Focal Loss, thr default 0.50", p3, 0.5),
             (f"FINAL: {SHORT[FIN]} + thr optimal ({final_thr:.3f})", pf, final_thr)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (ttl, p, t) in zip(axes, items):
        cm = confusion_matrix(y, (p >= t).astype(int))
        ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1, vmax=cm.max()))
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > 1000 else "black", fontsize=11, weight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Fraud"]); ax.set_yticklabels(["Normal", "Fraud"])
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(ttl, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "res_confusion_matrix.png"); plt.close(fig)


def pr_roc():
    y = np.load(RESULT_DIR / "y_test.npy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for tag, lab in LABEL.items():
        p = np.load(RESULT_DIR / f"prob_test_{tag}.npy")
        pr, rc, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(rc, pr, label=lab, color=COL[tag])
        axes[1].plot(fpr, tpr, label=lab, color=COL[tag])
    axes[0].axhline(y.mean(), ls=":", color="k", lw=.8, label=f"Acak ({y.mean():.4f})")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision–Recall curve (data testing)"); axes[0].legend(fontsize=7)
    axes[1].plot([0, 1], [0, 1], ls=":", color="k", lw=.8)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC curve (data testing)"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "res_pr_roc.png"); plt.close(fig)


def threshold_plot(final_thr):
    yv = np.load(RESULT_DIR / "y_val.npy")
    pv = np.load(RESULT_DIR / f"prob_val_{FIN}.npy")
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.plot(th, pr[:-1], label="Precision", color="#2E86AB")
    ax.plot(th, rc[:-1], label="Recall", color="#D7263D")
    ax.plot(th, f1, label="F1-score", color="#27AE60", lw=2)
    ax.axvline(0.5, ls=":", color="gray", label="thr default 0.50")
    ax.axvline(final_thr, ls="--", color="k", label=f"thr optimal {final_thr:.3f}")
    ax.set_xlabel("Threshold probabilitas"); ax.set_xlim(0, 1)
    ax.set_title(f"Precision/Recall/F1 vs threshold — model final {SHORT[FIN]} (data validation)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "res_threshold.png"); plt.close(fig)


def score_dist(final_thr):
    y = np.load(RESULT_DIR / "y_test.npy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, tag in zip(axes, ("S1_baseline_bce", "S3_focal")):
        p = np.load(RESULT_DIR / f"prob_test_{tag}.npy")
        bins = np.linspace(0, 1, 41)
        ax.hist(p[y == 0], bins=bins, color="#2E86AB", alpha=.6, label="Normal")
        ax.hist(p[y == 1], bins=bins, color="#D7263D", alpha=.8, label="Fraud")
        ax.set_yscale("log"); ax.set_title(f"Distribusi probabilitas — {LABEL[tag]}")
        ax.set_xlabel("P(fraud)"); ax.legend(fontsize=7)
        if tag == FIN:
            ax.axvline(final_thr, ls="--", color="k", lw=.8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "res_score_dist.png"); plt.close(fig)


def comparison_bar(rows):
    names = [r["label"] for r in rows]
    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for k, (m, c) in enumerate((("precision", "#2E86AB"), ("recall", "#D7263D"),
                                ("f1", "#27AE60"), ("pr_auc", "#8E44AD"))):
        vals = [r[m] for r in rows]
        bars = ax.bar(x + (k - 1.5) * w, vals, w, label=m.upper().replace("_", "-"), color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.2f}", ha="center",
                    fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7); ax.set_ylim(0, 1.15)
    ax.legend(ncol=4, fontsize=7, loc="upper left")
    ax.set_title("Perbandingan skenario pada data testing (seed 42)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "res_comparison.png"); plt.close(fig)


def architecture():
    layers_ = [("Input", "31 fitur\n(V1–V28, LogAmount,\nHour_sin, Hour_cos)", "#D5E8F0"),
               ("Dense 64", "L2=1e-4\nBatchNorm → ReLU\nDropout 0.3", "#AED6F1"),
               ("Dense 32", "L2=1e-4\nBatchNorm → ReLU\nDropout 0.3", "#AED6F1"),
               ("Dense 16", "L2=1e-4\nBatchNorm → ReLU\nDropout 0.2", "#AED6F1"),
               ("Dense 1", "Sigmoid\nP(fraud)", "#F5B7B1"),
               ("Threshold", "P ≥ t* → Fraud\n(t* dari validation)", "#ABEBC6")]
    shapes = ["(None, 31)", "(None, 64)", "(None, 32)", "(None, 16)", "(None, 1)", "label 0/1"]
    fig, ax = plt.subplots(figsize=(11, 2.9)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 3)
    for i, ((t, d, c), s) in enumerate(zip(layers_, shapes)):
        x0 = 0.1 + i * 2
        ax.add_patch(FancyBboxPatch((x0, .55), 1.7, 1.9, boxstyle="round,pad=0.03",
                                    fc=c, ec="#34495E", lw=1))
        ax.text(x0 + .85, 2.15, t, ha="center", weight="bold", fontsize=9)
        ax.text(x0 + .85, 1.35, d, ha="center", va="center", fontsize=7)
        ax.text(x0 + .85, .3, s, ha="center", fontsize=7, color="#555", family="monospace")
        if i < len(layers_) - 1:
            ax.annotate("", xy=(x0 + 2.0, 1.5), xytext=(x0 + 1.75, 1.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(6, 2.85, "Loss: Binary Cross-Entropy  |  Optimizer: Adam (lr=1e-3)  |  Ekspor: TFLite int8",
            ha="center", fontsize=8, style="italic")
    fig.tight_layout(); fig.savefig(FIG_DIR / "model_architecture.png"); plt.close(fig)


def main():
    res = load("all_results.json")
    s42 = {r["scenario"]: r for r in res if r["seed"] == 42}
    thr = json.load(open(RESULT_DIR / "final_selection.json"))["threshold"]
    global FIN
    FIN = json.load(open(RESULT_DIR / "final_selection.json"))["final_scenario"]
    assert abs(thr - s42[FIN]["thr_opt"]) < 1e-12
    curves(); confusion_matrices(thr); pr_roc(); threshold_plot(thr); score_dist(thr)
    architecture()
    rows = []
    for tag, lab in (("S1_baseline_bce", "S1 BCE\nthr 0.5"),
                     ("S2_weighted_bce", "S2 W-BCE\nthr 0.5"),
                     ("S3_focal", "S3 Focal\nthr 0.5")):
        rows.append({"label": lab, **s42[tag]["test@0.5"]})
    for tag in ("S1_baseline_bce", "S2_weighted_bce", "S3_focal"):
        lab = SHORT[tag] + "\nthr opt" + (" (FINAL)" if tag == FIN else "")
        rows.append({"label": lab, **s42[tag]["test@opt"]})
    comparison_bar(rows)


if __name__ == "__main__":
    main()
