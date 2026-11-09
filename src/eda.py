"""Exploratory Data Analysis (EDA) dataset Credit Card Fraud (ULB).
Menghasilkan grafik di outputs/figures dan ringkasan di outputs/results/eda_summary.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIG_DIR, RESULT_DIR
from preprocessing import clean, load_raw

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
C_NORMAL, C_FRAUD = "#2E86AB", "#D7263D"


def main():
    raw = load_raw()
    df, clean_info = clean(raw)
    summary = {"shape_raw": list(raw.shape), **clean_info}

    # ---------- 1. Distribusi kelas ----------
    vc = raw["Class"].value_counts().sort_index()
    summary["class_counts_raw"] = {int(k): int(v) for k, v in vc.items()}
    summary["fraud_ratio_raw_pct"] = round(100 * vc[1] / vc.sum(), 4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    bars = axes[0].bar(["Normal (0)", "Fraud (1)"], vc.values, color=[C_NORMAL, C_FRAUD])
    for b, v in zip(bars, vc.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    axes[0].set_title("Jumlah transaksi per kelas (skala linear)")
    axes[1].bar(["Normal (0)", "Fraud (1)"], vc.values, color=[C_NORMAL, C_FRAUD])
    axes[1].set_yscale("log")
    axes[1].set_title(f"Skala log — fraud hanya {summary['fraud_ratio_raw_pct']:.3f}%")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_01_class_distribution.png"); plt.close(fig)

    # ---------- 2. Statistik deskriptif ----------
    desc = raw.describe().T[["mean", "std", "min", "max"]]
    desc.round(4).to_csv(RESULT_DIR / "eda_describe.csv")
    summary["amount_stats"] = {
        k: round(float(v), 2) for k, v in raw["Amount"].describe().items()}
    summary["amount_by_class"] = {
        int(c): {k: round(float(v), 2) for k, v in g["Amount"].describe().items()}
        for c, g in raw.groupby("Class")}
    summary["time_max_hours"] = round(raw["Time"].max() / 3600, 2)
    summary["amount_zero"] = int((raw["Amount"] == 0).sum())

    # ---------- 3. Histogram Amount & Time ----------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].hist(raw["Amount"], bins=100, color=C_NORMAL)
    axes[0].set_yscale("log"); axes[0].set_title("Amount (asli) — sangat skewed")
    axes[0].set_xlabel("Amount (EUR)")
    for c, col, lab in ((0, C_NORMAL, "Normal"), (1, C_FRAUD, "Fraud")):
        axes[1].hist(np.log1p(raw.loc[raw.Class == c, "Amount"]), bins=60, alpha=.6,
                     density=True, color=col, label=lab)
        hour = (raw.loc[raw.Class == c, "Time"] % 86400) / 3600
        axes[2].hist(hour, bins=24, alpha=.6, density=True, color=col, label=lab)
    axes[1].set_title("log(1+Amount) per kelas"); axes[1].legend()
    axes[1].set_xlabel("log(1+Amount)")
    axes[2].set_title("Jam transaksi per kelas"); axes[2].set_xlabel("Jam ke-")
    axes[2].legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_02_amount_time.png"); plt.close(fig)

    # Proporsi fraud per jam
    hour_all = ((raw["Time"] % 86400) // 3600).astype(int)
    fraud_rate_hour = raw.groupby(hour_all)["Class"].mean() * 100
    summary["fraud_rate_by_hour_pct"] = {int(k): round(float(v), 3)
                                         for k, v in fraud_rate_hour.items()}

    # ---------- 4. Histogram fitur V1..V28 ----------
    fig, axes = plt.subplots(7, 4, figsize=(11, 15))
    for i, ax in enumerate(axes.ravel()):
        col = f"V{i + 1}"
        lo, hi = raw[col].quantile([0.001, 0.999])
        bins = np.linspace(lo, hi, 50)
        ax.hist(raw.loc[raw.Class == 0, col], bins=bins, density=True, alpha=.55,
                color=C_NORMAL, label="Normal")
        ax.hist(raw.loc[raw.Class == 1, col], bins=bins, density=True, alpha=.55,
                color=C_FRAUD, label="Fraud")
        ax.set_title(col, fontsize=9); ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Histogram (density) fitur PCA V1–V28 per kelas", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_03_hist_V.png"); plt.close(fig)

    # ---------- 5. Korelasi ----------
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Matriks korelasi Pearson antarfitur")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_04_corr_heatmap.png"); plt.close(fig)

    off_arr = corr.drop(index="Class", columns="Class").to_numpy(copy=True)
    np.fill_diagonal(off_arr, 0)
    names = [c for c in corr.columns if c != "Class"]
    off = pd.DataFrame(off_arr, index=names, columns=names)
    pairs = off.abs().stack().sort_values(ascending=False)
    pairs = pairs[pairs.index.map(lambda p: p[0] < p[1])]
    summary["top_feature_pairs_abs_corr"] = [
        [a, b, round(float(off.loc[a, b]), 3)] for (a, b) in pairs.index[:6]]
    vv = [f"V{i}" for i in range(1, 29)]
    v_off = corr.loc[vv, vv].values.copy(); np.fill_diagonal(v_off, 0)
    summary["max_abs_corr_among_V"] = round(float(np.abs(v_off).max()), 4)

    cc = corr["Class"].drop("Class").sort_values()
    summary["corr_with_class_top_neg"] = {k: round(float(v), 3) for k, v in cc.head(5).items()}
    summary["corr_with_class_top_pos"] = {k: round(float(v), 3) for k, v in cc.tail(5).items()}
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(cc.index, cc.values, color=[C_FRAUD if v > 0 else C_NORMAL for v in cc.values])
    ax.set_xticks(range(len(cc))); ax.set_xticklabels(cc.index, rotation=90)
    ax.set_title("Korelasi tiap fitur terhadap label Class"); ax.axhline(0, color="k", lw=.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_05_corr_class.png"); plt.close(fig)

    # ---------- 6. Boxplot fitur paling diskriminatif ----------
    top = list(cc.abs().sort_values(ascending=False).index[:6])
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    for ax, col in zip(axes, top):
        ax.boxplot([df.loc[df.Class == 0, col], df.loc[df.Class == 1, col]],
                   tick_labels=["Normal", "Fraud"], showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#dfe9f3"))
        ax.set_title(col)
    fig.suptitle("Boxplot 6 fitur dengan |korelasi| terbesar terhadap Class (tanpa outlier)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_06_boxplot_top.png"); plt.close(fig)

    # ---------- 7. Outlier (IQR) ----------
    q1, q3 = df[vv].quantile(.25), df[vv].quantile(.75)
    iqr = q3 - q1
    out_mask = ((df[vv] < q1 - 1.5 * iqr) | (df[vv] > q3 + 1.5 * iqr)).any(axis=1)
    summary["rows_with_iqr_outlier_pct"] = round(100 * out_mask.mean(), 2)
    summary["fraud_in_outlier_rows_pct"] = round(
        100 * df.loc[out_mask, "Class"].sum() / df["Class"].sum(), 2)

    # ---------- 8. Contoh sampel per kelas ----------
    cols = ["Time", "V1", "V2", "V3", "V4", "V14", "V17", "Amount", "Class"]
    samp = pd.concat([raw[raw.Class == 0].sample(3, random_state=1),
                      raw[raw.Class == 1].sample(3, random_state=1)])[cols]
    samp.round(3).to_csv(RESULT_DIR / "eda_samples.csv")

    with open(RESULT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
