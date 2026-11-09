"""Membuat data/contoh_transaksi_uji.csv untuk demo aplikasi.

20 transaksi diambil dari data TESTING (split identik dengan preprocessing.split):
6 fraud tertangkap (TP), 3 fraud lolos (FN), seluruh false positive (FP), dan sisanya
transaksi normal yang benar (TN), sehingga demo memperlihatkan prediksi benar dan salah.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import MODEL_DIR, RESULT_DIR, ROOT, SEED, TEST_SIZE
from preprocessing import clean, load_raw


def main():
    df, _ = clean(load_raw())
    y = df["Class"].values
    _, i_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y,
                               random_state=SEED)
    sel = json.load(open(RESULT_DIR / "final_selection.json"))
    p = np.load(RESULT_DIR / f"prob_test_{sel['final_scenario']}.npy")
    thr = json.load(open(MODEL_DIR / "threshold.json"))["threshold"]
    te = df.iloc[i_te].copy(); te["p"] = p
    tp = te[(te.Class == 1) & (te.p >= thr)].sample(6, random_state=1)
    fn = te[(te.Class == 1) & (te.p < thr)].sample(3, random_state=1)
    fp = te[(te.Class == 0) & (te.p >= thr)]
    tn = te[(te.Class == 0) & (te.p < thr)].sample(20 - 6 - 3 - len(fp), random_state=1)
    out = pd.concat([tp, fn, fp, tn]).sample(frac=1, random_state=3).drop(columns="p")
    out.to_csv(ROOT / "data" / "contoh_transaksi_uji.csv", index=False)
    print(len(out), "baris disimpan")


if __name__ == "__main__":
    main()
