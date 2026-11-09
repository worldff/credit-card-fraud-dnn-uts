"""Analisis kesalahan model final pada data testing (TP, FN, FP)."""
import json

import numpy as np
from sklearn.model_selection import train_test_split

from config import MODEL_DIR, RESULT_DIR, SEED, TEST_SIZE
from preprocessing import clean, load_raw


def main():
    df, _ = clean(load_raw())
    y = df["Class"].values
    # reproduksi indeks data testing (split identik dengan preprocessing.split)
    _, i_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y,
                               random_state=SEED)
    assert (y[i_te] == np.load(RESULT_DIR / "y_test.npy")).all()
    sel = json.load(open(RESULT_DIR / "final_selection.json"))
    p = np.load(RESULT_DIR / f"prob_test_{sel['final_scenario']}.npy")
    thr = json.load(open(MODEL_DIR / "threshold.json"))["threshold"]
    te = df.iloc[i_te].copy(); te["p"] = p
    groups = {"TP": te[(te.Class == 1) & (te.p >= thr)],
              "FN": te[(te.Class == 1) & (te.p < thr)],
              "FP": te[(te.Class == 0) & (te.p >= thr)]}
    out = {}
    for n, g in groups.items():
        out[n] = {"n": len(g), "amount_median": float(g.Amount.median()),
                  "V14_median": float(g.V14.median()), "V17_median": float(g.V17.median()),
                  "V12_median": float(g.V12.median()), "p_median": float(g.p.median()),
                  "p_max": float(g.p.max())}
    out["normal_V14_median"] = float(te[te.Class == 0].V14.median())
    out["normal_V17_median"] = float(te[te.Class == 0].V17.median())
    fn = groups["FN"]
    out["FN_p_below_0.01"] = int((fn.p < 0.01).sum())
    out["FN_amounts"] = sorted(round(float(a), 2) for a in fn.Amount)
    out["FP_p"] = sorted(round(float(a), 3) for a in groups["FP"].p)
    with open(RESULT_DIR / "error_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
