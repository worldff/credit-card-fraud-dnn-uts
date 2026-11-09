"""Penyempurnaan UTS #2 — penjelasan prediksi per transaksi (occlusion sensitivity).

Untuk satu transaksi, setiap fitur (atau kelompok fitur) secara bergantian diganti dengan
nilai rata-rata data latih (= 0 setelah standardisasi), lalu probabilitas fraud dihitung
ulang. Selisihnya adalah kontribusi fitur tersebut:

    kontribusi(f) = P(fraud | x) - P(fraud | x dengan f diganti rata-rata)

Nilai positif berarti fitur itu mendorong prediksi ke arah FRAUD. Metode ini tidak
membutuhkan gradien sehingga dapat dipakai langsung pada model TFLite di aplikasi
(Zeiler & Fergus, 2014 menggunakan prinsip occlusion yang sama untuk CNN).

Jalankan `python explain.py` untuk menganalisis fraud pada data testing.
"""
import json

import numpy as np
import pandas as pd

from config import FIG_DIR, MODEL_DIR, RESULT_DIR

MODEL_FEATURES = [f"V{i}" for i in range(1, 29)] + ["LogAmount", "Hour_sin", "Hour_cos"]
GROUPS = {**{f"V{i}": [i - 1] for i in range(1, 29)},
          "Amount": [28], "Time (jam)": [29, 30]}


def occlusion(predict_fn, Xs: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Xs: fitur terstandardisasi (n, 31). predict_fn: array -> probabilitas (n,)."""
    Xs = np.asarray(Xs, dtype="float32")
    n = len(Xs)
    batch = [Xs]
    for idx in GROUPS.values():
        Xo = Xs.copy(); Xo[:, idx] = 0.0
        batch.append(Xo)
    p = predict_fn(np.concatenate(batch)).reshape(len(GROUPS) + 1, n)
    contrib = pd.DataFrame((p[0] - p[1:]).T, columns=list(GROUPS))
    return p[0], contrib


def top_features(contrib_row: pd.Series, k: int = 5) -> pd.DataFrame:
    s = contrib_row.reindex(contrib_row.abs().sort_values(ascending=False).index)[:k]
    return pd.DataFrame({"fitur": s.index, "kontribusi": s.values,
                         "arah": np.where(s.values > 0, "→ FRAUD", "→ Normal")})


def tflite_predict_fn(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(model_path))

    def fn(X):
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        it.resize_tensor_input(i["index"], list(X.shape)); it.allocate_tensors()
        it.set_tensor(i["index"], X.astype("float32")); it.invoke()
        return it.get_tensor(o["index"]).ravel()
    return fn


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from preprocessing import prepare_all

    (_, _, X_te, _, _, y_te), _ = prepare_all()
    thr = json.load(open(MODEL_DIR / "threshold.json"))["threshold"]
    fn = tflite_predict_fn(MODEL_DIR / "fraud_dnn_int8dyn.tflite")
    fr = np.where(y_te == 1)[0]
    p, C = occlusion(fn, X_te[fr])
    tp, fnm = p >= thr, p < thr
    mean_tp = C[tp].mean().sort_values(ascending=False)
    top1 = C[tp].idxmax(axis=1).value_counts()
    res = {"n_fraud_test": int(len(fr)), "n_tp": int(tp.sum()), "n_fn": int(fnm.sum()),
           "mean_contrib_tp_top10": {k: float(v) for k, v in mean_tp.head(10).items()},
           "top1_feature_count_tp": {k: int(v) for k, v in top1.head(8).items()},
           "mean_contrib_fn_top5": {k: float(v) for k, v in
                                    C[fnm].mean().sort_values(ascending=False).head(5).items()},
           "mean_prob_tp": float(p[tp].mean()), "mean_prob_fn": float(p[fnm].mean())}
    json.dump(res, open(RESULT_DIR / "explain_eval.json", "w"), indent=2)

    top = mean_tp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top.index, top.values, color="#c0392b")
    ax.set_xlabel("Rata-rata kontribusi terhadap P(fraud)")
    ax.set_title(f"Fitur paling berpengaruh pada {int(tp.sum())} fraud yang terdeteksi (data uji)")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "uts_explain_tp.png", dpi=160); plt.close(fig)
    print(json.dumps(res, indent=2))
