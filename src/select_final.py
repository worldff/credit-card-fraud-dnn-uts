"""Memilih model final secara objektif berdasarkan data VALIDATION.

Kriteria: rata-rata F1-score validation pada threshold optimal (3 seed) dari
skenario utama S1/S2/S3. Data testing TIDAK dipakai untuk memilih model.
Model seed 42 dari skenario terpilih disalin sebagai model final aplikasi.
"""
import json
import numpy as np
from tensorflow import keras

from config import MODEL_DIR, RESULT_DIR, SEED
import model as _m  # noqa: F401  (registrasi BinaryFocalLoss untuk load_model)

MAIN = ["S1_baseline_bce", "S2_weighted_bce", "S3_focal"]


def main():
    res = json.load(open(RESULT_DIR / "all_results.json"))
    table = {}
    for sc in MAIN:
        rr = [r for r in res if r["scenario"] == sc]
        table[sc] = {
            "val_f1_opt_mean": float(np.mean([r["val@opt"]["f1"] for r in rr])),
            "val_pr_auc_mean": float(np.mean([r["val@opt"]["pr_auc"] for r in rr])),
        }
    best = max(MAIN, key=lambda s: (table[s]["val_f1_opt_mean"], table[s]["val_pr_auc_mean"]))
    r42 = next(r for r in res if r["scenario"] == best and r["seed"] == SEED)
    # simpan ulang tanpa state optimizer (hanya bobot + arsitektur) untuk deployment
    model = keras.models.load_model(MODEL_DIR / f"{best}.keras", compile=False)
    model.save(MODEL_DIR / "fraud_dnn_final.keras")
    with open(MODEL_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": r42["thr_opt"], "scenario": best, "seed": SEED}, f, indent=2)
    with open(RESULT_DIR / "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda s, **k: f.write(s + "\n"))
    out = {"selection_table": table, "final_scenario": best, "seed": SEED,
           "threshold": r42["thr_opt"]}
    with open(RESULT_DIR / "final_selection.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
