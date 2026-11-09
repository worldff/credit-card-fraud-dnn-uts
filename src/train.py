"""Pelatihan DNN untuk semua skenario + evaluasi pada data testing.

Skenario:
  S1_baseline_bce    : DNN + Binary Cross-Entropy biasa (versi Tugas Proyek 2)
  S2_weighted_bce    : DNN + BCE berbobot kelas (class_weight 'balanced')
  S3_focal           : DNN + Focal Loss (alpha=0.75, gamma=2)
  Setiap skenario dievaluasi pada threshold 0.5 dan threshold optimal F1
  dari data validation (fitur unik #1: optimasi threshold).
Variasi hyperparameter: focal loss gamma {1, 3}, lebar arsitektur, dan optimizer
RMSProp (pembanding Adam). Seluruh data dialirkan ke model melalui pipeline tf.data.
Pemilihan model final dilakukan oleh select_final.py berdasarkan data VALIDATION.
"""
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (BATCH_SIZE, EPOCHS, FOCAL_ALPHA, FOCAL_GAMMA, LEARNING_RATE,
                    MODEL_DIR, PATIENCE, RESULT_DIR, SEED)
from evaluate import best_threshold_f1, metrics_at
from model import BinaryFocalLoss, build_dnn, initial_bias
from data_pipeline import describe, make_dataset
from preprocessing import prepare_all

tf.config.experimental.enable_op_determinism()

SCENARIOS = [
    # nama, loss, gamma, units, dropout, seeds
    dict(name="S1_baseline_bce", loss="bce", seeds=[42, 7, 2024]),
    dict(name="S2_weighted_bce", loss="wbce", seeds=[42, 7, 2024]),
    dict(name="S3_focal", loss="focal", gamma=FOCAL_GAMMA, seeds=[42, 7, 2024]),
    dict(name="H1_focal_gamma1", loss="focal", gamma=1.0, seeds=[42]),
    dict(name="H2_focal_gamma3", loss="focal", gamma=3.0, seeds=[42]),
    dict(name="H3_focal_small", loss="focal", gamma=FOCAL_GAMMA, units=(32, 16),
         dropout=(0.2, 0.2), seeds=[42]),
    dict(name="H4_focal_large", loss="focal", gamma=FOCAL_GAMMA, units=(128, 64, 32),
         dropout=(0.4, 0.3, 0.2), seeds=[42]),
    dict(name="H5_bce_rmsprop", loss="bce", optimizer="rmsprop", seeds=[42]),
]


def make_optimizer(sc):
    if sc.get("optimizer") == "rmsprop":
        return keras.optimizers.RMSprop(LEARNING_RATE)
    return keras.optimizers.Adam(LEARNING_RATE)


def make_loss(sc):
    if sc["loss"] == "focal":
        return BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=sc["gamma"])
    return keras.losses.BinaryCrossentropy()


def class_weights(sc, y):
    if sc["loss"] != "wbce":
        return None
    pos = y.sum(); neg = len(y) - pos
    return {0: len(y) / (2 * neg), 1: len(y) / (2 * pos)}


def run_one(sc, seed, data):
    X_train, X_val, X_test, y_train, y_val, y_test = data
    keras.utils.set_random_seed(seed)
    model = build_dnn(X_train.shape[1], units=sc.get("units", (64, 32, 16)),
                      dropout=sc.get("dropout", (0.3, 0.3, 0.2)),
                      output_bias=initial_bias(y_train))
    model.compile(optimizer=make_optimizer(sc), loss=make_loss(sc),
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                           keras.metrics.AUC(curve="PR", name="pr_auc"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    cbs = [keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                         patience=PATIENCE, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max",
                                             factor=0.5, patience=4, min_lr=1e-5)]
    # ---- pipeline input tf.data ----
    ds_train = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True, seed=seed)
    ds_val = make_dataset(X_val, y_val, BATCH_SIZE)
    t0 = time.time()
    hist = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=cbs,
                     class_weight=class_weights(sc, y_train), verbose=0)
    train_time = time.time() - t0

    pred = lambda X: model.predict(make_dataset(X, batch_size=8192), verbose=0).ravel()
    p_tr, p_val, p_te = pred(X_train), pred(X_val), pred(X_test)
    thr, f1_val = best_threshold_f1(y_val, p_val)
    res = {
        "scenario": sc["name"], "seed": seed,
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmax(hist.history["val_pr_auc"]) + 1),
        "train_time_s": round(train_time, 1), "params": int(model.count_params()),
        "thr_opt": thr, "f1_val_at_thr_opt": f1_val,
        "test@0.5": metrics_at(y_test, p_te, 0.5),
        "test@opt": metrics_at(y_test, p_te, thr),
        "val@0.5": metrics_at(y_val, p_val, 0.5),
        "val@opt": metrics_at(y_val, p_val, thr),
        "train@opt": metrics_at(y_train, p_tr, thr),
    }
    hist_clean = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return model, res, hist_clean, (p_val, p_te)


def main():
    data, info = prepare_all()
    with open(RESULT_DIR / "data_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(info)
    y_val, y_test = data[4], data[5]
    np.save(RESULT_DIR / "y_val.npy", y_val); np.save(RESULT_DIR / "y_test.npy", y_test)
    pipe = [describe(make_dataset(data[0], data[3], BATCH_SIZE, shuffle=True), "train"),
            describe(make_dataset(data[1], data[4], BATCH_SIZE), "validation"),
            describe(make_dataset(data[2], data[5], BATCH_SIZE), "test")]
    with open(RESULT_DIR / "tfdata_pipeline.json", "w") as f:
        json.dump(pipe, f, indent=2)
    print(pipe)

    all_res = []
    for sc in SCENARIOS:
        for seed in sc["seeds"]:
            model, res, hist, (p_val, p_te) = run_one(sc, seed, data)
            all_res.append(res)
            t5, to = res["test@0.5"], res["test@opt"]
            print(f"{sc['name']:18s} seed={seed:<5d} ep={res['epochs_run']:3d} "
                  f"PR-AUC={t5['pr_auc']:.4f} ROC={t5['roc_auc']:.4f} | "
                  f"@0.5 P={t5['precision']:.3f} R={t5['recall']:.3f} F1={t5['f1']:.3f} | "
                  f"@{to['threshold']:.3f} P={to['precision']:.3f} R={to['recall']:.3f} "
                  f"F1={to['f1']:.3f}", flush=True)
            if seed == SEED:
                tag = sc["name"]
                with open(RESULT_DIR / f"history_{tag}.json", "w") as f:
                    json.dump(hist, f)
                np.save(RESULT_DIR / f"prob_val_{tag}.npy", p_val)
                np.save(RESULT_DIR / f"prob_test_{tag}.npy", p_te)
                if tag in ("S1_baseline_bce", "S2_weighted_bce", "S3_focal"):
                    model.save(MODEL_DIR / f"{tag}.keras")
            keras.backend.clear_session()

    with open(RESULT_DIR / "all_results.json", "w") as f:
        json.dump(all_res, f, indent=2)


if __name__ == "__main__":
    main()
