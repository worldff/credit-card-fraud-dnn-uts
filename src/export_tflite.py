"""Fitur unik #2: kompresi model final ke TensorFlow Lite.

Dua varian dibuat:
  - fraud_dnn_fp32.tflite      : konversi biasa (BatchNorm dilipat ke Dense)
  - fraud_dnn_int8dyn.tflite   : dynamic-range quantization (bobot int8)
Bukti dampak: ukuran file, latensi inferensi satu transaksi, dan metrik pada data
testing dengan threshold yang SAMA dengan model Keras.
"""
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import MODEL_DIR, RESULT_DIR
from evaluate import metrics_at
from preprocessing import prepare_all
import model as _m  # noqa: F401

try:                                   # LiteRT (pengganti tf.lite pada TF terbaru)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:                    # fallback TF lama
    Interpreter = tf.lite.Interpreter


def convert(model, quantize):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def tflite_predict(path, X):
    it = Interpreter(model_path=str(path))
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    it.resize_tensor_input(inp["index"], [len(X), X.shape[1]]); it.allocate_tensors()
    it.set_tensor(inp["index"], X); it.invoke()
    return it.get_tensor(out["index"]).ravel()


def latency_tflite(path, x1, n=2000):
    it = Interpreter(model_path=str(path)); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    for _ in range(50):
        it.set_tensor(inp["index"], x1); it.invoke()
    t0 = time.perf_counter()
    for _ in range(n):
        it.set_tensor(inp["index"], x1); it.invoke(); it.get_tensor(out["index"])
    return (time.perf_counter() - t0) / n * 1000


def latency_keras(model, x1, n=300):
    for _ in range(20):
        model(x1, training=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model(x1, training=False).numpy()
    return (time.perf_counter() - t0) / n * 1000


def main():
    (_, _, X_test, _, _, y_test), _ = prepare_all()
    thr = json.load(open(MODEL_DIR / "threshold.json"))["threshold"]
    kpath = MODEL_DIR / "fraud_dnn_final.keras"
    model = keras.models.load_model(kpath, compile=False)
    x1 = X_test[:1]

    p_keras = model.predict(X_test, batch_size=8192, verbose=0).ravel()
    rows = {"keras": {"file": kpath.name, "size_kb": kpath.stat().st_size / 1024,
                      "latency_ms": latency_keras(model, x1),
                      "test": metrics_at(y_test, p_keras, thr), "max_abs_diff": 0.0,
                      "pred_agreement": 1.0}}
    for name, q in (("tflite_fp32", False), ("tflite_int8dyn", True)):
        path = MODEL_DIR / f"fraud_dnn_{name.split('_')[1]}.tflite"
        path.write_bytes(convert(model, q))
        p = tflite_predict(path, X_test)
        rows[name] = {"file": path.name, "size_kb": path.stat().st_size / 1024,
                      "latency_ms": latency_tflite(path, x1),
                      "test": metrics_at(y_test, p, thr),
                      "max_abs_diff": float(np.abs(p - p_keras).max()),
                      "pred_agreement": float(np.mean((p >= thr) == (p_keras >= thr)))}
    with open(RESULT_DIR / "tflite_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    for k, v in rows.items():
        t = v["test"]
        print(f"{k:15s} {v['size_kb']:8.1f} KB  {v['latency_ms']:.4f} ms  "
              f"F1={t['f1']:.4f} P={t['precision']:.4f} R={t['recall']:.4f} "
              f"PR-AUC={t['pr_auc']:.4f} maxdiff={v['max_abs_diff']:.2e} "
              f"agree={v['pred_agreement']:.6f}")


if __name__ == "__main__":
    main()
