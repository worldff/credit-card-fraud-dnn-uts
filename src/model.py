"""Arsitektur DNN dan Binary Focal Loss (dipakai pada skenario S3 dan H1-H4)."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


@keras.utils.register_keras_serializable(package="fraud")
class BinaryFocalLoss(keras.losses.Loss):
    """Focal Loss (Lin et al., 2017) untuk klasifikasi biner.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - alpha   : bobot kelas positif (fraud); kelas negatif mendapat (1 - alpha).
    - gamma   : faktor fokus; menurunkan kontribusi sampel 'mudah' (p_t tinggi),
                sehingga gradien didominasi sampel sulit (fraud yang mirip transaksi normal).
    gamma = 0 dan alpha = 0.5 -> setara dengan BCE biasa (dikali 0.5).
    """

    def __init__(self, alpha=0.75, gamma=2.0, name="binary_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha, self.gamma = alpha, gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), y_pred.dtype)
        eps = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        return -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha, "gamma": self.gamma}


def build_dnn(n_features, units=(64, 32, 16), dropout=(0.3, 0.3, 0.2), l2=1e-4,
              output_bias=None, name="fraud_dnn"):
    """DNN fully-connected: [Dense -> BatchNorm -> ReLU -> Dropout] x N -> Dense(1, sigmoid)."""
    bias_init = (keras.initializers.Constant(output_bias)
                 if output_bias is not None else "zeros")
    inp = keras.Input(shape=(n_features,), name="fitur_transaksi")
    x = inp
    for i, (u, d) in enumerate(zip(units, dropout), start=1):
        x = layers.Dense(u, kernel_regularizer=regularizers.l2(l2),
                         kernel_initializer="he_normal", name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(d, name=f"dropout_{i}")(x)
    out = layers.Dense(1, activation="sigmoid", bias_initializer=bias_init,
                       name="prob_fraud")(x)
    return keras.Model(inp, out, name=name)


def initial_bias(y):
    """Bias awal = log(pos/neg) agar prediksi awal ~ prior kelas (stabilkan epoch awal)."""
    pos = float(np.sum(y)); neg = float(len(y) - pos)
    return np.log(pos / neg)
