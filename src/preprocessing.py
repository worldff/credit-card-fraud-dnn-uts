"""Pembersihan data, rekayasa fitur ringan, pembagian data, dan scaling.

Urutan sengaja dibuat: hapus duplikat -> split stratified -> fit scaler HANYA
pada data training, sehingga tidak ada kebocoran informasi (data leakage)
dari validation/testing ke training.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, MODEL_DIR, SEED, TEST_SIZE, VAL_SIZE


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Hapus baris duplikat dan baris dengan nilai kosong/tak hingga."""
    info = {"rows_raw": len(df)}
    info["missing_values"] = int(df.isna().sum().sum())
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    dup_mask = df.duplicated()
    info["duplicates"] = int(dup_mask.sum())
    info["duplicates_fraud"] = int(df.loc[dup_mask, "Class"].sum())
    df = df.loc[~dup_mask].reset_index(drop=True)
    info["rows_clean"] = len(df)
    return df, info


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Transformasi fitur:
    - Amount  -> log1p(Amount): distribusi sangat right-skewed (mean ~88, max ~25.691).
    - Time    -> Hour (jam ke-0..23) lalu dikodekan siklik (sin, cos) karena Time
                 hanya berarti 'detik sejak transaksi pertama' (2 hari), sehingga
                 pola yang relevan adalah jam dalam sehari, bukan nilai absolutnya.
    """
    out = df.copy()
    out["LogAmount"] = np.log1p(out["Amount"])
    hour = (out["Time"] % 86400) / 3600.0
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out.drop(columns=["Time", "Amount"])


FEATURES = [f"V{i}" for i in range(1, 29)] + ["LogAmount", "Hour_sin", "Hour_cos"]


def split(df: pd.DataFrame):
    X = df[FEATURES].values.astype("float32")
    y = df["Class"].values.astype("float32")
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    val_rel = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale(X_train, X_val, X_test, save=True):
    scaler = StandardScaler().fit(X_train)          # fit hanya di training
    if save:
        joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    return (scaler.transform(X_train).astype("float32"),
            scaler.transform(X_val).astype("float32"),
            scaler.transform(X_test).astype("float32"), scaler)


def prepare_all():
    df, info = clean(load_raw())
    df = engineer(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    X_train, X_val, X_test, scaler = scale(X_train, X_val, X_test)
    for name, y in (("train", y_train), ("val", y_val), ("test", y_test)):
        info[f"n_{name}"] = int(len(y))
        info[f"fraud_{name}"] = int(y.sum())
    return (X_train, X_val, X_test, y_train, y_val, y_test), info


def preprocess_raw_rows(df_raw: pd.DataFrame, scaler) -> np.ndarray:
    """Dipakai saat inferensi (aplikasi Streamlit) untuk data mentah format Kaggle."""
    return scaler.transform(engineer(df_raw)[FEATURES].values.astype("float32"))
