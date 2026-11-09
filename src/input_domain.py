"""Penyempurnaan UTS #1 — validasi input dan deteksi input di luar domain data latih.

Model DNN hanya dapat dipercaya pada rentang nilai yang pernah dilihat saat pelatihan.
Modul ini menyimpan rentang (minimum–maksimum) setiap fitur pada DATA LATIH, lalu
memeriksa setiap transaksi yang masuk ke aplikasi:

- TIDAK VALID   : ada nilai kosong / bukan angka, Amount negatif, atau Time negatif.
                  Transaksi ini tidak diprediksi.
- DI LUAR DOMAIN: satu atau lebih fitur berada di luar rentang data latih.
                  Transaksi tetap diprediksi, tetapi diberi peringatan agar diperiksa manual.

Jalankan `python input_domain.py` untuk membuat outputs/models/input_domain.json dan
mengevaluasi berapa banyak transaksi validation/testing yang ditandai.
"""
import json

import numpy as np
import pandas as pd

from config import MODEL_DIR, RESULT_DIR

RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
RANGE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]   # Time diubah ke jam (siklik)
DOMAIN_PATH = MODEL_DIR / "input_domain.json"


def build_domain(X_train_unscaled: np.ndarray, features: list[str]) -> dict:
    """Rentang tiap fitur mentah pada data latih (Amount dikembalikan dari LogAmount)."""
    df = pd.DataFrame(X_train_unscaled, columns=features)
    dom = {c: [float(df[c].min()), float(df[c].max())] for c in RANGE_COLS if c in df}
    la = df["LogAmount"]
    dom["Amount"] = [float(np.expm1(la.min())), float(np.expm1(la.max()))]
    return dom


def check_rows(df_raw: pd.DataFrame, domain: dict) -> pd.DataFrame:
    """Kembalikan status per baris: 'OK', 'DI LUAR DOMAIN', atau 'TIDAK VALID' + keterangannya."""
    status, notes = [], []
    num = df_raw[RAW_COLS].apply(pd.to_numeric, errors="coerce")
    for i in range(len(num)):
        row = num.iloc[i]
        bad = [c for c in RAW_COLS if not np.isfinite(row[c])]
        if bad:
            status.append("TIDAK VALID"); notes.append("nilai kosong/bukan angka: " + ", ".join(bad))
            continue
        if row["Amount"] < 0 or row["Time"] < 0:
            status.append("TIDAK VALID"); notes.append("Amount/Time tidak boleh negatif")
            continue
        out = [c for c in RANGE_COLS if not (domain[c][0] <= row[c] <= domain[c][1])]
        if out:
            status.append("DI LUAR DOMAIN"); notes.append("di luar rentang data latih: " + ", ".join(out))
        else:
            status.append("OK"); notes.append("")
    return pd.DataFrame({"cek_input": status, "keterangan_input": notes}, index=df_raw.index)


def _flag_rate(X_unscaled, features, domain):
    df = pd.DataFrame(X_unscaled, columns=features)
    df["Amount"] = np.expm1(df["LogAmount"])
    out = np.zeros(len(df), bool)
    for c in RANGE_COLS:
        out |= ~df[c].between(domain[c][0], domain[c][1])
    return out


if __name__ == "__main__":
    from preprocessing import FEATURES, clean, engineer, load_raw, split

    df, _ = clean(load_raw())
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(engineer(df))
    domain = build_domain(X_tr, FEATURES)
    json.dump(domain, open(DOMAIN_PATH, "w"), indent=1)

    ev = {}
    for name, X, y in (("validation", X_va, y_va), ("testing", X_te, y_te)):
        f = _flag_rate(X, FEATURES, domain)
        ev[name] = {"n": int(len(f)), "flagged": int(f.sum()),
                    "flagged_pct": float(100 * f.mean()),
                    "flagged_fraud": int(f[y == 1].sum()), "fraud": int((y == 1).sum())}
    # contoh input buatan untuk demonstrasi (bukan bagian dari data uji)
    demo = pd.DataFrame([
        {**{c: 0.0 for c in RAW_COLS}, "Time": 3600, "Amount": 50.0},          # normal
        {**{c: 0.0 for c in RAW_COLS}, "Time": 3600, "Amount": 60000.0},       # Amount ekstrem
        {**{c: 0.0 for c in RAW_COLS}, "Time": 3600, "Amount": 50.0, "V14": -40.0},
        {**{c: 0.0 for c in RAW_COLS}, "Time": 3600, "Amount": -5.0},          # tidak valid
    ])
    chk = check_rows(demo, domain)
    ev["contoh_buatan"] = [{"input": d, "cek_input": s, "keterangan": k}
                           for d, s, k in zip(["Amount 50 EUR, V1–V28 = 0",
                                               "Amount 60.000 EUR", "V14 = -40",
                                               "Amount -5 EUR"],
                                              chk.cek_input, chk.keterangan_input)]
    ev["domain_amount"] = domain["Amount"]
    ev["domain_V14"] = domain["V14"]
    json.dump(ev, open(RESULT_DIR / "input_domain_eval.json", "w"), indent=2)
    print(json.dumps(ev, indent=2))
