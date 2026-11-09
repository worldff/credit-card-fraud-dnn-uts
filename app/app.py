"""Aplikasi Streamlit — Deteksi Fraud Transaksi Kartu Kredit (DNN, model TFLite int8).

Versi UTS (penyempurnaan dari Tugas Proyek 1):
  1. Validasi input dan deteksi input di luar domain data latih (src/input_domain.py)
  2. Penjelasan prediksi per transaksi dengan occlusion sensitivity (src/explain.py)
  3. Pengaturan threshold interaktif beserta estimasi precision/recall dari data validation

Jalankan dari root repo:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:                                    # LiteRT (ringan, tanpa TensorFlow penuh)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from explain import occlusion, top_features  # noqa: E402
from input_domain import RAW_COLS, check_rows  # noqa: E402
from preprocessing import preprocess_raw_rows  # noqa: E402

MODEL_DIR = ROOT / "outputs" / "models"
RESULT_DIR = ROOT / "outputs" / "results"
SAMPLE_CSV = ROOT / "data" / "contoh_transaksi_uji.csv"


@st.cache_resource
def load_artifacts():
    # Fitur unik #2: model final terkompresi (TFLite, bobot int8) -> 10 KB
    model = Interpreter(model_path=str(MODEL_DIR / "fraud_dnn_int8dyn.tflite"))
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    thr = json.load(open(MODEL_DIR / "threshold.json"))["threshold"]
    domain = json.load(open(MODEL_DIR / "input_domain.json"))
    return model, scaler, thr, domain


@st.cache_data
def load_val_curve():
    """Probabilitas model final pada data validation untuk estimasi precision/recall."""
    sel = json.load(open(RESULT_DIR / "final_selection.json"))
    p = np.load(RESULT_DIR / f"prob_val_{sel['final_scenario']}.npy")
    y = np.load(RESULT_DIR / "y_val.npy")
    return p, y


def val_metrics(t):
    p, y = load_val_curve()
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, tp, fp, fn


def model_prob(X):
    model = load_artifacts()[0]
    inp, out_det = model.get_input_details()[0], model.get_output_details()[0]
    model.resize_tensor_input(inp["index"], list(X.shape))
    model.allocate_tensors()
    model.set_tensor(inp["index"], X.astype("float32"))
    model.invoke()
    return model.get_tensor(out_det["index"]).ravel()


def predict(df_raw, thr):
    """Validasi input -> preprocessing (sama dengan saat pelatihan) -> TFLite -> threshold."""
    _, scaler, _, domain = load_artifacts()
    out = df_raw.copy()
    out = out.join(check_rows(out, domain))
    ok = out["cek_input"] != "TIDAK VALID"
    out["prob_fraud"] = np.nan
    out["prediksi"] = "—"
    if ok.any():
        X = preprocess_raw_rows(out.loc[ok, RAW_COLS].astype(float), scaler)
        prob = model_prob(X)
        out.loc[ok, "prob_fraud"] = prob
        out.loc[ok, "prediksi"] = np.where(prob >= thr, "FRAUD", "Normal")
    return out


def explain_one(row_raw):
    scaler = load_artifacts()[1]
    X = preprocess_raw_rows(pd.DataFrame([row_raw])[RAW_COLS].astype(float), scaler)
    _, contrib = occlusion(model_prob, X)
    return top_features(contrib.iloc[0], k=6)


def show_explanation(row_raw):
    tf_ = explain_one(row_raw)
    st.markdown("**Penjelasan prediksi** — fitur yang paling memengaruhi probabilitas fraud "
                "(occlusion: fitur diganti nilai rata-rata data latih, lalu dilihat perubahan "
                "probabilitasnya).")
    c1, c2 = st.columns([1, 1])
    c1.dataframe(tf_.assign(kontribusi=tf_.kontribusi.map(lambda v: f"{v:+.4f}")),
                 hide_index=True, width="stretch")
    c2.bar_chart(tf_, x="fitur", y="kontribusi", horizontal=True, sort="-kontribusi", height=240)


st.set_page_config(page_title="Deteksi Fraud Kartu Kredit", page_icon="💳", layout="wide")
_, _, T_OPT, DOMAIN = load_artifacts()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Informasi model")
    st.write("**Arsitektur:** DNN 64-32-16 (BatchNorm, ReLU, Dropout)")
    tfl_path = MODEL_DIR / "fraud_dnn_int8dyn.tflite"
    st.metric("Ukuran model TFLite int8", f"{tfl_path.stat().st_size / 1024:.1f} KB")
    try:
        tfl = json.load(open(RESULT_DIR / "tflite_results.json"))
        st.caption(f"Model Keras asli: {tfl['keras']['size_kb']:.1f} KB · keputusan sama "
                   f"{100 * tfl['tflite_int8dyn']['pred_agreement']:.3f}% pada data uji")
    except Exception:
        pass
    st.metric("Threshold optimal (dari data validation)", f"{T_OPT:.4f}")
    st.caption("Fitur unik 1: threshold dipilih dari data validation, bukan 0,5. "
               "Fitur unik 2: model dikompresi ke TensorFlow Lite int8.")
    st.divider()
    st.subheader("Pengaturan threshold")
    THR = st.slider("Threshold keputusan", 0.01, 0.99, float(T_OPT), 0.0001,
                    format="%.4f", help="Bawaan: threshold optimal. Turunkan untuk menangkap "
                    "lebih banyak fraud (recall naik), naikkan untuk mengurangi alarm palsu.")
    prec, rec, tp, fp, fn = val_metrics(THR)
    st.caption(f"Estimasi pada data validation (71 fraud): precision **{prec:.1%}**, "
               f"recall **{rec:.1%}** (TP {tp}, FP {fp}, FN {fn}).")
    if abs(THR - T_OPT) > 1e-4:
        st.warning("Threshold berbeda dari threshold optimal yang dilaporkan.")

st.title("💳 Deteksi Fraud Transaksi Kartu Kredit")
st.caption("Deep Neural Network · dataset Credit Card Fraud Detection (ULB, Kaggle)")

tab1, tab2 = st.tabs(["Unggah CSV", "Input manual"])


def show_results(df):
    res = predict(df, THR)
    ok = res["cek_input"] != "TIDAK VALID"
    res["prediksi_thr_0.5"] = np.where(~ok, "—", np.where(res["prob_fraud"] >= 0.5, "FRAUD", "Normal"))
    n_f = int((res["prediksi"] == "FRAUD").sum())
    n_05 = int((res["prediksi_thr_0.5"] == "FRAUD").sum())
    n_ood = int((res["cek_input"] == "DI LUAR DOMAIN").sum())
    n_bad = int((~ok).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah transaksi", f"{len(res):,}".replace(",", "."))
    c2.metric("Terdeteksi fraud", n_f, delta=f"{n_f - n_05:+d} vs thr 0,5")
    c3.metric("Threshold dipakai", f"{THR:.4f}")
    c4.metric("Input di luar domain / tidak valid", f"{n_ood} / {n_bad}")
    if n_bad:
        st.error(f"{n_bad} transaksi tidak valid dan tidak diprediksi (lihat kolom keterangan_input).")
    if n_ood:
        st.warning(f"{n_ood} transaksi memiliki nilai di luar rentang data latih; "
                   "prediksinya perlu diperiksa manual.")
    cols = ["prob_fraud", "prediksi", "prediksi_thr_0.5", "cek_input", "keterangan_input",
            "Amount", "Time"]
    if "Class" in res.columns:
        res["label_asli"] = np.where(res["Class"] == 1, "FRAUD", "Normal")
        res["status"] = np.where(~ok, "—", np.where(res["label_asli"] == res["prediksi"],
                                                     "✅ Benar", "❌ Salah"))
        jenis = {("FRAUD", "FRAUD"): "TP", ("Normal", "Normal"): "TN",
                 ("Normal", "FRAUD"): "FP", ("FRAUD", "Normal"): "FN"}
        res["jenis"] = [jenis.get((a, p), "—") for a, p in zip(res["label_asli"], res["prediksi"])]
        tp_ = int((res.jenis == "TP").sum()); fp_ = int((res.jenis == "FP").sum())
        fn_ = int((res.jenis == "FN").sum())
        st.write(f"**Evaluasi terhadap label asli:** benar {int((res.status == '✅ Benar').sum())} "
                 f"dari {int(ok.sum())} · TP {tp_} · FN {fn_} · FP {fp_} · "
                 f"recall {tp_ / max(tp_ + fn_, 1):.2%} · precision {tp_ / max(tp_ + fp_, 1):.2%}")
        cols = ["label_asli", "status", "jenis"] + cols
    with st.expander("Fitur unik 1 — perbandingan threshold 0,5 vs threshold yang dipakai"):
        st.write(f"Dengan threshold 0,5 terdeteksi **{n_05}** transaksi fraud; dengan threshold "
                 f"{THR:.4f} terdeteksi **{n_f}** transaksi fraud.")
    res.insert(0, "no", np.arange(1, len(res) + 1))
    view = res[["no"] + cols].sort_values("prob_fraud", ascending=False, na_position="last").head(500)
    st.dataframe(view, width="stretch", hide_index=True,
                 height=min(38 * (len(view) + 1) + 4, 760))
    st.download_button("Unduh hasil prediksi (CSV)", res.to_csv(index=False).encode(),
                       "hasil_prediksi.csv")
    valid_no = res.loc[ok, "no"].tolist()
    if valid_no:
        st.divider()
        pick = st.selectbox("Jelaskan prediksi transaksi nomor", valid_no,
                            index=valid_no.index(int(view["no"].iloc[0])) if len(view) else 0)
        r = res.loc[res["no"] == pick].iloc[0]
        st.write(f"Transaksi no. {pick}: prediksi **{r['prediksi']}**, "
                 f"P(fraud) = {r['prob_fraud']:.4f}"
                 + (f", label asli **{r['label_asli']}**" if "label_asli" in res else ""))
        show_explanation({c: r[c] for c in RAW_COLS})


with tab1:
    st.write("Unggah file CSV berformat sama dengan dataset Kaggle (kolom `Time`, `V1`–`V28`, "
             "`Amount`; kolom `Class` opsional untuk mengecek prediksi benar/salah).")
    f = st.file_uploader("File CSV", type="csv")
    use_sample = SAMPLE_CSV.exists() and st.checkbox("Gunakan contoh data uji (20 transaksi)")
    df = pd.read_csv(f) if f is not None else (pd.read_csv(SAMPLE_CSV) if use_sample else None)
    if df is not None:
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            st.error(f"Kolom tidak ditemukan: {missing}")
        else:
            show_results(df)

with tab2:
    st.write("Isi nilai fitur satu transaksi (V1–V28 adalah komponen PCA), atau ambil dari "
             "contoh data uji.")
    sample = pd.read_csv(SAMPLE_CSV) if SAMPLE_CSV.exists() else None
    base = {c: (50.0 if c == "Amount" else 0.0) for c in RAW_COLS}
    true_label = None
    if sample is not None:
        opts = ["(kosong)"] + [f"Contoh {i + 1} — label asli: "
                               f"{'FRAUD' if r.Class == 1 else 'Normal'}"
                               for i, r in sample.iterrows()]
        pick = st.selectbox("Isi otomatis dari contoh", opts)
        if pick != "(kosong)":
            row = sample.iloc[opts.index(pick) - 1]
            base = {c: float(row[c]) for c in RAW_COLS}
            true_label = "FRAUD" if row["Class"] == 1 else "Normal"
    cols = st.columns(6)
    vals = {}
    for i, c in enumerate(RAW_COLS):
        vals[c] = cols[i % 6].number_input(c, value=base[c], format="%.4f",
                                           key=f"in_{c}_{base[c]}")
    if st.button("Prediksi", type="primary"):
        res = predict(pd.DataFrame([vals]), THR)
        r = res.iloc[0]
        if r["cek_input"] == "TIDAK VALID":
            st.error(f"Input tidak valid: {r['keterangan_input']}. Transaksi tidak diprediksi.")
        else:
            if r["cek_input"] == "DI LUAR DOMAIN":
                st.warning(f"Peringatan: {r['keterangan_input']}. Prediksi di bawah ini perlu "
                           "diperiksa manual karena model belum pernah melihat nilai seperti ini.")
            p, label = float(r["prob_fraud"]), r["prediksi"]
            (st.error if label == "FRAUD" else st.success)(
                f"Prediksi: **{label}** — P(fraud) = {p:.4f} (threshold {THR:.4f})")
            if true_label:
                st.info(f"Label asli: **{true_label}** → prediksi "
                        f"{'BENAR ✅' if true_label == label else 'SALAH ❌'}")
            show_explanation(vals)
