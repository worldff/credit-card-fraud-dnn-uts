"""Pipeline input berbasis tf.data (sesuai RTM: 'Bangun input pipeline menggunakan tf.data').

Data hasil preprocessing (array float32) diubah menjadi tensor lalu dialirkan ke model
melalui tf.data.Dataset dengan tahapan:
    from_tensor_slices -> cache -> shuffle (khusus training) -> batch -> prefetch
"""
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def make_dataset(X, y=None, batch_size=2048, shuffle=False, seed=42):
    """Membuat tf.data.Dataset dari array fitur (dan label bila ada).

    - from_tensor_slices : memotong tensor (N, 31) menjadi elemen per transaksi.
    - cache              : menyimpan elemen di memori setelah epoch pertama.
    - shuffle            : mengacak urutan setiap epoch (buffer = seluruh data training,
                           sehingga sampel fraud tersebar merata ke semua batch).
    - batch              : mengelompokkan elemen menjadi tensor (batch_size, 31).
    - prefetch           : menyiapkan batch berikutnya selagi model memproses batch saat ini.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y) if y is not None else X)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def describe(ds, name):
    """Ringkasan isi dataset: spesifikasi tensor dan jumlah batch."""
    return {"name": name, "element_spec": str(ds.element_spec),
            "num_batches": int(ds.cardinality().numpy())}
