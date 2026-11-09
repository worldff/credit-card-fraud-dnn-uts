"""Menyiapkan data/creditcard.csv dari salinan dataset Kaggle yang disimpan di repositori.

File asli Kaggle (archive.zip dari https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud,
berisi creditcard.csv 150.828.752 byte) dipecah menjadi beberapa bagian < 25 MB agar dapat
diunggah ke GitHub melalui browser:

    data/creditcard_kaggle.zip.part1, .part2, .part3

Skrip ini menggabungkan bagian-bagian tersebut, mengekstrak creditcard.csv, lalu memeriksa
checksum MD5 agar dipastikan sama persis dengan file asli Kaggle.

Pemakaian:  python src/get_dataset.py
"""
import hashlib
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "creditcard.csv"
PARTS = sorted((ROOT / "data").glob("creditcard_kaggle.zip.part*"))
MD5_ASLI = "e90efcb83d69faf99fcab8b0255024de"   # md5 creditcard.csv dari Kaggle


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def ensure_dataset(verbose=True) -> Path:
    if DATA.exists():
        if verbose:
            print("Dataset sudah ada:", DATA)
        return DATA
    if not PARTS:
        raise FileNotFoundError("Bagian dataset (data/creditcard_kaggle.zip.part*) tidak ditemukan")
    buf = io.BytesIO(b"".join(p.read_bytes() for p in PARTS))
    with zipfile.ZipFile(buf) as z:
        z.extract("creditcard.csv", DATA.parent)
    ok = md5(DATA) == MD5_ASLI
    if verbose:
        print(f"Digabung dari {len(PARTS)} bagian -> {DATA} ({DATA.stat().st_size:,} byte), "
              f"MD5 {'cocok dengan file asli Kaggle' if ok else 'TIDAK COCOK'}")
    if not ok:
        DATA.unlink()
        raise ValueError("Checksum dataset tidak cocok; bagian file mungkin rusak/tidak lengkap")
    return DATA


if __name__ == "__main__":
    ensure_dataset()
