"""Konfigurasi global proyek Deteksi Fraud Kartu Kredit (DNN)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
RESULT_DIR = ROOT / "outputs" / "results"

SEED = 42
TEST_SIZE = 0.15   # 15% data testing
VAL_SIZE = 0.15    # 15% data validation (dari total)

# Hyperparameter pelatihan
BATCH_SIZE = 2048
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 10

# Focal loss
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

for d in (FIG_DIR, MODEL_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)
