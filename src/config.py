from pathlib import Path
import os

# ===========================
# Paths & directories
# ===========================

# Project root (financial-fraud-expert-system/)
BASE_DIR = Path(__file__).resolve().parents[1]

# --- Data directories ---
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Model directories ---
MODEL_DIR = BASE_DIR / "models"
MODEL_TEXT_DIR = MODEL_DIR / "text"
MODEL_IMAGE_DIR = MODEL_DIR / "image"
MODEL_FUSION_DIR = MODEL_DIR / "fusion"

# Create all required directories if they do not exist
for d in [
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    MODEL_TEXT_DIR,
    MODEL_IMAGE_DIR,
    MODEL_FUSION_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ===========================
# OCR configuration
# ===========================

OCR_CONFIG = {
    "lang": "eng",  # language (later we can switch to 'eng+deu')
    "oem": 3,       # OCR Engine Mode
    "psm": 6,       # Page Segmentation Mode
}

# Default OCR output file (for generic OCR scripts)
OCR_OUTPUT_JSONL = INTERIM_DIR / "ocr_results.jsonl"


def ensure_directories() -> None:
    """
    Backward-compatible helper used by some scripts.
    Directories are already created above, but we keep this
    function so older code that calls it still works.
    """
    for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        os.makedirs(d, exist_ok=True)
