from pathlib import Path
import os

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# OCR configuration settings
OCR_CONFIG = {
    "lang": "eng",   # later we can change to 'eng+deu' if needed
    "oem": 3,        # OCR Engine Mode
    "psm": 6,        # Page Segmentation Mode
}

# Output file for OCR results
OCR_OUTPUT_JSONL = INTERIM_DIR / "ocr_results.jsonl"

def ensure_directories():
    """Create required directories if they do not exist."""
    for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        os.makedirs(d, exist_ok=True)
