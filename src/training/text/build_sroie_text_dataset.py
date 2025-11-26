"""
Build final SROIE text dataset (REAL + FRAUD) for training.

- Reads REAL cleaned texts from:  data/processed/sroie_text_sample.jsonl
- Reads FRAUD synthetic texts from: data/processed/sroie_synthetic_fraud_text.jsonl
- Writes combined dataset to:     data/processed/sroie_text_dataset.jsonl

Each output line is a JSON object with:
    {
        "clean_text": "<text>",
        "label": 0 or 1      # 0 = REAL, 1 = FRAUD
    }
"""

import json
from pathlib import Path
from typing import List, Dict

from src.config import PROCESSED_DIR


# ---- File paths ----
REAL_FILE = PROCESSED_DIR / "sroie_text_sample.jsonl"
SYNTH_FAKE_FILE = PROCESSED_DIR / "sroie_synthetic_fraud_text.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "sroie_text_dataset.jsonl"


def load_real_texts() -> List[str]:
    """Load cleaned REAL texts from sroie_text_sample.jsonl."""
    texts: List[str] = []

    if not REAL_FILE.exists():
        raise FileNotFoundError(f"REAL file not found: {REAL_FILE}")

    with REAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # We expect a field called "clean_text"
            text = row.get("clean_text", "").strip()
            if text:
                texts.append(text)

    return texts


def load_fraud_texts() -> List[str]:
    """Load FRAUD synthetic texts from sroie_synthetic_fraud_text.jsonl."""
    texts: List[str] = []

    if not SYNTH_FAKE_FILE.exists():
        print(f"[WARNING] Fraud file not found: {SYNTH_FAKE_FILE}")
        print("          Continuing with REAL texts only.")
        return texts

    with SYNTH_FAKE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # !!! ÖNEMLİ !!!
            # Eğer senin dosyanda alan adı farklıysa (ör: "text"),
            # buradaki "fraud_text" kısmını ona göre değiştir:
            text = row.get("fraud_text", "").strip()
            if text:
                texts.append(text)

    return texts


def build_dataset() -> None:
    """Combine REAL + FRAUD texts and save final dataset."""
    real_texts = load_real_texts()
    fraud_texts = load_fraud_texts()

    print(f"Loaded {len(real_texts)} REAL texts.")
    print(f"Loaded {len(fraud_texts)} FRAUD texts.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as f_out:

        # REAL samples -> label = 0
        for txt in real_texts:
            record: Dict[str, object] = {
                "clean_text": txt,
                "label": 0,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

        # FRAUD samples -> label = 1
        for txt in fraud_texts:
            record = {
                "clean_text": txt,
                "label": 1,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\nSaved {n_written} samples to:")
    print(f"  {OUTPUT_FILE}")


def main() -> None:
    build_dataset()


if __name__ == "__main__":
    main()
