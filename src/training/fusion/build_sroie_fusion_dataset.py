# src/training/fusion/build_sroie_fusion_dataset.py

"""
Build final SROIE fusion dataset (REAL + FRAUD) for training.

- REAL textler: data/processed/sroie_text_sample.jsonl  -> "clean_text"
- FRAUD textler: data/processed/sroie_synthetic_fraud_text.jsonl -> "fraud_text"
- Image dataset: data/processed/sroie_image_dataset.jsonl -> "image_path", "label"

Her satır:
{
    "text": "...",
    "image_path": "...",
    "text_proba_real": float,
    "text_proba_fraud": float,
    "image_proba_real": float,
    "image_proba_fraud": float,
    "label": 0 veya 1   # 0 = REAL, 1 = FRAUD
}
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import PROCESSED_DIR
from src.training.text.predict_sroie_text import predict_text
from src.training.image.predict_image_classifier import predict_image


# Text kaynakları
REAL_TEXT_FILE = PROCESSED_DIR / "sroie_text_sample.jsonl"
FAKE_TEXT_FILE = PROCESSED_DIR / "sroie_synthetic_fraud_text.jsonl"

# Image dataset (REAL + FRAUD)
IMAGE_DATA_FILE = PROCESSED_DIR / "sroie_image_dataset.jsonl"

# Çıkış
OUTPUT_FILE = PROCESSED_DIR / "sroie_fusion_dataset.jsonl"


def load_real_texts() -> List[str]:
    """REAL (0) textleri oku."""
    texts: List[str] = []
    with open(REAL_TEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row.get("clean_text", ""))
    return texts


def load_fraud_texts() -> List[str]:
    """FRAUD (1) textleri oku."""
    texts: List[str] = []
    with open(FAKE_TEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # build_sroie_synthetic_fraud_text.py içinde "fraud_text" alanını yazmıştık
            texts.append(row.get("fraud_text", ""))
    return texts


def load_images_by_label(label: int) -> List[Path]:
    """
    Image dataset'ten verilen label'a sahip görselleri oku.
    label: 0 = REAL, 1 = FRAUD
    """
    images: List[Path] = []
    with open(IMAGE_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("label") == label:
                images.append(Path(row["image_path"]))
    return images


def pair_samples(
    texts: List[str], images: List[Path]
) -> List[Tuple[str, Path]]:
    """
    Text ve image'ları index’e göre eşleştir.
    Aynı doküman olmak zorunda değil; önemli olan aynı sınıf (REAL/REAL, FRAUD/FRAUD).
    """
    n = min(len(texts), len(images))
    return list(zip(texts[:n], images[:n]))


def main() -> None:
    print("=== Building SROIE fusion dataset ===")
    print(f"Reading REAL texts from:    {REAL_TEXT_FILE}")
    print(f"Reading FRAUD texts from:   {FAKE_TEXT_FILE}")
    print(f"Reading image data from:    {IMAGE_DATA_FILE}")

    # Textleri yükle
    real_texts = load_real_texts()
    fraud_texts = load_fraud_texts()

    # Image'ları yükle
    real_images = load_images_by_label(0)
    fraud_images = load_images_by_label(1)

    print(f"Loaded {len(real_texts)} REAL texts,  {len(real_images)} REAL images")
    print(f"Loaded {len(fraud_texts)} FRAUD texts, {len(fraud_images)} FRAUD images")

    # Pair'ler
    real_pairs = pair_samples(real_texts, real_images)
    fraud_pairs = pair_samples(fraud_texts, fraud_images)

    print(f"Using {len(real_pairs)} REAL pairs and {len(fraud_pairs)} FRAUD pairs")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        # REAL samples (label = 0)
        for text, img_path in real_pairs:
            text_res = predict_text(text)
            img_res = predict_image(img_path)

            record: Dict = {
                "text": text,
                "image_path": str(img_path),
                "text_proba_real": float(text_res["proba_real"]),
                "text_proba_fraud": float(text_res["proba_fraud"]),
                "image_proba_real": float(img_res["proba_real"]),
                "image_proba_fraud": float(img_res["proba_fraud"]),
                "label": 0,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

        # FRAUD samples (label = 1)
        for text, img_path in fraud_pairs:
            text_res = predict_text(text)
            img_res = predict_image(img_path)

            record = {
                "text": text,
                "image_path": str(img_path),
                "text_proba_real": float(text_res["proba_real"]),
                "text_proba_fraud": float(text_res["proba_fraud"]),
                "image_proba_real": float(img_res["proba_real"]),
                "image_proba_fraud": float(img_res["proba_fraud"]),
                "label": 1,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\n✅ Saved {n_written} fusion records to:")
    print(f"   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
