from pathlib import Path
import json

from src.config import RAW_DIR, PROCESSED_DIR

# Directories
SROIE_TRAIN_IMG_DIR = RAW_DIR / "sroie" / "train" / "img"
FRAUD_IMG_DIR = PROCESSED_DIR / "sroie_synthetic_fraud_images"

# Output manifest file (image path + label)
# label: 0 = REAL, 1 = FRAUD
OUTPUT_FILE = PROCESSED_DIR / "sroie_image_dataset.jsonl"


def main() -> None:
    records = []

    # Real images
    real_images = sorted(SROIE_TRAIN_IMG_DIR.glob("*.jpg"))
    print(f"Found {len(real_images)} REAL images.")
    for img_path in real_images:
        records.append(
            {
                "image_path": str(img_path),
                "label": 0,
            }
        )

    # Fraud / synthetic images
    fraud_images = sorted(FRAUD_IMG_DIR.glob("*.jpg"))
    print(f"Found {len(fraud_images)} FRAUD images.")
    for img_path in fraud_images:
        records.append(
            {
                "image_path": str(img_path),
                "label": 1,
            }
        )

    # Write JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} records to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
