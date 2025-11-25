import json
from pathlib import Path

import pytesseract
from PIL import Image

from src.config import RAW_DIR, INTERIM_DIR, OCR_CONFIG, ensure_directories
from src.utils.file_utils import list_image_files

# We will read a small sample from SROIE train/images
SROIE_TRAIN_IMG_DIR = RAW_DIR / "sroie" / "train" / "img"
OUTPUT_FILE = INTERIM_DIR / "sroie_ocr_sample.jsonl"
NUM_SAMPLES = 5  # how many documents to process as a sample


def run_ocr_on_image(image_path: Path) -> str:
    """
    Run OCR on a single image file and return extracted text.
    """
    image = Image.open(image_path).convert("RGB")
    custom_config = f'-l {OCR_CONFIG["lang"]} --oem {OCR_CONFIG["oem"]} --psm {OCR_CONFIG["psm"]}'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text


def main():
    """
    Run OCR on a small sample of SROIE train images and save results as JSONL.
    """
    ensure_directories()

    if not SROIE_TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"SROIE train image directory not found: {SROIE_TRAIN_IMG_DIR}")

    all_images = list_image_files(SROIE_TRAIN_IMG_DIR)
    if not all_images:
        raise RuntimeError(f"No images found under {SROIE_TRAIN_IMG_DIR}")

    sample_images = all_images[:NUM_SAMPLES]
    print(f"Found {len(all_images)} images in SROIE train; processing first {len(sample_images)}.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for img_path in sample_images:
            try:
                text = run_ocr_on_image(img_path)
                record = {
                    "image_path": str(img_path),
                    "ocr_text": text,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"OCR done: {img_path.name}")
            except Exception as e:
                print(f"[WARNING] OCR failed for {img_path}: {e}")

    print(f"\nSaved OCR sample to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
