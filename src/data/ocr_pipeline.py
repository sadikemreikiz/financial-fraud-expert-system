import json
from pathlib import Path

import pytesseract
from PIL import Image

from src.config import RAW_DIR, OCR_OUTPUT_JSONL, OCR_CONFIG, ensure_directories
from src.utils.file_utils import list_image_files


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
    Run OCR on all files under data/raw and save results as JSONL.
    """
    ensure_directories()

    image_files = list_image_files(RAW_DIR)
    print(f"Found {len(image_files)} files under {RAW_DIR}")

    OCR_OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(OCR_OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for img_path in image_files:
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


if __name__ == "__main__":
    main()
