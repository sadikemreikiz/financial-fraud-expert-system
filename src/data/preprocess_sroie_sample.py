import json

from src.config import INTERIM_DIR, PROCESSED_DIR
from src.data.preprocessing import clean_text, extract_basic_features

INPUT_FILE = INTERIM_DIR / "sroie_ocr_sample.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "sroie_text_sample.jsonl"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"OCR sample file not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in, start=1):
            row = json.loads(line)
            raw_text = row.get("ocr_text", "")

            clean = clean_text(raw_text)
            features = extract_basic_features(raw_text)

            out = {
                "image_path": row["image_path"],
                "clean_text": clean,
                "digit_count": features["digit_count"],
                "line_count": features["line_count"],
                "euro_amounts": features["euro_amounts"],
                "dates": features["dates"],
            }

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"Processed sample {i}: {row['image_path']}")

    print(f"\nSaved preprocessed SROIE sample to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
