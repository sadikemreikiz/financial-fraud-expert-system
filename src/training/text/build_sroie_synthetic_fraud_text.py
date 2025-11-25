import json
import re
from typing import List, Tuple

from src.config import PROCESSED_DIR

INPUT_FILE = PROCESSED_DIR / "sroie_text_sample.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "sroie_synthetic_fraud_text.jsonl"


amount_pattern = re.compile(r"\b\d+(\.\d+)?\b")


def make_tampered_text(text: str) -> Tuple[str, bool]:
    """
    Very simple synthetic tampering:
    - find the first numeric amount
    - increase it by 50%
    - replace it in the text

    Returns (tampered_text, success_flag).
    """
    match = amount_pattern.search(text)
    if not match:
        return text, False

    original_str = match.group(0)
    try:
        value = float(original_str)
    except ValueError:
        return text, False

    new_value = round(value * 1.5, 2)
    new_str = f"{new_value:.2f}"

    tampered_text = text[: match.start()] + new_str + text[match.end():]
    return tampered_text, True


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    num_real = 0
    num_fraud = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for line in f_in:
            row = json.loads(line)
            clean_text = row["clean_text"]

            # real version
            real_record = {
                "image_path": row["image_path"],
                "clean_text": clean_text,
                "label": 0,
            }
            f_out.write(json.dumps(real_record, ensure_ascii=False) + "\n")
            num_real += 1

            # tampered (fraud) version
            tampered_text, ok = make_tampered_text(clean_text)
            if ok:
                fraud_record = {
                    "image_path": row["image_path"],
                    "clean_text": tampered_text,
                    "label": 1,
                }
                f_out.write(json.dumps(fraud_record, ensure_ascii=False) + "\n")
                num_fraud += 1

    print(f"Saved synthetic fraud dataset to: {OUTPUT_FILE}")
    print(f"Real samples: {num_real}, Fraud samples: {num_fraud}")


if __name__ == "__main__":
    main()
