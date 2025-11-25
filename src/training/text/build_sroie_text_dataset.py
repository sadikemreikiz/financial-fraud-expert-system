import json
from pathlib import Path

from src.config import PROCESSED_DIR

INPUT_FILE = PROCESSED_DIR / "sroie_text_sample.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "sroie_text_dataset.jsonl"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for line in f_in:
            row = json.loads(line)

            # Dummy label (0 = real)
            row["label"] = 0

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
