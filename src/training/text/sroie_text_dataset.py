import json
from typing import List

from src.config import PROCESSED_DIR

SROIE_TEXT_SAMPLE = PROCESSED_DIR / "sroie_text_sample.jsonl"


def load_sroie_text_sample() -> List[str]:
    """
    Load cleaned text from the SROIE sample file.
    For now we only return the clean_text list (no labels yet).
    """
    texts: List[str] = []

    if not SROIE_TEXT_SAMPLE.exists():
        raise FileNotFoundError(f"SROIE text sample not found: {SROIE_TEXT_SAMPLE}")

    with open(SROIE_TEXT_SAMPLE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["clean_text"])

    return texts


def main():
    texts = load_sroie_text_sample()
    print(f"Loaded {len(texts)} SROIE samples.")

    # Simple stats: word counts of each sample
    word_counts = [len(t.split()) for t in texts]
    print("Word counts:", word_counts)

    # Print first 300 chars of first sample just to inspect
    if texts:
        print("\n=== FIRST SAMPLE (300 chars) ===")
        print(texts[0][:300], "...")


if __name__ == "__main__":
    main()
