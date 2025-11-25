import json
from pathlib import Path

from src.data.preprocessing import clean_text, extract_basic_features
from src.config import OCR_OUTPUT_JSONL

def main():
    # Load the OCR results
    with open(OCR_OUTPUT_JSONL, "r", encoding="utf-8") as f:
        line = f.readline()  # read first sample
        data = json.loads(line)

    raw_text = data.get("ocr_text", "")
    print("\n=== RAW OCR TEXT ===")
    print(raw_text[:500], "...\n")   # print only first 500 chars

    # Clean the text
    clean = clean_text(raw_text)
    print("=== CLEANED TEXT ===")
    print(clean[:500], "...\n")

    # Extract basic features
    features = extract_basic_features(raw_text)
    print("=== FEATURES ===")
    print(features)

if __name__ == "__main__":
    main()
