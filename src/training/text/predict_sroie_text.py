import json
from typing import Dict

import joblib

from src.config import PROCESSED_DIR, MODEL_TEXT_DIR

MODEL_PATH = MODEL_TEXT_DIR / "sroie_text_clf.pkl"
SAMPLE_FILE = PROCESSED_DIR / "sroie_text_sample.jsonl"


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    return bundle  # {"vectorizer": ..., "clf": ...}


def predict_text(text: str) -> Dict:
    bundle = load_model()
    vectorizer = bundle["vectorizer"]
    clf = bundle["clf"]

    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]
    pred = clf.predict(X)[0]

    return {
        "pred_label": int(pred),
        "proba_real": float(proba[0]),
        "proba_fraud": float(proba[1]),
    }


def load_first_sample_text() -> str:
    """Helper: read first clean_text from sroie_text_sample.jsonl."""
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        line = next(f)
        row = json.loads(line)
        return row["clean_text"]


def main():
    sample_text = load_first_sample_text()
    print("=== SAMPLE TEXT (first 200 chars) ===")
    print(sample_text[:200], "...\n")

    result = predict_text(sample_text)
    label_str = "FRAUD" if result["pred_label"] == 1 else "REAL"

    print("=== PREDICTION ===")
    print(f"Predicted label: {label_str} (raw={result['pred_label']})")
    print(f"P(real=0):  {result['proba_real']:.3f}")
    print(f"P(fraud=1): {result['proba_fraud']:.3f}")


if __name__ == "__main__":
    main()
