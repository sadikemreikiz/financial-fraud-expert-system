import json
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from src.config import PROCESSED_DIR, MODEL_TEXT_DIR

INPUT_FILE = PROCESSED_DIR / "sroie_synthetic_fraud_text.jsonl"
MODEL_PATH = MODEL_TEXT_DIR / "sroie_text_clf.pkl"


def load_dataset() -> Tuple[List[str], List[int]]:
    texts = []
    labels = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["clean_text"])
            labels.append(row["label"])

    return texts, labels


def main():
    print(f"Loading dataset from: {INPUT_FILE}")
    texts, labels = load_dataset()

    print(f"Loaded {len(texts)} samples.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Classifier
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_vec, y_train)

    # Evaluation
    preds = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    print("\n=== Accuracy ===")
    print(acc)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    # Save model + vectorizer
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "clf": clf}, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
