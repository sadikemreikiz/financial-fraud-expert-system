# src/predict/predict_sroie_fusion.py

"""
Final multimodal fraud detection inference pipeline.

This script:
    1. Uses the trained text classifier (logistic regression)
    2. Uses the trained image classifier (ResNet)
    3. Uses the trained fusion classifier (logistic regression on probs)
    4. Runs inference for a given invoice (text + image)
    5. Produces:
        - Text fraud probability
        - Image fraud probability
        - Fusion fraud probability
        - Final decision
        - Expert explanation (rule-based)
"""

from pathlib import Path
import json
from typing import Dict, Any

import numpy as np
import joblib

from src.config import PROCESSED_DIR
from src.training.text.predict_sroie_text import predict_text
from src.training.image.predict_image_classifier import predict_image

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------

PROJECT_ROOT = PROCESSED_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

FUSION_MODEL_PATH = MODELS_DIR / "fusion" / "sroie_fusion_clf.pkl"


# -------------------------------------------------------------
# Expert rule-based reasoning
# -------------------------------------------------------------

def build_explanation(text_fraud: float, image_fraud: float, fusion_fraud: float) -> str:
    """
    Build a simple human-readable explanation based on the three fraud scores.
    """
    parts = []

    # Text model interpretation
    if text_fraud > 0.7:
        parts.append(f"Text model strongly indicates FRAUD (p={text_fraud:.2f}).")
    elif text_fraud < 0.3:
        parts.append(f"Text model suggests REAL (p_fraud={text_fraud:.2f}).")
    else:
        parts.append(f"Text model is uncertain (p_fraud={text_fraud:.2f}).")

    # Image model interpretation
    if image_fraud > 0.7:
        parts.append(f"Image model strongly indicates FRAUD (p={image_fraud:.2f}).")
    elif image_fraud < 0.3:
        parts.append(f"Image model suggests REAL (p_fraud={image_fraud:.2f}).")
    else:
        parts.append(f"Image model is uncertain (p_fraud={image_fraud:.2f}).")

    # Fusion interpretation
    if fusion_fraud > 0.7:
        parts.append(f"Fusion classifier predicts FRAUD with high confidence (p={fusion_fraud:.2f}).")
    elif fusion_fraud < 0.3:
        parts.append(f"Fusion classifier predicts REAL (p_fraud={fusion_fraud:.2f}).")
    else:
        parts.append(f"Fusion classifier is uncertain (p_fraud={fusion_fraud:.2f}).")

    return " ".join(parts)


# -------------------------------------------------------------
# Main multimodal prediction function
# -------------------------------------------------------------

def predict_invoice(text: str, image_path: str) -> Dict[str, Any]:
    """
    Run multimodal inference for a single invoice (text + image).
    """

    # ----------------------
    # 1. TEXT prediction
    # ----------------------
    text_output = predict_text(text)
    text_fraud = float(text_output["proba_fraud"])
    text_real = float(text_output["proba_real"])

    # ----------------------
    # 2. IMAGE prediction
    # ----------------------
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image_output = predict_image(image_path)
    image_fraud = float(image_output["proba_fraud"])
    image_real = float(image_output["proba_real"])

    # ----------------------
    # 3. FUSION prediction
    # ----------------------
    fusion_clf = joblib.load(FUSION_MODEL_PATH)

    # Feature vector must match what we used in training:
    # [text_real, text_fraud, image_real, image_fraud, text_margin, image_margin]
    x = np.array([
        text_real,
        text_fraud,
        image_real,
        image_fraud,
        text_real - text_fraud,
        image_real - image_fraud,
    ]).reshape(1, -1)

    fusion_proba = fusion_clf.predict_proba(x)[0]
    fusion_fraud = float(fusion_proba[1])
    fusion_real = float(fusion_proba[0])

    # ----------------------
    # 4. Final decision
    # ----------------------
    final_label = "FRAUD" if fusion_fraud > 0.5 else "REAL"

    # ----------------------
    # 5. Explanation
    # ----------------------
    explanation = build_explanation(
        text_fraud=text_fraud,
        image_fraud=image_fraud,
        fusion_fraud=fusion_fraud,
    )

    # ----------------------
    # 6. Structured output
    # ----------------------
    return {
        "text": text,
        "image_path": image_path,

        "text_model": text_output,
        "image_model": image_output,

        "fusion_proba_real": fusion_real,
        "fusion_proba_fraud": fusion_fraud,

        "final_label": final_label,
        "expert_explanation": explanation,
    }


# -------------------------------------------------------------
# Simple CLI example
# -------------------------------------------------------------

if __name__ == "__main__":
    # TODO: replace with a real invoice text and image path from your dataset
    sample_text = "This is a sample invoice issued by ACME Corp. for 3 EUR."
    sample_image = "data/raw/sroie/train/img/X00016469612.jpg" 

    result = predict_invoice(sample_text, sample_image)
    print(json.dumps(result, indent=4))
