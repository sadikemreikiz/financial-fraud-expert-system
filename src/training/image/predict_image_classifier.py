from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from src.config import MODEL_IMAGE_DIR


# Path to the trained model
MODEL_PATH = MODEL_IMAGE_DIR / "sroie_image_classifier.pt"

# Class names: must match training
CLASS_NAMES = ["REAL", "FRAUD"]


def build_model() -> torch.nn.Module:
    """
    Rebuild the same ResNet18 architecture used during training
    and load the trained weights.
    """
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Change final layer to 2 classes (REAL / FRAUD)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    # Load trained weights
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


def get_transforms() -> transforms.Compose:
    """
    Preprocessing pipeline for a single image.
    This should roughly match what we used during training.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Simple normalization (ImageNet-like)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_image(image_path: Path) -> Dict:
    """
    Run the trained classifier on a single image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    preprocess = get_transforms()

    # Load image
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = CLASS_NAMES[pred_idx]

    return {
        "image_path": str(image_path),
        "pred_index": pred_idx,
        "pred_label": pred_label,
        "proba_real": float(probs[0]),
        "proba_fraud": float(probs[1]),
    }


def main():
    """
    Small manual demo – change test_image_path to try different receipts.
    """
    test_image_path = Path("data/raw/sroie/train/img/X00016469612.jpg")

    if not test_image_path.exists():
        raise FileNotFoundError(f"Test image not found at {test_image_path}")

    result = predict_image(test_image_path)

    print("\n=== IMAGE PREDICTION ===")
    print(f"Image   : {result['image_path']}")
    print(f"Label   : {result['pred_label']}  "
          f"(REAL={result['proba_real']:.3f}, FRAUD={result['proba_fraud']:.3f})")


if __name__ == "__main__":
    main()
