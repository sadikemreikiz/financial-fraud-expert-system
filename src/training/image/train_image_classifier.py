import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from src.config import PROCESSED_DIR, MODEL_IMAGE_DIR


# ===========================
# Dataset Class
# ===========================

class SROIEImageDataset(Dataset):
    def __init__(self, jsonl_file: Path, transform=None):
        self.items = []
        self.transform = transform

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.items.append(row)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        img_path = Path(row["image_path"])
        label = row["label"]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# ===========================
# Model Training Function
# ===========================

def train_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
    epochs: int = 3,
):

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # REAL vs FRAUD

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")

        # eval
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")

    return model


# ===========================
# MAIN
# ===========================

def main():
    jsonl_path = PROCESSED_DIR / "sroie_image_dataset.jsonl"

    print(f"Loading dataset: {jsonl_path}")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # pretrained ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = SROIEImageDataset(jsonl_path, transform=transform)

    # small train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = train_model(train_loader, test_loader, device=device, epochs=3)

    # save model
    MODEL_IMAGE_DIR.mkdir(exist_ok=True, parents=True)
    out_path = MODEL_IMAGE_DIR / "sroie_image_classifier.pt"
    torch.save(model.state_dict(), out_path)

    print(f"\nSaved image classifier to:\n{out_path}")


if __name__ == "__main__":
    main()
