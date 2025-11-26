import cv2
import numpy as np
from pathlib import Path
import random

from src.config import RAW_DIR, PROCESSED_DIR

INPUT_DIR = RAW_DIR / "sroie" / "train" / "img"
OUTPUT_DIR = PROCESSED_DIR / "sroie_synthetic_fraud_images"

def blur_logo(img):
    h, w = img.shape[:2]
    # Logo region (top-left corner)
    x1, y1, x2, y2 = 0, 0, int(w * 0.25), int(h * 0.15)
    logo = img[y1:y2, x1:x2]
    logo = cv2.GaussianBlur(logo, (31, 31), 0)
    img[y1:y2, x1:x2] = logo
    return img

def blur_stamp(img):
    h, w = img.shape[:2]
    # Approx bottom-right corner region
    x1, y1, x2, y2 = int(w * 0.7), int(h * 0.7), w, h
    stamp = img[y1:y2, x1:x2]
    stamp = cv2.GaussianBlur(stamp, (41, 41), 0)
    img[y1:y2, x1:x2] = stamp
    return img

def warp_text(img):
    h, w = img.shape[:2]
    src = np.float32([[0,0], [w,0], [0,h]])
    dst = np.float32([[0,0], [w*0.9, h*0.05], [w*0.05, h*0.95]])
    matrix = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(img, matrix, (w, h))
    return warped

def generate_fraud_versions(img):
    ops = [blur_logo, blur_stamp, warp_text]
    selected = random.choice(ops)
    return selected(img.copy())

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = list(INPUT_DIR.glob("*.jpg"))

    for i, img_path in enumerate(images[:50]):  # generate fraud for 50 images
        img = cv2.imread(str(img_path))
        fraud_img = generate_fraud_versions(img)
        out_path = OUTPUT_DIR / f"fraud_{img_path.name}"
        cv2.imwrite(str(out_path), fraud_img)
        print(f"[+] Saved fraud version: {out_path}")

if __name__ == "__main__":
    main()
