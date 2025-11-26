# Financial Fraud Expert System

This is my master's thesis project: an AI-based expert system for detecting fraud in financial documents using NLP, Computer Vision, and Explainable AI.

Capabilities include:
- OCR-based text extraction
- Text-only fraud detection models (baseline)
- Image-based fraud detection models (CNN/ResNet)
- Multimodal fusion (BERT + ResNet)
- Explainable AI (SHAP, LIME, Grad-CAM)
- Expert rule layer (heuristic decision logic)



# Financial Fraud Expert System

This project is my master’s thesis work: an AI-based system for detecting fraud in financial documents (invoices).  
It includes text classification, image classification, and a small fusion model that combines both signals.

### ✔ Current Features
- Text fraud detection (Logistic Regression)
- Image fraud detection (ResNet18)
- Fusion model (combines text + image probabilities)
- Final prediction script with simple explanation

### ✔ Run Final Prediction

python -m src.predict.predict_sroie_fusion

### ✔ Project Structure

src/
├── training/ # text, image, fusion training scripts
├── predict/ # final multimodal prediction
├── utils/ # config and helpers


### ✔ Notes
- Dataset and model files are ignored (not included in GitHub).
- The project will be expanded and improved step-by-step.

---
Author: **Sadik Emre Ikiz**

