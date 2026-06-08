# 🔬 Breast Cancer Histopathological Image Classifier

A deep learning pipeline for binary classification of breast cancer histopathology images from the **BreaKHis v1** dataset using **ResNet50 transfer learning** in PyTorch. Achieves **99.07% test accuracy** with robust class imbalance handling and is deployable as an interactive **Streamlit web app** with Grad-CAM visualisation.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Class Imbalance Handling](#class-imbalance-handling)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Grad-CAM Visualisation](#grad-cam-visualisation)
- [Future Work](#future-work)
- [References](#references)

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate detection from histopathological images is critical for improving patient outcomes. This project builds an end-to-end binary classifier that categorises microscopic tissue images as either **Benign** or **Malignant**.

| Attribute        | Detail |
|------------------|--------|
| Task             | Binary Classification (Benign vs Malignant) |
| Dataset          | BreaKHis v1 |
| Model            | ResNet50 (pretrained on ImageNet, fine-tuned) |
| Framework        | PyTorch |
| Magnifications   | 40X, 100X, 200X, 400X (all combined) |
| Test Accuracy    | **99.07%** |
| Deployment       | Streamlit Web App |

---

## 🗂 Dataset

The **Breast Cancer Histopathological Database (BreaKHis)** was collected at the P&D Laboratory in Paraná, Brazil.

- **Total images:** 7,909 PNG images (700×460px, RGB)
- **Benign:** 2,480 images
- **Malignant:** 5,429 images
- **Magnifications:** 40X, 100X, 200X, 400X
- **Patients:** 82

### Tumour Types

| Category  | Type                  | Code |
|-----------|-----------------------|------|
| Benign    | Adenosis              | A    |
| Benign    | Fibroadenoma          | F    |
| Benign    | Phyllodes Tumour      | PT   |
| Benign    | Tubular Adenoma       | TA   |
| Malignant | Ductal Carcinoma      | DC   |
| Malignant | Lobular Carcinoma     | LC   |
| Malignant | Mucinous Carcinoma    | MC   |
| Malignant | Papillary Carcinoma   | PC   |

### Folder Structure

```
BreaKHis_v1/
└── histology_slides/
    └── breast/
        ├── benign/
        │   └── SOB/
        │       ├── adenosis/
        │       ├── fibroadenoma/
        │       ├── phyllodes_tumor/
        │       └── tubular_adenoma/
        └── malignant/
            └── SOB/
                ├── ductal_carcinoma/
                ├── lobular_carcinoma/
                ├── mucinous_carcinoma/
                └── papillary_carcinoma/
```

### Filename Convention

```
SOB_B_TA-14-4659-40-001.png
 │   │  │   │    │   └── Sequence number
 │   │  │   │    └────── Magnification (40X)
 │   │  │   └─────────── Slide ID
 │   │  └─────────────── Tumour type (TA = Tubular Adenoma)
 │   └────────────────── Class (B = Benign, M = Malignant)
 └────────────────────── Biopsy procedure (SOB)
```

### Download

Request access at: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

Or download via Kaggle: https://www.kaggle.com/datasets/ambarish/breakhis

---

## 📁 Project Structure

```
breast_cancer_classifier/
├── breakhis_classifier.py        # CNN trained from scratch (baseline)
├── breakhis_resnet50.py          # ResNet50 transfer learning
├── breakhis_resnet50_balanced.py # ResNet50 + class imbalance handling ✅ best
├── app.py                        # Streamlit web app
├── requirements.txt              # Python dependencies
├── best_model_balanced.pth       # Saved model weights (after training)
├── training_history_balanced.png # Loss & accuracy curves
├── confusion_matrix_balanced.png # Test set confusion matrix
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU

### Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-classifier.git
cd breast-cancer-classifier
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch
torchvision
streamlit
scikit-learn
Pillow
numpy
matplotlib
seaborn
gdown
```

---

## 🚀 Usage

### 1. Training

Update `DATA_ROOT` in the script to point to your dataset:

```python
DATA_ROOT = '/path/to/BreaKHis_v1/histology_slides/breast'
```

**Run the best model (ResNet50 + balanced):**

```bash
python breakhis_resnet50_balanced.py
```

**Run in Google Colab:**

```python
# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Then run the script
exec(open('breakhis_resnet50_balanced.py').read())
```

### 2. Inference (single image)

```python
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)
model.load_state_dict(torch.load("best_model_balanced.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open("your_image.png").convert("RGB")
tensor = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    output = model(tensor)
    prob = torch.sigmoid(output).item()
    label = "Malignant" if prob > 0.5 else "Benign"
    confidence = prob if prob > 0.5 else 1 - prob

print(f"Prediction: {label} ({confidence * 100:.2f}% confidence)")
```

### 3. Streamlit App

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

### ResNet50 Backbone

ResNet50 pretrained on ImageNet (1.2M images, 1000 classes) is used as a feature extractor. The final fully connected layer is replaced with a custom binary classification head:

```
ResNet50 Backbone (frozen in Phase 1, unfrozen in Phase 2)
    └── AdaptiveAvgPool2d → Flatten → [2048]
         └── Linear(2048 → 256)
              └── ReLU
                   └── Dropout(0.5)
                        └── Linear(256 → 1)   ← sigmoid → binary prediction
```

**Total trainable parameters (Phase 2):** ~23.5M

---

## 🏋️ Training Strategy

Training is split into two phases to prevent catastrophic forgetting of ImageNet features:

### Phase 1 — Feature Extraction (Epochs 1–5)

- Backbone is **frozen** (no gradient updates)
- Only the new classifier head is trained
- Learning rate: `1e-3`
- Allows the head to adapt to histopathology features before touching the backbone

### Phase 2 — Fine-Tuning (Epochs 6–20)

- Entire network is **unfrozen**
- Differential learning rates applied:
  - Classifier head: `lr = 1e-3`
  - Backbone: `lr = 1e-4` (10× smaller to preserve pretrained features)
- Cosine Annealing LR scheduler

### Hyperparameters

| Parameter        | Value          |
|------------------|----------------|
| Image Size       | 224 × 224      |
| Batch Size       | 32             |
| Total Epochs     | 20             |
| Optimiser        | Adam           |
| Weight Decay     | 1e-4           |
| LR (head)        | 1e-3           |
| LR (backbone)    | 1e-4           |
| LR Schedule      | CosineAnnealing|

### Data Augmentation (Training only)

```python
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
```

### Data Split

Stratified split to preserve class ratios across all sets:

| Split | Size  | Percentage |
|-------|-------|------------|
| Train | 5,538 | 70%        |
| Val   | 1,184 | 15%        |
| Test  | 1,187 | 15%        |

---

## ⚖️ Class Imbalance Handling

The dataset has a ~2:1 imbalance (malignant:benign). Two complementary strategies are applied:

### Strategy 1 — Weighted Loss

`BCEWithLogitsLoss` with `pos_weight = n_benign / n_malignant ≈ 0.457`

This mathematically scales the loss to penalise missed benign predictions proportionally more, correcting for the imbalance at the objective level.

```python
pos_weight = torch.tensor([n_benign / n_malignant])
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Strategy 2 — Weighted Random Sampler

Each training batch is constructed with ~50/50 class balance by over-sampling the minority (benign) class:

```python
class_weights  = {0: 1.0/n_benign, 1: 1.0/n_malignant}
sample_weights = [class_weights[label] for _, label in train_samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
```

**Impact:** Accuracy improved from **97.81% → 99.07%** and benign precision improved from 0.96 → 0.99.

---

## 📊 Results

### Comparison: Without vs With Balancing

| Metric             | Without Balancing | With Balancing |
|--------------------|-------------------|----------------|
| Test Accuracy      | 97.81%            | **99.07%**     |
| Benign Precision   | 0.96              | **0.99**       |
| Benign Recall      | 0.97              | **0.98**       |
| Malignant Precision| 0.99              | **0.99**       |
| Malignant Recall   | 0.98              | **0.99**       |
| Macro F1           | 0.97              | **0.99**       |

### Final Classification Report (Balanced Model)

```
              precision    recall  f1-score   support

      Benign       0.99      0.98      0.99       372
   Malignant       0.99      0.99      0.99       815

    accuracy                           0.99      1187
   macro avg       0.99      0.99      0.99      1187
weighted avg       0.99      0.99      0.99      1187
```

### Training Progress

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1     | 70.31%    | 84.04%  |
| 5     | 86.04%    | 87.67%  |
| 10    | 97.72%    | 97.30%  |
| 15    | 98.88%    | 98.48%  |
| 20    | 99.73%    | 98.06%  |
| Best  | —         | **98.73%** |

---

## 🌐 Streamlit App

The app provides an interactive interface to classify uploaded histology images.

### Features

- Upload `.png` or `.jpg` histology images
- Real-time Benign / Malignant prediction with confidence score
- **Grad-CAM heatmap** overlay showing which tissue regions influenced the decision
- Auto-detects magnification level from the filename
- Model performance summary in the sidebar

### Running Locally

```bash
streamlit run app.py
```

### Deploying to Streamlit Cloud

1. Push your project folder to GitHub (include `app.py`, `requirements.txt`, and `best_model_balanced.pth`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and click **Deploy**

> ⚠️ **Note:** Streamlit Cloud free tier has a 1GB memory limit. ResNet50 fits within this but may be slow on CPU.

---

## 🔍 Grad-CAM Visualisation

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions of a tissue image the model focuses on when making a prediction. This is critical for clinical interpretability — a pathologist can verify whether the model is attending to biologically meaningful regions.

The heatmap is generated from the last convolutional layer (`model.layer4[-1]`), upsampled to the input image size, and overlaid in a jet colourmap.

```
Red regions   → high activation (most influential for the prediction)
Blue regions  → low activation
```

---

## 🔭 Future Work

- **Per-magnification analysis** — evaluate accuracy separately at 40X, 100X, 200X, 400X
- **Multi-class classification** — extend to all 8 tumour subtypes
- **Ensemble models** — combine predictions across magnification levels
- **Longer training** — model was still improving at epoch 20; 30+ epochs may push accuracy further
- **Lighter deployment model** — swap ResNet50 for EfficientNetB0 for faster inference in the Streamlit app
- **Cross-patient validation** — split by patient ID rather than image to avoid data leakage

---

## 📚 References

1. Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016). *A Dataset for Breast Cancer Histopathological Image Classification*. IEEE Transactions on Biomedical Engineering, 63(7), 1455–1462.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.

3. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.

---

## 📄 License

This project is for non-commercial research use only, in accordance with the BreaKHis dataset license (Creative Commons Attribution 4.0 International).
