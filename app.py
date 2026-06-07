"""
BreaKHis Breast Cancer Classifier — Streamlit App
Upload a histology image → get Benign / Malignant prediction
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gdown
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Histology Classifier",
    page_icon="🔬",
    layout="centered"
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "best_model_balanced.pth"
IMG_SIZE    = (224, 224)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Benign", "Malignant"]
CLASS_COLORS = {"Benign": "#2ecc71", "Malignant": "#e74c3c"}

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output.backward()
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations  = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap /= heatmap.max()
        return heatmap


def overlay_gradcam(image_pil, heatmap):
    img_array = np.array(image_pil.resize(IMG_SIZE)) / 255.0
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, Image.BILINEAR)
    ) / 255.0
    colormap   = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    overlay    = 0.6 * img_array + 0.4 * heatmap_colored
    overlay    = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🔬 Breast Cancer Histology Classifier")
st.markdown("""
Upload a histopathology image from the **BreaKHis** dataset (or any compatible image)
and the model will classify it as **Benign** or **Malignant**.

> Model: ResNet50 fine-tuned on BreaKHis | Test Accuracy: **99.07%**
""")

st.divider()

# Check model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found. Please upload it below.")
    uploaded_model = st.file_uploader("Upload model weights (.pth)", type=["pth"])
    if uploaded_model:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.read())
        st.success("Model uploaded! Please refresh the page.")
        st.stop()
    else:
        st.stop()

# Load model
with st.spinner("Loading model..."):
    model = load_model()
st.success(f"Model loaded — running on **{str(DEVICE).upper()}**")

st.divider()

# Image upload
uploaded_file = st.file_uploader(
    "Upload a histology image (.png or .jpg)",
    type=["png", "jpg", "jpeg"]
)

show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    # Predict
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    if show_gradcam:
        input_tensor.requires_grad = True
        gradcam = GradCAM(model, model.layer4[-1])

    with torch.no_grad() if not show_gradcam else torch.enable_grad():
        if show_gradcam:
            output = model(input_tensor)
        else:
            output = model(input_tensor)

    prob       = torch.sigmoid(output).item()
    pred_idx   = 1 if prob > 0.5 else 0
    pred_label = CLASS_NAMES[pred_idx]
    confidence = prob if pred_idx == 1 else 1 - prob
    color      = CLASS_COLORS[pred_label]

    with col2:
        if show_gradcam:
            st.subheader("Grad-CAM Heatmap")
            try:
                heatmap     = gradcam.generate(input_tensor)
                overlay_img = overlay_gradcam(image, heatmap)
                st.image(overlay_img, use_column_width=True)
            except Exception:
                st.info("Grad-CAM unavailable for this input.")
        else:
            st.subheader("Preview")
            st.image(image, use_column_width=True)

    st.divider()
    st.subheader("Prediction")

    # Result badge
    st.markdown(
        f"<h2 style='color:{color}; text-align:center;'>{pred_label}</h2>",
        unsafe_allow_html=True
    )

    # Confidence bar
    st.markdown(f"**Confidence: {confidence * 100:.2f}%**")
    st.progress(confidence)

    # Probability breakdown
    st.markdown("**Class Probabilities**")
    bcol1, bcol2 = st.columns(2)
    bcol1.metric("Benign",    f"{(1 - prob) * 100:.2f}%")
    bcol2.metric("Malignant", f"{prob * 100:.2f}%")

    st.divider()

    # Magnification hint from filename
    filename = uploaded_file.name
    mag = None
    for m in ["40X", "100X", "200X", "400X"]:
        if m in filename:
            mag = m
            break
    if mag:
        st.info(f"Detected magnification from filename: **{mag}**")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Dataset:** BreaKHis v1  
    **Model:** ResNet50 (fine-tuned)  
    **Training:** 7,909 images  
    **Test Accuracy:** 99.07%  
    **Classes:** Benign / Malignant  
    **Magnifications:** 40X, 100X, 200X, 400X  
    """)
    st.divider()
    st.header("📊 Model Performance")
    st.markdown("""
    | Class | Precision | Recall | F1 |
    |---|---|---|---|
    | Benign | 0.99 | 0.98 | 0.99 |
    | Malignant | 0.99 | 0.99 | 0.99 |
    """)
    st.divider()
    st.markdown("Built with PyTorch + Streamlit")
