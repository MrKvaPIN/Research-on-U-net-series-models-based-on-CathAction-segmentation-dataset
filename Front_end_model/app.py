import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# =========================
# Paths / constants
# =========================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

NUM_CLASSES = 3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Background(0), class1, class2
CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([255, 80, 80], dtype=np.uint8),
    2: np.array([80, 255, 120], dtype=np.uint8),
}


# =========================
# Model defs (from training scripts)
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpAttBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 16))
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.att(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch, dropout=0.0)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, dropout=0.0)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, dropout=0.1)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, dropout=0.1)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(base_ch * 8, base_ch * 16, dropout=0.2)

        self.dec4 = UpAttBlock(base_ch * 16, base_ch * 8, base_ch * 8, dropout=0.1)
        self.dec3 = UpAttBlock(base_ch * 8, base_ch * 4, base_ch * 4, dropout=0.1)
        self.dec2 = UpAttBlock(base_ch * 4, base_ch * 2, base_ch * 2, dropout=0.0)
        self.dec1 = UpAttBlock(base_ch * 2, base_ch, base_ch, dropout=0.0)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        c = self.center(self.pool4(e4))
        d4 = self.dec4(c, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)


class RecurrentBlock(nn.Module):
    def __init__(self, out_ch, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = x
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t=2):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.rcnn = nn.Sequential(RecurrentBlock(out_ch, t=t), RecurrentBlock(out_ch, t=t))

    def forward(self, x):
        x = self.conv_1x1(x)
        x1 = self.rcnn(x)
        return x + x1


class UpRRCNNAttBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 16))
        self.rrcnn = RRCNNBlock(out_ch + skip_ch, out_ch, t=t)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.att(x, skip)
        x = torch.cat((skip, x), dim=1)
        x = self.rrcnn(x)
        return x


class AttentionR2UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_ch=32, t=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = RRCNNBlock(in_channels, base_ch, t=t)
        self.enc2 = RRCNNBlock(base_ch, base_ch * 2, t=t)
        self.enc3 = RRCNNBlock(base_ch * 2, base_ch * 4, t=t)
        self.enc4 = RRCNNBlock(base_ch * 4, base_ch * 8, t=t)
        self.center = RRCNNBlock(base_ch * 8, base_ch * 16, t=t)

        self.dec4 = UpRRCNNAttBlock(base_ch * 16, base_ch * 8, base_ch * 8, t=t)
        self.dec3 = UpRRCNNAttBlock(base_ch * 8, base_ch * 4, base_ch * 4, t=t)
        self.dec2 = UpRRCNNAttBlock(base_ch * 4, base_ch * 2, base_ch * 2, t=t)
        self.dec1 = UpRRCNNAttBlock(base_ch * 2, base_ch, base_ch, t=t)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(c, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)


# =========================
# Utility
# =========================
def list_checkpoints():
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted([p.name for p in CHECKPOINT_DIR.glob("*.pt")])


def infer_model_type(filename, state_dict):
    name = filename.lower()
    if "attention_r2" in name or "r2unet" in name:
        return "attention_r2unet"
    if "attention" in name and "unet" in name:
        return "attention_unet"
    if "unet" in name:
        return "smp_unet"

    keys = list(state_dict.keys())
    if any(k.startswith("encoder.") for k in keys):
        return "smp_unet"
    if any("rrcnn" in k for k in keys):
        return "attention_r2unet"
    return "attention_unet"


def build_model(model_type):
    if model_type == "smp_unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        )
        return model, 768
    if model_type == "attention_unet":
        return AttentionUNet(in_channels=3, num_classes=NUM_CLASSES, base_ch=32), 768
    return AttentionR2UNet(in_channels=3, num_classes=NUM_CLASSES, base_ch=32, t=2), 512


@st.cache_resource(show_spinner=False)
def load_model(ckpt_name):
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(ckpt_path, map_location=device)
    model_type = infer_model_type(ckpt_name, state_dict)

    model, input_res = build_model(model_type)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model, input_res, model_type, str(device)


def preprocess_image(image_rgb, input_res):
    img_resized = cv2.resize(image_rgb, (input_res, input_res), interpolation=cv2.INTER_LINEAR)
    x = img_resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0)
    return x


def mask_to_color(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, clr in CLASS_COLORS.items():
        color[mask == cls] = clr
    return color


def blend_overlay(image_rgb, color_mask, alpha=0.45):
    overlay = image_rgb.copy()
    fg = np.any(color_mask != 0, axis=-1)
    overlay[fg] = ((1 - alpha) * image_rgb[fg] + alpha * color_mask[fg]).astype(np.uint8)
    return overlay


def run_inference(model, image_rgb, input_res, device):
    x = preprocess_image(image_rgb, input_res).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred = cv2.resize(pred, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred


# =========================
# UI
# =========================
st.set_page_config(page_title="Catheter Segmentation Demo", layout="wide")
st.title("Catheter Segmentation Model Viewer")
st.caption("Select a checkpoint, upload an image, and view the segmentation overlay.")

ckpts = list_checkpoints()
if not ckpts:
    st.error(f"No checkpoint files found. Please check the directory: {CHECKPOINT_DIR}")
    st.stop()

col_left, col_right = st.columns([1, 2])

with col_left:
    ckpt_name = st.selectbox("Select Model (.pt)", ckpts, index=0)
    alpha = st.slider("Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    uploaded = st.file_uploader("Upload an Image for Inference", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    model, input_res, model_type, device_name = load_model(ckpt_name)
    st.info(f"Model: {model_type} | Input: {input_res}x{input_res} | Device: {device_name}")
    stats_placeholder = st.empty()

with col_right:
    if uploaded is None:
        st.warning("Please upload an image from the left panel first.")
    else:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Failed to read the image. Please try another file.")
            st.stop()

        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred_mask = run_inference(model, image_rgb, input_res, torch.device(device_name))

        color_mask = mask_to_color(pred_mask)
        overlay = blend_overlay(image_rgb, color_mask, alpha=alpha)

        c1, c2, c3 = st.columns(3)
        c1.image(image_rgb, caption="Original Image", use_container_width=True)
        c2.image(color_mask, caption="Predicted Mask (Color)", use_container_width=True)
        c3.image(overlay, caption="Overlay Result", use_container_width=True)

        cls_vals, cls_counts = np.unique(pred_mask, return_counts=True)
        total = pred_mask.size
        stats = []
        for cls, cnt in zip(cls_vals.tolist(), cls_counts.tolist()):
            ratio = 100.0 * cnt / total
            stats.append({"class": int(cls), "pixels": int(cnt), "ratio(%)": round(ratio, 3)})

        with stats_placeholder.container():
            st.subheader("Prediction Distribution")
            st.table(stats)
