import os, glob, random
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# Training Configuration
# =============================================================================
ROOT_DIR        = r"D:\FYP\Data_set\segmentation"
TRAIN_IMG_DIR   = os.path.join(ROOT_DIR, "phantom_train_3", "images")
TRAIN_MSK_DIR   = os.path.join(ROOT_DIR, "phantom_train_3", "masks")

NUM_CLASSES     = 3
EPOCHS          = 30

TOTAL_SAMPLES   = 4743
TRAIN_SAMPLES   = 4043

TRAIN_RES       = 768
VAL_RES         = 768

BATCH_SIZE      = 2
NUM_WORKERS     = 2

CHECKPOINT_DIR  = "checkpoints"
BEST_MODEL_NAME = "best_attention_unet.pt"

SEED            = 42
EARLY_STOP_PATIENCE = 8


# =============================================================================
# Fix random seed for reproducibility
# =============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset: images read by cv2; masks read by np.load
# =============================================================================
class CathDatasetNPY(Dataset):
    """
    Custom PyTorch Dataset for loading catheter/phantom X-ray images and their
    corresponding segmentation masks for multi-class semantic segmentation.
    """

    def __init__(self, img_dir, mask_dir, stems, transform=None):
        """
        Initialize the dataset.

        Args:
            img_dir (str): Directory containing input images.
            mask_dir (str): Directory containing mask .npy files.
            stems (list): List of filename stems to include in this split.
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.stems = stems
        self.transform = transform

    def __len__(self):
        """Return the number of samples in this dataset split."""
        return len(self.stems)

    def __getitem__(self, idx):
        s = self.stems[idx]

        img_path = glob.glob(os.path.join(self.img_dir, s + ".*"))[0]
        msk_path = os.path.join(self.mask_dir, s + ".npy")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.load(msk_path).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask


# =============================================================================
# Attention U-Net
# =============================================================================

class ConvBlock(nn.Module):
    """
    Standard U-Net Convolutional Block.

    Each ConvBlock consists of two consecutive 3x3 convolution layers, each followed
    by Batch Normalization and ReLU activation. This is the fundamental encoding unit
    in the U-Net architecture. An optional Dropout2d layer can be added for regularization.

    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> [Dropout2d]
    """

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            # First 3x3 conv: spatial feature extraction.
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            # Second 3x3 conv: further refine features.
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            # Optional dropout for regularization in deeper layers.
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for U-Net skip connections.

    Attention Gates suppress irrelevant background features and highlight salient
    regions in the skip connection, allowing the decoder to focus on the most
    relevant spatial locations for accurate segmentation.

    The gate takes two inputs:
    - g (gate signal): Decoder features from the deeper layer (coarser, more semantic).
    - x (skip signal): Encoder features from the corresponding encoder layer (finer, more spatial detail).

    Both are first projected to an intermediate channel size, summed, activated,
    then projected to a single-channel attention map (0~1), which is multiplied
    element-wise with the skip signal x.

    Mathematically:
      Attention = sigmoid(W_g(g) + W_x(x))  -> [0,1] attention map
      Output = x * Attention  (element-wise multiplication)
    """

    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        # Project gate signal from g_ch to inter_ch.
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        # Project skip signal from x_ch to inter_ch.
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        # Final 1x1 conv to produce a single-channel attention map (scalar per pixel).
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # ReLU activation before combining gate and skip signals.
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine: sum and activate. This is where attention scores are computed.
        psi = self.relu(g1 + x1)

        # Project to single channel and apply sigmoid to get [0,1] attention weights.
        psi = self.psi(psi)

        # Element-wise multiply: attenuate or amplify each spatial location in x.
        return x * psi


class UpAttBlock(nn.Module):
    """
    Decoder Block with Upsampling and Attention Gate (used in Attention U-Net).

    This block performs two operations in sequence:
    1. Upsamples the decoder feature map from (in_ch) to (out_ch) using ConvTranspose2d.
    2. Applies the Attention Gate to filter the encoder skip connection.
    3. Concatenates the upsampled features with the filtered skip connection.
    4. Passes the concatenation through a ConvBlock to fuse features.

    This is the core building block of the Attention U-Net decoder.
    """

    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        # 2x upsampling via transposed convolution.
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # Attention gate: filters the skip connection using decoder features as gate signal.
        self.att = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 16))
        # Convolutional block to fuse upsampled features and filtered skip connection.
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle potential 1-pixel resolution mismatch due to pooling/upsampling rounding.
        # This can happen when image size is not perfectly divisible by 2^n.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # Step 2: Apply attention gate to filter the skip connection.
        # The gate signal is x (upsampled decoder features), the skip signal is skip (encoder features).
        skip = self.att(x, skip)

        # Step 3: Concatenate along the channel dimension.
        x = torch.cat([skip, x], dim=1)

        # Step 4: Fuse via ConvBlock.
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net: U-Net with Attention Gates in all skip connections.

    The Attention U-Net is identical to a standard U-Net in structure (encoder-decoder
    with skip connections), except that each skip connection passes through an
    Attention Gate before concatenation. The attention mechanism adaptivelyweights
    the encoder features, suppressing irrelevant background and focusing on the target
    organ/structure during decoding.

    Architecture:
      Encoder: 4 levels of ConvBlock + MaxPool, channel progression: 32 -> 64 -> 128 -> 256 -> 512.
      Bottleneck: ConvBlock at 512 channels.
      Decoder: 4 levels of UpAttBlock, channel progression reversed.
      Output: 1x1 conv -> NUM_CLASSES channels.

    Key difference from U-Net:
      - Standard U-Net: skip = encoder features directly concatenated with upsampled decoder features.
      - Attention U-Net: skip = AttentionGate(upsampled_decoder, encoder_features) then concatenated.
    """

    def __init__(self, in_channels=3, num_classes=3, base_ch=32):
        super().__init__()

        # ==================== Encoder ====================
        # Level 1: 3 -> 32 channels, no dropout.
        self.enc1 = ConvBlock(in_channels, base_ch, dropout=0.0)
        self.pool1 = nn.MaxPool2d(2)

        # Level 2: 32 -> 64 channels, no dropout.
        self.enc2 = ConvBlock(base_ch, base_ch * 2, dropout=0.0)
        self.pool2 = nn.MaxPool2d(2)

        # Level 3: 64 -> 128 channels, dropout 0.1.
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, dropout=0.1)
        self.pool3 = nn.MaxPool2d(2)

        # Level 4: 128 -> 256 channels, dropout 0.1.
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, dropout=0.1)
        self.pool4 = nn.MaxPool2d(2)

        # ==================== Bottleneck ====================
        # Deepest features: 256 -> 512 channels, dropout 0.2.
        self.center = ConvBlock(base_ch * 8, base_ch * 16, dropout=0.2)

        # ==================== Decoder ====================
        # Each decoder block upsamples, applies attention to skip connection, concatenates, and fuses.
        # Level 4: 512 -> 256 channels (decoder) + skip from enc4 (256ch) -> output 256ch.
        self.dec4 = UpAttBlock(base_ch * 16, base_ch * 8, base_ch * 8, dropout=0.1)
        # Level 3: 256 -> 128 channels + skip from enc3 (128ch) -> output 128ch.
        self.dec3 = UpAttBlock(base_ch * 8, base_ch * 4, base_ch * 4, dropout=0.1)
        # Level 2: 128 -> 64 channels + skip from enc2 (64ch) -> output 64ch.
        self.dec2 = UpAttBlock(base_ch * 4, base_ch * 2, base_ch * 2, dropout=0.0)
        # Level 1: 64 -> 32 channels + skip from enc1 (32ch) -> output 32ch.
        self.dec1 = UpAttBlock(base_ch * 2, base_ch, base_ch, dropout=0.0)

        # Final 1x1 convolution to map from base_ch to num_classes.
        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        # Bottleneck: no skip connection.
        c  = self.center(self.pool4(e4))

        d4 = self.dec4(c, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.out_conv(d1)
        return out


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def mean_dice(pred, target, num_classes=NUM_CLASSES, eps=1e-7):
    dices = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        if denom == 0:
            continue
        dices.append((2 * inter + eps) / (denom + eps))
    return float(np.mean(dices)) if dices else 0.0


@torch.no_grad()
def mean_iou(pred, target, num_classes=NUM_CLASSES, eps=1e-7):
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append((inter + eps) / (union + eps))
    return float(np.mean(ious)) if ious else 0.0


# =============================================================================
# Loss Functions
# =============================================================================

class DiceLossMultiClass(nn.Module):
    """
    Multi-class Dice Loss for semantic segmentation.

    Computes L = 1 - mean(Dice_c) across all classes.
    Effective for imbalanced segmentation tasks.
    """

    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = (probs * target_1h).sum(dim=dims)
        denom = probs.sum(dim=dims) + target_1h.sum(dim=dims)

        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class FocalLossMultiClass(nn.Module):
    """
    Focal Loss for multi-class semantic segmentation.

    L = ((1 - pt)^gamma) * CE, focuses on hard examples.
    gamma=2.0 by default.
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class HybridSegLoss(nn.Module):
    """
    Hybrid loss: 60% Dice Loss + 40% Focal Loss.

    Combines the overlap-optimizing Dice Loss with the class-imbalance-handling Focal Loss.
    Particularly effective for catheter/guidewire segmentation with small foreground regions.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.dice = DiceLossMultiClass(num_classes)
        self.focal = FocalLossMultiClass(gamma=2.0)

    def forward(self, logits, target):
        return 0.6 * self.dice(logits, target) + 0.4 * self.focal(logits, target)


# =============================================================================
# Training and Validation
# =============================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total = 0.0

    for x, y in tqdm(loader, leave=False):
        # Move data to device with non_blocking=True for async GPU transfer.
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward pass: reduces memory usage and speeds up training on CUDA.
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        # Backward pass with scaled gradients (required for mixed precision).
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, device, num_classes=NUM_CLASSES):
    model.eval()
    total = 0.0
    dices, ious = [], []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        total += loss.item() * x.size(0)

        pred = torch.argmax(logits, dim=1)
        dices.append(mean_dice(pred.cpu(), y.cpu(), num_classes))
        ious.append(mean_iou(pred.cpu(), y.cpu(), num_classes))

    return total / len(loader.dataset), float(np.mean(dices)), float(np.mean(ious))


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main entry point: set up data, model, optimizer, scheduler, and run training.
    """
    seed_everything(SEED)

    train_imgs = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*")))
    train_stems = [os.path.splitext(os.path.basename(p))[0] for p in train_imgs]

    random.seed(SEED)
    random.shuffle(train_stems)

    train_stems = train_stems[:TOTAL_SAMPLES]
    tr_stems, va_stems = train_stems[:TRAIN_SAMPLES], train_stems[TRAIN_SAMPLES:]
    print("Using stems -> train:", len(tr_stems), "val:", len(va_stems))

    # -------------------------------------------------------------------------
    # Training augmentation pipeline (online, applied per-batch).
    # -------------------------------------------------------------------------
    train_tf = A.Compose([
        A.Resize(TRAIN_RES, TRAIN_RES),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),          # Vertical flip (20% chance) — specific to this script.
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(0.0, 0.05),
            rotate=(-15, 15),
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),            # Gaussian noise augmentation (specific to this script).
        A.Normalize(),
        ToTensorV2()
    ])

    # Validation: deterministic preprocessing only.
    val_tf = A.Compose([
        A.Resize(VAL_RES, VAL_RES),
        A.Normalize(),
        ToTensorV2()
    ])

    train_ds = CathDatasetNPY(TRAIN_IMG_DIR, TRAIN_MSK_DIR, tr_stems, transform=train_tf)
    val_ds   = CathDatasetNPY(TRAIN_IMG_DIR, TRAIN_MSK_DIR, va_stems, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Model: Attention U-Net with base_ch=32.
    # -------------------------------------------------------------------------
    model = AttentionUNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_ch=32
    ).to(device)

    loss_fn = HybridSegLoss(NUM_CLASSES)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # ReduceLROnPlateau: reduce LR when val_dice plateaus (mode="max" because higher Dice is better).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best = -1.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        va_loss, va_dice, va_iou = validate(model, val_loader, loss_fn, device, NUM_CLASSES)

        scheduler.step(va_dice)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={va_loss:.4f} | "
            f"mDice={va_dice:.4f} | "
            f"mIoU={va_iou:.4f}"
        )

        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME))
            print(f"  Saved best model: mDice={best:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")


if __name__ == "__main__":
    main()
