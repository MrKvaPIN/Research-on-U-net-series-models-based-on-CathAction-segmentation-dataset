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
# Root directory of the dataset, containing subfolders for images and masks.
ROOT_DIR        = r"D:\FYP\Data_set\segmentation"
TRAIN_IMG_DIR   = os.path.join(ROOT_DIR, "phantom_train_3", "images")
TRAIN_MSK_DIR   = os.path.join(ROOT_DIR, "phantom_train_3", "masks")

NUM_CLASSES     = 3
EPOCHS          = 40

TOTAL_SAMPLES   = 4743
TRAIN_SAMPLES   = 4043

TRAIN_RES       = 768
VAL_RES         = 768

BATCH_SIZE      = 1
NUM_WORKERS     = 2

CHECKPOINT_DIR  = "checkpoints"
BEST_MODEL_NAME = "best_attention_r2unet.pt"

SEED            = 42


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

    Images are loaded via OpenCV (cv2) and masks are loaded from .npy files.
    Augmentations are applied via Albumentations when a transform pipeline is provided.
    """

    def __init__(self, img_dir, mask_dir, stems, transform=None):
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
# Attention R2U-Net
# =============================================================================

class RecurrentBlock(nn.Module):
    """
    Recurrent Convolutional Block used in R2U-Net.

    This block applies the same convolutional operation t times (default t=2) in a
    recurrent fashion. At each time step, the input is the sum of the original
    feature map and the output of the previous step. This allows the block to
    capture contextual information across deeper layers without increasing the
    number of parameters significantly.

    The recurrent structure effectively broadens the receptive field and enables
    the model to accumulate spatial information over multiple time steps.

    Equation: x_t = f(W * (x + x_{t-1}) + b),  for t = 1,...,T
    where x_0 = 0, f = ReLU, W = shared convolution weights.
    """

    def __init__(self, out_ch, t=2):
        super().__init__()
        self.t = t
        # Shared 3x3 convolution: same weights are reused at every recurrent step.
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = x
        for i in range(self.t):
            if i == 0:
                # First step: no residual connection from previous output (x_{t-1} = 0).
                x1 = self.conv(x)
            else:
                # Subsequent steps: add the previous output as residual.
                x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block (RRCNN Block).

    This is the core building block of R2U-Net. It consists of:
    1. A 1x1 convolution to project the input channel count to the output channel count.
    2. A recurrent block (RecurrentBlock) applied twice sequentially (t steps each).
    3. A residual/skip connection: output = RRCNN(x) + x.

    The residual connection facilitates gradient flow during backpropagation and
    stabilizes training of deep recurrent architectures.
    """

    def __init__(self, in_ch, out_ch, t=2):
        super().__init__()
        # 1x1 convolution: projects input to the representation space of the recurrent block.
        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        # Two stacked RecurrentBlocks for deeper recurrent feature extraction.
        self.rcnn = nn.Sequential(
            RecurrentBlock(out_ch, t=t),
            RecurrentBlock(out_ch, t=t)
        )

    def forward(self, x):
        x = self.conv_1x1(x)
        # Pass through the two recurrent blocks.
        x1 = self.rcnn(x)
        # Residual connection: RRCNN(x) = RCNN(x) + x.
        return x + x1


class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for skip connections.

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
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_ch)
        )

        # Project skip signal from x_ch to inter_ch.
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_ch)
        )

        # Final 1x1 conv to produce a single-channel attention map (scalar per pixel).
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
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


class UpRRCNNAttBlock(nn.Module):
    """
    Decoder Block combining Upsampling, Attention Gate, and RRCNN (used in Attention R2U-Net).

    This is the core decoder building block of Attention R2U-Net. It performs:
    1. Upsample the decoder feature map via ConvTranspose2d.
    2. Apply the Attention Gate to filter the encoder skip connection.
    3. Concatenate upsampled features with filtered skip connection.
    4. Pass through an RRCNN block (instead of a plain ConvBlock) to fuse features with
       recurrent residual connections, enabling deeper contextual modeling at each decoder level.

    This combines the benefits of both R2U-Net (recurrent residual modeling)
    and Attention U-Net (gated skip connections).
    """

    def __init__(self, in_ch, skip_ch, out_ch, t=2):
        super().__init__()
        # 2x upsampling via transposed convolution.
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # Attention gate: filters the skip connection using decoder features as gate signal.
        self.att = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 16))
        # RRCNN block to fuse upsampled features and filtered skip connection.
        # This is the key difference from Attention U-Net, which uses a plain ConvBlock here.
        self.rrcnn = RRCNNBlock(out_ch + skip_ch, out_ch, t=t)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle potential 1-pixel resolution mismatch due to pooling/upsampling rounding.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # Step 2: Apply attention gate to filter the skip connection.
        skip = self.att(x, skip)

        # Step 3: Concatenate along the channel dimension.
        x = torch.cat((skip, x), dim=1)

        # Step 4: Fuse via RRCNN block (recurrent residual fusion, not plain ConvBlock).
        x = self.rrcnn(x)
        return x


class AttentionR2UNet(nn.Module):
    """
    Attention R2U-Net: Combines Recurrent Residual U-Net (R2U-Net) with Attention Gates.

    This model merges the advantages of both architectures:
    - From R2U-Net: Recurrent Residual Convolutional blocks (RRCNN) replace standard convolutions,
      enabling the model to capture richer spatial and contextual information over time steps.
    - From Attention U-Net: Attention Gates in skip connections suppress irrelevant background
      features, focusing the decoder on salient regions.

    Architecture:
      Encoder: 4 levels of RRCNN blocks + MaxPool, channel progression: 32 -> 64 -> 128 -> 256 -> 512.
      Bottleneck: RRCNN block at 512 channels.
      Decoder: 4 levels of UpRRCNNAttBlock (upsample + attention + RRCNN), channel progression reversed.
      Output: 1x1 conv -> NUM_CLASSES channels.

    This is the most complex model in the 4-script comparison, combining both
    recurrent modeling and attention-based feature selection.
    """

    def __init__(self, in_channels=3, num_classes=3, base_ch=32, t=2):
        super().__init__()

        # ==================== Encoder ====================
        # Progressive downsampling via MaxPool, doubling channels at each level.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 1: 3 -> 32 channels
        self.enc1 = RRCNNBlock(in_channels, base_ch, t=t)
        # Level 2: 32 -> 64 channels
        self.enc2 = RRCNNBlock(base_ch, base_ch * 2, t=t)
        # Level 3: 64 -> 128 channels
        self.enc3 = RRCNNBlock(base_ch * 2, base_ch * 4, t=t)
        # Level 4: 128 -> 256 channels
        self.enc4 = RRCNNBlock(base_ch * 4, base_ch * 8, t=t)

        # ==================== Bottleneck ====================
        # Deepest features: 256 -> 512 channels
        self.center = RRCNNBlock(base_ch * 8, base_ch * 16, t=t)

        # ==================== Decoder ====================
        # Each decoder block: UpRRCNNAttBlock = upsampling + attention gate + RRCNN fusion.
        # Level 4: 512 -> 256 channels (decoder) + skip from enc4 (256ch) -> output 256ch.
        self.dec4 = UpRRCNNAttBlock(base_ch * 16, base_ch * 8, base_ch * 8, t=t)
        # Level 3: 256 -> 128 channels + skip from enc3 (128ch) -> output 128ch.
        self.dec3 = UpRRCNNAttBlock(base_ch * 8, base_ch * 4, base_ch * 4, t=t)
        # Level 2: 128 -> 64 channels + skip from enc2 (64ch) -> output 64ch.
        self.dec2 = UpRRCNNAttBlock(base_ch * 4, base_ch * 2, base_ch * 2, t=t)
        # Level 1: 64 -> 32 channels + skip from enc1 (32ch) -> output 32ch.
        self.dec1 = UpRRCNNAttBlock(base_ch * 2, base_ch, base_ch, t=t)

        # Final 1x1 convolution to map from base_ch to num_classes.
        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        # Level 2: downsample then apply RRCNN.
        e2 = self.enc2(self.pool(e1))
        # Level 3: downsample then apply RRCNN.
        e3 = self.enc3(self.pool(e2))
        # Level 4: downsample then apply RRCNN.
        e4 = self.enc4(self.pool(e3))
        # Bottleneck: downsample then apply RRCNN (deepest features).
        c  = self.center(self.pool(e4))

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
    """
    Compute the mean Dice coefficient across all classes.

    The Dice coefficient measures the overlap between the predicted region
    and the ground truth region for each class: Dice = 2*|A∩B| / (|A|+|B|).

    Args:
        pred (torch.Tensor): Predicted class indices per pixel, shape (N, H, W).
        target (torch.Tensor): Ground truth class indices per pixel, shape (N, H, W).
        num_classes (int): Total number of classes (including background).
        eps (float): Small epsilon value to avoid division by zero.

    Returns:
        float: Mean Dice coefficient across all classes. Higher is better (max=1.0).
    """
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

    Dice Loss directly optimizes the Dice coefficient, which is especially useful
    for imbalanced segmentation tasks where some classes occupy much fewer pixels.
    The loss is computed as: L = 1 - mean(Dice_c) across all classes.
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

    Focal Loss down-weights the contribution of easy examples and focuses on
    hard or misclassified samples. It is particularly effective for class-imbalanced
    datasets: L = -alpha * (1-pt)^gamma * log(pt).
    Here alpha is implicitly 1.0 and gamma=2.0.
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        total += loss.item() * x.size(0)

        # Argmax over channel dimension gives predicted class index per pixel.
        pred = torch.argmax(logits, dim=1)
        dices.append(mean_dice(pred.cpu(), y.cpu(), num_classes))
        ious.append(mean_iou(pred.cpu(), y.cpu(), num_classes))

    return total / len(loader.dataset), float(np.mean(dices)), float(np.mean(ious))


# =============================================================================
# Main Function
# =============================================================================

def main():
    seed_everything(SEED)

    train_imgs = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*")))
    train_stems = [os.path.splitext(os.path.basename(p))[0] for p in train_imgs]

    random.seed(SEED)
    random.shuffle(train_stems)

    train_stems = train_stems[:TOTAL_SAMPLES]
    tr_stems, va_stems = train_stems[:TRAIN_SAMPLES], train_stems[TRAIN_SAMPLES:]
    print("Using stems -> train:", len(tr_stems), "val:", len(va_stems))

    # -------------------------------------------------------------------------
    # Data augmentation pipeline for training set (Albumentations).
    # These augmentations are applied online during training to increase diversity.
    # -------------------------------------------------------------------------
    train_tf = A.Compose([
        # Resize all images to a fixed resolution (768x768) for batch processing.
        A.Resize(TRAIN_RES, TRAIN_RES),
        # Random horizontal flip: 50% chance, helps model generalize to left/right orientation.
        A.HorizontalFlip(p=0.5),
        # Random vertical flip: 20% chance — specific to this script (not in U-Net_R2U).
        A.VerticalFlip(p=0.2),
        # Random affine transformation: scale (0.9~1.1), slight translation, rotation (-15~15 deg).
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(0.0, 0.05),
            rotate=(-15, 15),
            p=0.5
        ),
        # Random brightness/contrast adjustment.
        A.RandomBrightnessContrast(p=0.3),
        # Gaussian noise augmentation — specific to this script.
        A.GaussNoise(p=0.2),
        # Normalize pixel values to zero-mean, unit-variance.
        A.Normalize(),
        # Convert image and mask to PyTorch tensors (C, H, W) format.
        ToTensorV2()
    ])

    # Minimal preprocessing for validation set (deterministic, no randomness).
    val_tf = A.Compose([
        A.Resize(VAL_RES, VAL_RES),
        A.Normalize(),
        ToTensorV2()
    ])

    # Create dataset and dataloader instances for train and validation splits.
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

    # Use CUDA GPU if available, otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Model: Attention R2U-Net with base_ch=32, t=2 recurrent steps.
    # Combines recurrent residual blocks (R2U-Net) with attention gates (Attention U-Net).
    # -------------------------------------------------------------------------
    model = AttentionR2UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_ch=32,
        t=2
    ).to(device)

    # -------------------------------------------------------------------------
    # Loss, Optimizer, and Scheduler.
    # -------------------------------------------------------------------------
    loss_fn = HybridSegLoss(NUM_CLASSES)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # ReduceLROnPlateau: reduce learning rate by factor=0.5 when val_dice plateaus.
    # mode="max" because higher Dice is better.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6   # Minimum learning rate floor — specific to this script.
    )

    # Mixed-precision gradient scaler.
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # Create checkpoint directory and initialize tracking variables.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best = -1.0

    # -------------------------------------------------------------------------
    # Training Loop.
    # -------------------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        # Train for one epoch and get average training loss.
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        # Evaluate on validation set: get loss, mDice, and mIoU.
        va_loss, va_dice, va_iou = validate(model, val_loader, loss_fn, device, NUM_CLASSES)

        # Step the scheduler based on validation mDice (ReduceLROnPlateau, mode="max").
        scheduler.step(va_dice)

        # Print epoch summary with all key metrics.
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={va_loss:.4f} | "
            f"mDice={va_dice:.4f} | "
            f"mIoU={va_iou:.4f}"
        )

        # Save model whenever validation mDice improves.
        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME))
            print(f"  Saved best model: mDice={best:.4f}")


if __name__ == "__main__":
    main()
