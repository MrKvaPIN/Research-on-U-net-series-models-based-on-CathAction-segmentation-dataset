"""
U-Net: Standard U-Net architecture with ResNet34 encoder (without ImageNet pretrain).
ResNet34 is used as the backbone encoder, followed by a symmetric decoder with skip connections.
"""

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
import segmentation_models_pytorch as smp


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

BATCH_SIZE      = 2
NUM_WORKERS     = 2

CHECKPOINT_DIR  = "checkpoints"
BEST_MODEL_NAME = "best_unet.pt"


# =============================================================================
# Dataset: images read by cv2; masks read by np.load
# =============================================================================
class CathDatasetNPY(Dataset):
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

        # Load image: glob searches for any file with the same stem and any extension.
        # This handles both .jpg and .png images automatically.
        img_path = glob.glob(os.path.join(self.img_dir, s + ".*"))[0]
        # Mask is stored as a .npy file (NumPy array), values in {0, 1, 2}.
        msk_path = os.path.join(self.mask_dir, s + ".npy")

        # Read image in color (3 channels: BGR from cv2).
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Convert BGR to RGB for standard 3-channel convention.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask: shape (H, W), values are integers representing class labels.
        mask = np.load(msk_path).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask


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
        # Convert target from (N, H, W) indices to (N, C, H, W) one-hot encoding.
        target_1h = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Compute per-class intersection and union for Dice.
        dims = (0, 2, 3)  # sum over batch, height, and width dimensions.
        inter = (probs * target_1h).sum(dim=dims)
        denom = probs.sum(dim=dims) + target_1h.sum(dim=dims)

        # Per-class Dice coefficient with smoothing to avoid zero division.
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class FocalLossMultiClass(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        # Probability of the true class, derived from the cross-entropy formulation.
        pt = torch.exp(-ce)
        # Focal weight: reduces loss for well-classified examples (pt near 1).
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

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, leave=False):
        # Move data to the appropriate device.
        x, y = x.to(device), y.to(device)
        # Forward pass: get logits from model.
        logits = model(x)
        # Compute loss between predictions and ground truth.
        loss = loss_fn(logits, y)

        # Backward pass: zero gradients, compute gradients, update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate weighted loss for computing average.
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, device, num_classes=NUM_CLASSES):
    """
    Evaluate the model on the validation set.

    Computes loss, mean Dice, and mean IoU as primary segmentation metrics.

    Args:
        model (nn.Module): The segmentation model to evaluate.
        loader (DataLoader): Validation data loader.
        loss_fn (nn.Module): Loss function to compute validation loss.
        device (str): Device to run validation on ("cuda" or "cpu").
        num_classes (int): Number of segmentation classes.

    Returns:
        tuple: (avg_val_loss, mean_dice, mean_iou) - all floats.
    """
    model.eval()
    total = 0.0
    dices, ious = [], []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
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
    train_imgs = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*")))
    # Extract stems (filename without extension) to match images with .npy masks.
    train_stems = [os.path.splitext(os.path.basename(p))[0] for p in train_imgs]

    # Shuffle stems with a fixed seed for reproducibility.
    random.seed(42)
    random.shuffle(train_stems)

    # Take only the first TOTAL_SAMPLES stems, then split into train and val.
    train_stems = train_stems[:TOTAL_SAMPLES]
    tr_stems, va_stems = train_stems[:TRAIN_SAMPLES], train_stems[TRAIN_SAMPLES:]
    print("Using stems -> train:", len(tr_stems), "val:", len(va_stems))

    # -----------------------------------------------------------------------------
    # Data augmentation pipeline for training set (Albumentations).
    # These augmentations are applied online during training to increase diversity.
    # -----------------------------------------------------------------------------
    train_tf = A.Compose([
        # Resize all images to a fixed resolution (768x768) for batch processing.
        A.Resize(TRAIN_RES, TRAIN_RES),
        # Random horizontal flip: 50% chance, helps model generalize to left/right orientation.
        A.HorizontalFlip(p=0.5),
        # Random affine transformation: scale (0.9~1.1), slight translation, rotation (-15~15 deg).
        # Probability 50%, simulates mild viewpoint variations.
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(0.0, 0.05),
            rotate=(-15, 15),
            p=0.5
        ),
        # Random brightness/contrast adjustment: helps model be robust to imaging variations.
        A.RandomBrightnessContrast(p=0.3),
        # Normalize pixel values to zero-mean, unit-variance (ImageNet-like normalization).
        A.Normalize(),
        # Convert image and mask to PyTorch tensors (C, H, W) format.
        ToTensorV2()
    ])

    val_tf = A.Compose([
        A.Resize(VAL_RES, VAL_RES),
        A.Normalize(),
        ToTensorV2()
    ])

    # Create dataset and dataloader instances for train and validation splits.
    train_ds = CathDatasetNPY(TRAIN_IMG_DIR, TRAIN_MSK_DIR, tr_stems, transform=train_tf)
    val_ds   = CathDatasetNPY(TRAIN_IMG_DIR, TRAIN_MSK_DIR, va_stems, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Use CUDA GPU if available, otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    ).to(device)

    # -------------------------------------------------------------------------
    # Loss, Optimizer, and Scheduler.
    # -------------------------------------------------------------------------
    loss_fn = HybridSegLoss(NUM_CLASSES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Create checkpoint directory and initialize tracking variables.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best = -1.0  # Track the best (highest) validation mDice seen so far.

    # -------------------------------------------------------------------------
    # Training Loop.
    # -------------------------------------------------------------------------
    for epoch in range(1, EPOCHS):
        # Train for one epoch and get average training loss.
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        # Evaluate on validation set: get loss, mDice, and mIoU.
        va_loss, va_dice, va_iou = validate(model, val_loader, loss_fn, device, NUM_CLASSES)

        # Step the scheduler based on validation loss (ReduceLROnPlateau).
        scheduler.step(va_loss)

        # Print epoch summary with all key metrics.
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | "
              f"val_loss={va_loss:.4f} | mDice={va_dice:.4f} | mIoU={va_iou:.4f}")

        # Save model whenever validation mDice improves.
        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME))
            print(f"  Saved best model: mDice={best:.4f}")


if __name__ == "__main__":
    main()
