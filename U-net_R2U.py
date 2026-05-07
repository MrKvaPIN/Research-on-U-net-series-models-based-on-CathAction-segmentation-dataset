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
# Path to training images (supports jpg/png/bmp etc.)
TRAIN_IMG_DIR   = os.path.join(ROOT_DIR, "phantom_train_3", "images")
# Path to training masks stored as .npy files (values: 0=background, 1=class1, 2=class2)
TRAIN_MSK_DIR   = os.path.join(ROOT_DIR, "phantom_train_3", "masks")

# Number of segmentation classes (background + 2 foreground classes)
NUM_CLASSES     = 3
# Total number of training epochs
EPOCHS          = 40

# Total number of samples to load from the dataset (for subset training)
TOTAL_SAMPLES   = 4743
# Number of samples used for training (the rest are used for validation)
TRAIN_SAMPLES   = 4043

# Input resolution for the training set (width = height = 768)
TRAIN_RES       = 768
# Input resolution for the validation set (same as training for fair comparison)
VAL_RES         = 768

# Directory where model checkpoints will be saved
CHECKPOINT_DIR  = "checkpoints"
# Filename of the best model checkpoint (based on highest validation mDice)
BEST_MODEL_NAME = "best_R2unet.pt"


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

        # Load image: glob searches for any file with the same stem and any extension.
        img_path = glob.glob(os.path.join(self.img_dir, s + ".*"))[0]
        # Mask is stored as a .npy file (NumPy array), values in {0, 1, 2}.
        msk_path = os.path.join(self.mask_dir, s + ".npy")

        # Read image in color (3 channels: BGR from cv2).
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Convert BGR to RGB for standard 3-channel convention.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask: shape (H, W), values are integers representing class labels.
        mask = np.load(msk_path).astype(np.uint8)

        # Apply augmentation if a transform pipeline was provided.
        # Albumentations applies the same geometric and intensity transforms to both
        # the image and the mask simultaneously to maintain alignment.
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        # Convert mask to PyTorch long tensor for use with CrossEntropyLoss.
        # Shape: (H, W) - class indices per pixel.
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
    """
    Hybrid loss combining Dice Loss and Focal Loss for semantic segmentation.

    Combines 60% Dice Loss + 40% Focal Loss to leverage the strengths of both:
    - Dice Loss: directly optimizes overlap metric, good for small structures.
    - Focal Loss: handles class imbalance, focuses on hard examples.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.dice = DiceLossMultiClass(num_classes)
        self.focal = FocalLossMultiClass(gamma=2.0)

    def forward(self, logits, target):
        return 0.6 * self.dice(logits, target) + 0.4 * self.focal(logits, target)


# =============================================================================
# R2U-Net Modules
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
        # bias=True is used here (unlike most other blocks in this project).
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = None
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

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        t (int): Number of recurrent steps in the RecurrentBlock.
    """

    def __init__(self, in_ch, out_ch, t=2):
        super().__init__()
        # 1x1 convolution: projects input to the representation space of the recurrent block.
        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
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


class UpConv(nn.Module):
    """
    Upsampling Convolutional Block.

    This block performs upsampling (2x scale factor) using bilinear interpolation (Upsample)
    followed by a 3x3 convolution with BatchNorm and ReLU. Unlike ConvTranspose2d,
    this uses a simple upsampling operation which is less prone to checkerboard artifacts.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            # Bilinear interpolation upsampling by factor of 2.
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 3x3 convolution to refine features after upsampling.
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class R2UNet(nn.Module):
    """
    R2U-Net: U-Net with Recurrent Residual Convolutional Blocks.

    R2U-Net replaces every standard convolutional block in the encoder and decoder
    of a U-Net with RRCNN blocks. The network consists of:

    - Encoder (left side): 5 levels of RRCNN blocks, each followed by max pooling (2x downsample).
    - Bottleneck: RRCNN block at the deepest level.
    - Decoder (right side): 4 levels of Upsampling + RRCNN blocks with skip connections.
    - Output: 1x1 convolution mapping to NUM_CLASSES channels.

    The recurrent mechanism in each RRCNN block captures both forward and backward
    spatial dependencies, which is beneficial for medical image segmentation where
    structural continuity is important.

    Architecture summary:
      Encoder: RRCNN(64) -> RRCNN(128) -> RRCNN(256) -> RRCNN(512) -> RRCNN(1024)
      Decoder: Up(RC=512) -> RRCNN(512) -> Up(RC=256) -> RRCNN(256) -> ...
               Up(RC=64) -> RRCNN(64) -> 1x1 conv -> 3 classes
    """

    def __init__(self, img_ch=3, output_ch=3, t=2):
        super().__init__()

        # Encoder: progressively downsample by max pooling, doubling channels each level.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 1: 3 -> 64 channels
        self.rrcnn1 = RRCNNBlock(img_ch, 64, t=t)
        # Level 2: 64 -> 128 channels
        self.rrcnn2 = RRCNNBlock(64, 128, t=t)
        # Level 3: 128 -> 256 channels
        self.rrcnn3 = RRCNNBlock(128, 256, t=t)
        # Level 4: 256 -> 512 channels
        self.rrcnn4 = RRCNNBlock(256, 512, t=t)
        # Level 5 (bottleneck): 512 -> 1024 channels
        self.rrcnn5 = RRCNNBlock(512, 1024, t=t)

        # Decoder: upsampling with skip connections from the encoder.
        # Level 5 -> Level 4: 1024 -> 512 channels
        self.up5 = UpConv(1024, 512)
        self.up_rrcnn5 = RRCNNBlock(1024, 512, t=t)

        # Level 4 -> Level 3: 512 -> 256 channels
        self.up4 = UpConv(512, 256)
        self.up_rrcnn4 = RRCNNBlock(512, 256, t=t)

        # Level 3 -> Level 2: 256 -> 128 channels
        self.up3 = UpConv(256, 128)
        self.up_rrcnn3 = RRCNNBlock(256, 128, t=t)

        # Level 2 -> Level 1: 128 -> 64 channels
        self.up2 = UpConv(128, 64)
        self.up_rrcnn2 = RRCNNBlock(128, 64, t=t)

        # Final 1x1 convolution to map from 64 channels to the number of segmentation classes.
        self.out_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.rrcnn1(x)

        # Level 2: downsample then apply RRCNN.
        x2 = self.maxpool(x1)
        x2 = self.rrcnn2(x2)

        # Level 3: downsample then apply RRCNN.
        x3 = self.maxpool(x2)
        x3 = self.rrcnn3(x3)

        # Level 4: downsample then apply RRCNN.
        x4 = self.maxpool(x3)
        x4 = self.rrcnn4(x4)

        # Level 5 (bottleneck): downsample then apply RRCNN (deepest features).
        x5 = self.maxpool(x4)
        x5 = self.rrcnn5(x5)

        # ==================== Decoder ====================
        # Level 5 -> 4: Upsample 2x, concat with encoder skip (x4), then RRCNN.
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_rrcnn5(d5)

        # Level 4 -> 3: Upsample 2x, concat with encoder skip (x3), then RRCNN.
        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_rrcnn4(d4)

        # Level 3 -> 2: Upsample 2x, concat with encoder skip (x2), then RRCNN.
        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_rrcnn3(d3)

        # Level 2 -> 1: Upsample 2x, concat with encoder skip (x1), then RRCNN.
        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_rrcnn2(d2)

        # Final 1x1 conv to produce class scores (logits).
        out = self.out_conv(d2)
        return out


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
        A.Affine(scale=(0.90, 1.10), translate_percent=(0.0, 0.05), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        # Normalize pixel values to zero-mean, unit-variance (ImageNet-like normalization).
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

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Use CUDA GPU if available, otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Model: R2U-Net with recurrent steps t=2 (default).
    # R2U-Net uses recurrent residual conv blocks instead of plain conv blocks.
    # -------------------------------------------------------------------------
    model = R2UNet(
        img_ch=3,
        output_ch=NUM_CLASSES,
        t=2
    ).to(device)

    # -------------------------------------------------------------------------
    # Loss, Optimizer, and Scheduler.
    # -------------------------------------------------------------------------
    loss_fn = HybridSegLoss(NUM_CLASSES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Create checkpoint directory and initialize tracking variables.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best = -1.0

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
