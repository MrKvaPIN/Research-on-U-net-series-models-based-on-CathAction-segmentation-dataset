import os, glob
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# -------------------------
# Dataset: image=cv2, mask=np.load(.npy)
# -------------------------
class CathDatasetNPY(Dataset):
    def __init__(self, img_dir, mask_dir, stems, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.stems = stems
        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        s = self.stems[idx]
        img_path = glob.glob(os.path.join(self.img_dir, s + ".*"))[0]
        msk_path = os.path.join(self.mask_dir, s + ".npy")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.load(msk_path).astype(np.uint8)  # (H,W) values {0,1,2}

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask


@torch.no_grad()
def per_class_dice_iou(pred, target, num_classes=3, eps=1e-7):
    # pred/target: (N,H,W)
    dices, ious = [], []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)

        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        union = (p | t).sum().item()

        dice = (2 * inter + eps) / (denom + eps) if denom > 0 else float("nan")
        iou  = (inter + eps) / (union + eps) if union > 0 else float("nan")

        dices.append(dice)
        ious.append(iou)
    return dices, ious


@torch.no_grad()
def evaluate(model, loader, device, num_classes=3):
    model.eval()

    # 累积统计（全数据集级别，更稳定）
    inter = np.zeros(num_classes, dtype=np.float64)
    denom = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)  # (N,H,W)

        for c in range(num_classes):
            p = (pred == c)
            t = (y == c)
            inter[c] += (p & t).sum().item()
            denom[c] += p.sum().item() + t.sum().item()
            union[c] += (p | t).sum().item()

    dice = (2 * inter + 1e-7) / (denom + 1e-7)
    iou  = (inter + 1e-7) / (union + 1e-7)

    # 背景也算进平均（与你训练脚本的 mean_dice/mean_iou逻辑一致）
    mdice = float(np.nanmean(dice))
    miou  = float(np.nanmean(iou))

    return dice.tolist(), iou.tolist(), mdice, miou


def main():
    # ===== 改这里：你的数据根目录 =====
    ROOT = r"D:\FYP\Data_set\segmentation"
    test_img_dir = os.path.join(ROOT, "phantom_test", "images")
    test_msk_dir = os.path.join(ROOT, "phantom_test", "masks")

    # 模型参数路径
    CKPT = r"checkpoints\best_unet.pt"

    num_classes = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 注意：这里的模型定义必须与训练时一致
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # 加载你训练好的权重，不需要 imagenet 权重
        in_channels=3,
        classes=num_classes,
        activation=None
    ).to(device)

    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)

    # test stems
    test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*")))
    test_stems = [os.path.splitext(os.path.basename(p))[0] for p in test_imgs]
    print("Test samples:", len(test_stems))

    tf = A.Compose([
        A.Resize(512, 512),   # 必须与你训练/验证用的尺寸一致
        A.Normalize(),
        ToTensorV2()
    ])

    test_ds = CathDatasetNPY(test_img_dir, test_msk_dir, test_stems, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    dice, iou, mdice, miou = evaluate(model, test_loader, device, num_classes=num_classes)

    # 输出结果
    print("\nPer-class Dice (class 0/1/2):", [round(x, 4) for x in dice])
    print("Per-class IoU  (class 0/1/2):", [round(x, 4) for x in iou])
    print("Mean Dice:", round(mdice, 4))
    print("Mean IoU :", round(miou, 4))


if __name__ == "__main__":
    main()