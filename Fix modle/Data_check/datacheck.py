import os, glob
import numpy as np
import cv2

# Missing masks/images 是否为空（不空就说明有对不上的文件名，需要先修正）

# Mask type 是 GRAYSCALE 还是 COLOR，以及 unique 值大概长什么样

ROOT = r"D:\FYP\Data_set\segmentation"  
IMG_DIR = os.path.join(ROOT, "Phantom_test", "images")
MSK_DIR = os.path.join(ROOT, "Phantom_test", "masks")

def stem(p):
    return os.path.splitext(os.path.basename(p))[0]

img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*")))
msk_paths = sorted(glob.glob(os.path.join(MSK_DIR, "*")))

print("train images:", len(img_paths))
print("train masks :", len(msk_paths))

img_stems = set(stem(p) for p in img_paths)
msk_stems = set(stem(p) for p in msk_paths)

print("Missing masks for images:", sorted(list(img_stems - msk_stems))[:20])
print("Missing images for masks:", sorted(list(msk_stems - img_stems))[:20])

# ---- 随机抽样检查 mask 的 dtype / shape / unique ----
sample_paths = msk_paths[:20]
all_uniques = set()
shapes = set()
dtypes = set()
binary_count = 0
three_class_count = 0
other_class_counts = {}

for p in sample_paths:
    ext = os.path.splitext(p)[1].lower()
    if ext == ".npy":
        m = np.load(p)
    else:
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)

    dtypes.add(str(m.dtype))
    shapes.add(tuple(m.shape))

    # 只统计前几个 unique，避免过大
    u = np.unique(m)
    for x in u[:50]:
        # numpy 标量转 python 标量
        all_uniques.add(int(x) if np.issubdtype(type(x), np.integer) else float(x))

# ---- 全量统计每个 mask 的分类数 ----
for p in msk_paths:
    ext = os.path.splitext(p)[1].lower()
    if ext == ".npy":
        m = np.load(p)
    else:
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)

    class_count = len(np.unique(m))

    if class_count == 2:
        binary_count += 1
    elif class_count == 3:
        three_class_count += 1
    else:
        other_class_counts[class_count] = other_class_counts.get(class_count, 0) + 1

print("\nMask dtypes:", dtypes)
print("Mask resolution:", shapes)
print("Unique class values (partial):", list(all_uniques)[:30])
print("Unique class count (sampled):", len(all_uniques))
print("\n2-class masks:", binary_count)
print("3-class masks:", three_class_count)

if other_class_counts:
    print("Other class-count masks:", dict(sorted(other_class_counts.items())))

# ---- 额外：检查是否是 one-hot（最后一维是类别） ----
for s in shapes:
    if len(s) == 3 and s[2] <= 10:
        print("\nLooks like one-hot or multi-channel mask with channels =", s[2])
        break
