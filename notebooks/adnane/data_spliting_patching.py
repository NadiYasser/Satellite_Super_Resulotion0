import os
import glob
import random
from PIL import Image

# Paths
dataset_hr = "./satellite-dataset/HR"
dataset_lr = "./satellite-dataset/LR"
output_dir = "./satellite-processed"

# Parameters
hr_size = 128
scale = 4
lr_size = hr_size // scale
stride = hr_size  # non-overlapping patches

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "HR"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "LR"), exist_ok=True)

# Get all HR images
images_hr = glob.glob(os.path.join(dataset_hr, "*.*"))
images_hr.sort()
random.shuffle(images_hr)

# Split images (not patches yet)
n = len(images_hr)
train_imgs = images_hr[:int(n * train_ratio)]
val_imgs   = images_hr[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
test_imgs  = images_hr[int(n * (train_ratio + val_ratio)):]

# Patch extraction function
def extract_patches(hr_path, lr_dir, split):
    img_hr = Image.open(hr_path).convert("RGB")
    w, h = img_hr.size
    filename = os.path.basename(hr_path).split('.')[0]

    # Find corresponding LR image
    lr_path = os.path.join(lr_dir, os.path.basename(hr_path))
    if not os.path.exists(lr_path):
        print(f"LR image missing: {lr_path}")
        return
    img_lr = Image.open(lr_path).convert("RGB")

    patch_id = 0
    for x in range(0, w - hr_size + 1, stride):
        for y in range(0, h - hr_size + 1, stride):
            hr_patch = img_hr.crop((x, y, x + hr_size, y + hr_size))
            lr_patch = img_lr.crop((x // scale, y // scale, x // scale + lr_size, y // scale + lr_size))

            hr_patch.save(os.path.join(output_dir, split, "HR", f"{filename}_{patch_id}.png"))
            lr_patch.save(os.path.join(output_dir, split, "LR", f"{filename}_{patch_id}.png"))
            patch_id += 1

# Process splits
for img in train_imgs:
    extract_patches(img, dataset_lr, "train")
for img in val_imgs:
    extract_patches(img, dataset_lr, "val")
for img in test_imgs:
    extract_patches(img, dataset_lr, "test")

print("Patch extraction done!")
