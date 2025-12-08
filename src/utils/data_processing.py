import os
import shutil
import random
from tqdm import tqdm
from PIL import Image
import yaml




# LOAD CONFIG FROM config.yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

RAW_DATA = cfg["paths"]["raw_data"]
OUTPUT_ROOT = cfg["paths"]["output_root"]

SCALE = cfg["dataset"]["scale"]
GRID = cfg["dataset"]["grid"]
HR_PATCH = cfg["dataset"]["hr_patch"]

# Si lr_patch = -1 dans le yaml → on calcule automatiquement
LR_PATCH = cfg["dataset"]["lr_patch"] or (HR_PATCH // SCALE)

train_ratio = cfg["dataset"]["train_ratio"]
val_ratio = cfg["dataset"]["val_ratio"]
test_ratio = cfg["dataset"]["test_ratio"]

HR_CROP = HR_PATCH * GRID
LR_CROP = LR_PATCH * GRID

print("=== CONFIG LOADED ===")
print(f"RAW_DATA: {RAW_DATA}")
print(f"GRID: {GRID}  → {GRID*GRID} patches")
print(f"HR_PATCH: {HR_PATCH}  | LR_PATCH: {LR_PATCH}")
print(f"HR_CROP: {HR_CROP}  | LR_CROP: {LR_CROP}")





#  CRÉATION DES DOSSIERS

splits = ["train", "val", "test"]
subfolders = ["HR", "LR"]

for split in splits:
    for sub in subfolders:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, sub), exist_ok=True)


#  FONCTION CROP CENTRÉE

def center_crop(img, target_size):
    w, h = img.size
    left = (w - target_size) // 2
    top = (h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    return img.crop((left, top, right, bottom))


#  SPLIT DU DATASET RAW

hr_files = sorted(os.listdir(os.path.join(RAW_DATA, "HR")))
random.shuffle(hr_files)

n = len(hr_files)
print(f"Total images HR trouvées: {n}")

train_end = int(n * train_ratio)
val_end   = train_end + int(n * val_ratio)

train_files = hr_files[:train_end]
val_files   = hr_files[train_end:val_end]
test_files  = hr_files[val_end:]

print(f"Train: {len(train_files)} images")
print(f"Val:   {len(val_files)} images")
print(f"Test:  {len(test_files)} images")


#  FONCTION PATCHING

def process_files(file_list, split_name):
    hr_src = os.path.join(RAW_DATA, "HR")
    lr_src = os.path.join(RAW_DATA, "LR")

    out_hr_dir = os.path.join(OUTPUT_ROOT, split_name, "HR")
    out_lr_dir = os.path.join(OUTPUT_ROOT, split_name, "LR")

    for img_name in tqdm(file_list, desc=f"Patching {split_name}"):

        hr_path = os.path.join(hr_src, img_name)
        lr_path = os.path.join(lr_src, img_name)

        if not os.path.exists(lr_path):
            print(f" LR missing for {img_name}, skipped")
            continue

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # Crop carré multiple du grid
        hr = center_crop(hr, HR_CROP)
        lr = center_crop(lr, LR_CROP)

        base = os.path.splitext(img_name)[0]

        for i in range(GRID):
            for j in range(GRID):

                # HR patch
                xh = j * HR_PATCH
                yh = i * HR_PATCH
                hr_patch = hr.crop((xh, yh, xh + HR_PATCH, yh + HR_PATCH))

                # LR patch
                xl = j * LR_PATCH
                yl = i * LR_PATCH
                lr_patch = lr.crop((xl, yl, xl + LR_PATCH, yl + LR_PATCH))

                patch_name = f"{base}_r{i:02d}_c{j:02d}.png"

                hr_patch.save(os.path.join(out_hr_dir,    patch_name))
                lr_patch.save(os.path.join(out_lr_dir,    patch_name))


#  APPLY SPLIT + PATCHING
process_files(train_files, "train")
process_files(val_files,   "val")
process_files(test_files,  "test")

print("\n DONE! Patches created in data/preprocessed/")
