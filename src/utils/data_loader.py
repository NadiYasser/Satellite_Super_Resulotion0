import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# DATASET SANS AUGMENTATION

class SRDataset(Dataset):
    """
    Dataset Super-Resolution simple :
    - Charge les patches LR/HR
    - Convertit en tensor
    - Aucune augmentation
    """
    def __init__(self, root, split="train"):
        self.hr_dir = os.path.join(root, split, "HR")
        self.lr_dir = os.path.join(root, split, "LR")

        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        fname = os.path.basename(hr_path)
        lr_path = os.path.join(self.lr_dir, fname)

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        return self.to_tensor(lr), self.to_tensor(hr)



# DATASET AVEC AUGMENTATION


class SRDatasetAug(Dataset):
    """
    Dataset Super-Resolution avec Data Augmentation :
    - Flip horizontal/vertical
    - Rotations 0°, 90°, 180°, 270°
    """
    def __init__(self, root, split="train"):
        self.hr_dir = os.path.join(root, split, "HR")
        self.lr_dir = os.path.join(root, split, "LR")

        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, "*.png")))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        fname = os.path.basename(hr_path)
        lr_path = os.path.join(self.lr_dir, fname)

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # ----- Data Augmentation Sync LR-HR -----
        if random.random() < 0.5:
            hr = TF.hflip(hr)
            lr = TF.hflip(lr)

        if random.random() < 0.5:
            hr = TF.vflip(hr)
            lr = TF.vflip(lr)

        angle = random.choice([0, 90, 180, 270])
        hr = TF.rotate(hr, angle)
        lr = TF.rotate(lr, angle)

        return self.to_tensor(lr), self.to_tensor(hr)




def create_loaders(
        root,
        batch_size=64,
        num_workers=4,
        use_augmentation=True
    ):
    """
    Retourne train_loader, val_loader, test_loader
    """

    # Choose Train Dataset Type
    if use_augmentation:
        train_set = SRDatasetAug(root, split="train")
    else:
        train_set = SRDataset(root, split="train")

    val_set = SRDataset(root, split="val")
    test_set = SRDataset(root, split="test")

    # DataLoaders 
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    print(f"\n DATA LOADED:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")

    return train_loader, val_loader, test_loader
