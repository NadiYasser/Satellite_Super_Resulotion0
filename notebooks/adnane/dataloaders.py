# this files contains dataloaders for esrgan model only, it not used by other models in this project


# imports 
import torch
import os 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader


# Dataset class 
class Satellite_dataset(Dataset):
    def __init__(self, root_dir, split = "train"):
        super().__init__()
        self.split = split
        self.lr_dir = os.path.join(root_dir, split, "LR")
        self.hr_dir = os.path.join(root_dir, split, "HR")
        
        self.images_files = sorted(os.listdir(self.lr_dir))
        
        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.images_files)
    
    def __getitem__(self, index):
        filename = self.images_files[index]
        lr = Image.open(os.path.join(self.lr_dir, filename)).convert("RGB")
        hr = Image.open(os.path.join(self.hr_dir, filename)).convert("RGB")
        
        lr = self.lr_transform(lr)
        hr = self.hr_transform(hr)

        
        return lr, hr


def create_loaders_for_ESRGAN(
        root,
        batch_size=64,
        num_workers=4
    ):
    """
    Retourne train_loader, val_loader, test_loader
    """
    train_set = Satellite_dataset(root, split="train")
    val_set = Satellite_dataset(root, split="val")
    test_set = Satellite_dataset(root, split="test")

    # DataLoaders 
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
        )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n DATA LOADED:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")

    return train_loader, val_loader, test_loader