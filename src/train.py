import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.config import CONFIG
from utils.data_loader import create_loaders
from models.models_architecture import SRCNN        # ou EDSR, etc.
from utils.helper_functions import train_sr, test_sr, plot_sr_progress

device = "cuda" if torch.cuda.is_available() else "cpu"


# CONFIG FROM YAML 
data_root      = CONFIG["paths"]["output_root"]
batch_size     = CONFIG["training"]["batch_size"]
num_workers    = CONFIG["training"]["num_workers"]
use_aug        = CONFIG["training"].get("use_augmentation", True)



# LOAD DATA 
train_loader, val_loader, test_loader = create_loaders(
    root=data_root,
    batch_size=batch_size,
    num_workers=num_workers,
    use_augmentation=use_aug
)



# HYPERPARAMS FROM CONFIG 
lr              = CONFIG["training"]["lr"]
weight_decay    = CONFIG["training"]["weight_decay"]
num_epochs      = CONFIG["training"]["epochs"]
step_size       = CONFIG["training"]["scheduler_step_size"]
gamma           = CONFIG["training"]["scheduler_gamma"]




model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)




train_losses, val_losses = [], []
train_psnrs,  val_psnrs  = [], []

best_psnr = 0.0
best_model_path = CONFIG["model"]["best_model_path"]
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)



for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss, train_psnr = train_sr(
        model, train_loader, criterion, optimizer, device, scheduler
    )
    val_loss, val_psnr = test_sr(
        model, val_loader, criterion, device
    )


    # save best model
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), best_model_path)
        print(f"🔥 New best model saved with Val PSNR = {best_psnr:.2f} dB")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_psnrs.append(train_psnr)
    val_psnrs.append(val_psnr)

    print(f"Train loss: {train_loss:.6f} | Train PSNR: {train_psnr:.2f} dB")
    print(f"Val   loss: {val_loss:.6f} | Val   PSNR: {val_psnrs[-1]:.2f} dB")
    print(f"  ➤ LR: {optimizer.param_groups[0]['lr']:.8f}")

# courbes
plot_sr_progress(train_losses, val_losses, train_psnrs, val_psnrs)
