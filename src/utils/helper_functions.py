from tqdm import tqdm
import math
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

# ============================================================
#                     PSNR FUNCTION
# ============================================================

def calc_psnr(sr, hr):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio)
    """
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse.item())


# ============================================================
#                    TRAINING FUNCTION
# ============================================================

def train_sr(model, train_loader, loss_fn, optimizer, device, scheduler=None):
    model.train()
    epoch_loss = 0.0
    epoch_psnr = 0.0

    # Add tqdm progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for lr, hr in pbar:
        lr = lr.to(device)
        hr = hr.to(device)

        # Upscale LR → HR size (x4)
        lr_up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        # Forward
        sr = model(lr_up)

        # Loss
        loss = loss_fn(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_psnr = calc_psnr(sr, hr)

        epoch_loss += loss.item()
        epoch_psnr += batch_psnr

        # tqdm live update
        pbar.set_postfix({
            "loss": loss.item(),
            "psnr": batch_psnr
        })

    # Scheduler step per epoch
    if scheduler is not None:
        scheduler.step()

    return epoch_loss / len(train_loader), epoch_psnr / len(train_loader)


# ============================================================
#                    TEST FUNCTION
# ============================================================

@torch.no_grad()
def test_sr(model, test_loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_psnr = 0.0

    pbar = tqdm(test_loader, desc="Testing", leave=False)

    for lr, hr in pbar:
        lr = lr.to(device)
        hr = hr.to(device)

        lr_up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        sr = model(lr_up)
        loss = loss_fn(sr, hr)

        batch_psnr = calc_psnr(sr, hr)

        epoch_loss += loss.item()
        epoch_psnr += batch_psnr

        pbar.set_postfix({
            "loss": loss.item(),
            "psnr": batch_psnr
        })

    return epoch_loss / len(test_loader), epoch_psnr / len(test_loader)


# ============================================================
#                    PLOT FUNCTION
# ============================================================

def plot_sr_progress(train_loss, test_loss, train_psnr, test_psnr):
    """
    Plot training curves for Super-Resolution:
    - Loss per epoch
    - PSNR per epoch
    """
    plt.figure(figsize=(12, 4))
    
    # ===== Loss =====
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, c='r', label="Train Loss")
    plt.plot(test_loss, c='b', label="Test Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    
    # ===== PSNR =====
    plt.subplot(1, 2, 2)
    plt.plot(train_psnr, c='r', label="Train PSNR")
    plt.plot(test_psnr, c='b', label="Test PSNR")
    plt.title("PSNR per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
