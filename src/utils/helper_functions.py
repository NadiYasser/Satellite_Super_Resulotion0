from tqdm import tqdm
import math
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import init
import matplotlib.pyplot as plt 
from torch.amp import autocast, GradScaler


# PSNR FUNCTION


def calc_psnr(sr, hr):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio)
    """
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse.item())





# TRAIN (avec AMP )
def train_sr(model, train_loader, loss_fn, optimizer, device,
            scale_factor=4, model_requires_upscale=False,
            scheduler=None, use_amp=False, scaler=None):

    model.train()
    epoch_loss = 0.0
    epoch_psnr = 0.0

    if use_amp and scaler is None:
        scaler = GradScaler()

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for lr, hr in pbar:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Préparation entrée modèle
        if model_requires_upscale:
            lr_in = F.interpolate(lr, scale_factor=scale_factor,
                                mode="bicubic", align_corners=False)
        else:
            lr_in = lr

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                sr = model(lr_in)
                loss = loss_fn(sr, hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            sr = model(lr_in)
            loss = loss_fn(sr, hr)
            loss.backward()
            optimizer.step()

        batch_psnr = calc_psnr(sr.detach(), hr)

        epoch_loss += loss.item()
        epoch_psnr += batch_psnr

        pbar.set_postfix({"loss": loss.item(), "psnr": batch_psnr})

    if scheduler is not None:
        scheduler.step()
    
    if use_amp:
        return epoch_loss / len(train_loader), epoch_psnr / len(train_loader), scaler
    else:
        return epoch_loss / len(train_loader), epoch_psnr / len(train_loader)
    
    





# TEST FUNCTION


@torch.no_grad()
def val_sr(model, val_loader, loss_fn, device, 
            scale_factor=4, model_requires_upscale=True):

    model.eval()
    epoch_loss = 0.0
    epoch_psnr = 0.0
    pbar = tqdm(val_loader, desc="Validation", leave=False)

    for lr, hr in pbar:
        lr = lr.to(device)
        hr = hr.to(device)

        if model_requires_upscale:
            lr_in = F.interpolate(lr, scale_factor=scale_factor, mode="bicubic", align_corners=False)
        else:
            lr_in = lr  

        sr = model(lr_in)
        loss = loss_fn(sr, hr)
        batch_psnr = calc_psnr(sr, hr)

        epoch_loss += loss.item()
        epoch_psnr += batch_psnr

        pbar.set_postfix({"loss": loss.item(), "psnr": batch_psnr})

    return epoch_loss / len(val_loader), epoch_psnr / len(val_loader)



# PLOT FUNCTION


def plot_sr_progress(train_loss, val_loss, train_psnr, val_psnr):
    """
    Plot training curves for Super-Resolution:
    - Loss per epoch
    - PSNR per epoch
    """
    plt.figure(figsize=(12, 4))
    
    # Loss 
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, c='r', label="Train Loss")
    plt.plot(val_loss, c='b', label="Val Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    
    # PSNR
    plt.subplot(1, 2, 2)
    plt.plot(train_psnr, c='r', label="Train PSNR")
    plt.plot(val_psnr, c='b', label="Val PSNR")
    plt.title("PSNR per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()



# ESRGAN 
def calculate_psnr(sr, hr, data_range=2.0):
    """
    FUNCTION Used by ESRGAN
    Calculates PSNR between SR and HR tensors in [-1, 1] range.
    sr, hr: Tensors of shape (B, C, H, W)
    """
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))


def initialize_weights(m):
    """
    Standard Kaiming Initialization for ESRGAN components.
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        
        # Scaling down the initial weights slightly as suggested in ESRGAN paper
        m.weight.data *= 0.1 
        
        if m.bias is not None:
            init.zeros_(m.bias)
            
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.zeros_(m.bias)