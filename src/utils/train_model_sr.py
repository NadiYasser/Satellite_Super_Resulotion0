import os 
import json
import torch 
from importlib import reload
from src.utils.helper_functions import train_sr, val_sr, plot_sr_progress


def train_model_sr(
    model,
    model_name,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    scale_factor,
    model_requires_upscale,
    best_model_path,
    history_path,
    use_amp=False,
    scaler=None
):


    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    best_psnr = 0.0
    start_epoch = 0

    # LOAD HISTORY
    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []

    if os.path.exists(history_path):
        print(" Loading training history...")
        with open(history_path, "r") as f:
            history = json.load(f)

        train_losses = history["train_losses"]
        val_losses   = history["val_losses"]
        train_psnrs  = history["train_psnrs"]
        val_psnrs    = history["val_psnrs"]
        start_epoch = len(train_losses) 
    else:
        print("No previous training history found.")

    # LOAD CHECKPOINT
    if os.path.exists(best_model_path):
        print("Loading checkpoint:", best_model_path)
        checkpoint = torch.load(best_model_path, map_location=device)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            best_psnr = checkpoint["best_psnr"]
            print(f" Resume from epoch {start_epoch} | Best PSNR = {best_psnr:.2f}")
        else:
            print(" Old checkpoint without optimizer/scheduler. Loading model only.")
            model.load_state_dict(checkpoint)

    else:
        print(f" Training {model_name} from scratch")

    # TRAIN LOOP
    num_epochs += start_epoch

    for epoch in range(start_epoch, num_epochs):
        print(f"\n [{model_name}] Epoch {epoch+1}/{num_epochs}")

        # ---- TRAIN ----
        if use_amp:
            train_loss, train_psnr, scaler = train_sr(
                model=model,
                train_loader=train_loader,
                loss_fn=criterion,
                optimizer=optimizer,
                device=device,
                scale_factor=4,
                model_requires_upscale=False,
                scheduler=scheduler,
                use_amp=True,
                scaler=scaler
            )
        else:
            train_loss, train_psnr = train_sr(
                model=model,
                train_loader=train_loader,
                loss_fn=criterion,
                optimizer=optimizer,
                device=device,
                scale_factor=4,
                model_requires_upscale=model_requires_upscale,
                scheduler=scheduler,
                use_amp=False
            )

        # ---- VALIDATION ----
        val_loss, val_psnr = val_sr(
            model=model,
            val_loader=val_loader,
            loss_fn=criterion,
            device=device,
            scale_factor=scale_factor,
            model_requires_upscale=model_requires_upscale
        )

        # ---- SAVE BEST MODEL ----
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_psnr": best_psnr
            }, best_model_path)

            print(f" New BEST model saved at epoch {epoch+1} with PSNR = {best_psnr:.2f}")

        # ---- SAVE HISTORY ----
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnrs.append(train_psnr)
        val_psnrs.append(val_psnr)

        with open(history_path, "w") as f:
            json.dump({
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_psnrs": train_psnrs,
                "val_psnrs": val_psnrs
            }, f)

        print(f"Train loss: {train_loss:.6f} | Train PSNR: {train_psnr:.2f} dB")
        print(f"Val   loss: {val_loss:.6f} | Val   PSNR: {val_psnr:.2f} dB")
        print(f"-> LR: {optimizer.param_groups[0]['lr']:.8f}")


    # FINAL PLOT
    plot_sr_progress(train_losses, val_losses, train_psnrs, val_psnrs)

    return 0
