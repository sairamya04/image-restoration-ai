# ============================================================
# FINAL INTEGRATED MODEL — TRAINING
# Exact training configuration used for the submitted model.
# ============================================================

import os
import time
import torch
import torch.nn.functional as F


def gradient_loss(pred, target):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)
    target_x = F.conv2d(target, sobel_x, padding=1)
    target_y = F.conv2d(target, sobel_y, padding=1)
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def frequency_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def total_restoration_loss(pred, target):
    l1 = F.l1_loss(pred, target)
    grad = gradient_loss(pred, target)
    freq = frequency_loss(pred, target)
    total = l1 + 0.1 * grad + 0.01 * freq
    return total, l1, grad, freq


# Run this script after 01_dataset.py and 03_model.py in the same
# notebook/session, preserving the original notebook workflow.

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.99),
    weight_decay=1e-6
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,
    eta_min=1e-6
)

CHECKPOINT_DIR = "/kaggle/working/final_integrated_model"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_PATH = os.path.join(CHECKPOINT_DIR, "best_integrated_model.pth")

best_psnr = -float("inf")
best_epoch = 0
NUM_EPOCHS = 30

print("=" * 80)
print("FINAL INTEGRATED MODEL — TRAINING")
print("=" * 80)
start_time = time.perf_counter()

for epoch in range(NUM_EPOCHS):
    model.train()
    train_total = train_l1 = train_grad = train_freq = 0.0

    for lr, gt in train_loader:
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(lr)
        loss, l1, grad, freq = total_restoration_loss(pred, gt)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_total += loss.item()
        train_l1 += l1.item()
        train_grad += grad.item()
        train_freq += freq.item()

    train_total /= len(train_loader)
    train_l1 /= len(train_loader)
    train_grad /= len(train_loader)
    train_freq /= len(train_loader)

    model.eval()
    psnr_values = []
    val_total = 0.0

    with torch.no_grad():
        for lr, gt in val_loader:
            lr = lr.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            pred = torch.clamp(model(lr), 0.0, 1.0)
            loss, _, _, _ = total_restoration_loss(pred, gt)
            val_total += loss.item()
            mse = torch.mean((pred - gt) ** 2, dim=(1,2,3))
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
            psnr_values.extend(psnr.cpu().tolist())

    val_total /= len(val_loader)
    mean_psnr = sum(psnr_values) / len(psnr_values)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] Train: {train_total:.6f} Val: {val_total:.6f} PSNR: {mean_psnr:.5f} dB LR: {current_lr:.2e}")
    print(f"       L1={train_l1:.6f} Grad={train_grad:.6f} Freq={train_freq:.6f}")

    if mean_psnr > best_psnr:
        best_psnr = mean_psnr
        best_epoch = epoch + 1
        torch.save({
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "psnr": best_psnr,
            "val_loss": val_total
        }, BEST_PATH)
        print(f"       ✓ NEW BEST MODEL | {best_psnr:.5f} dB")

elapsed = time.perf_counter() - start_time
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Best Epoch : {best_epoch}")
print(f"Best PSNR  : {best_psnr:.5f} dB")
print(f"Training time : {elapsed / 60:.2f} min")
print("Checkpoint:", BEST_PATH)
print("Checkpoint exists:", os.path.exists(BEST_PATH))
print("=" * 80)
