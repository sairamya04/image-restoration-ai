import argparse
import os
import time

import torch
import torch.nn.functional as F

from data.dataset import make_loaders
from models.integrated_model import IntegratedRestorationModel


SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def gradient_loss(pred, target):
    sx = SOBEL_X.to(device=pred.device, dtype=pred.dtype)
    sy = SOBEL_Y.to(device=pred.device, dtype=pred.dtype)
    pred_x = F.conv2d(pred, sx, padding=1)
    pred_y = F.conv2d(pred, sy, padding=1)
    target_x = F.conv2d(target, sx, padding=1)
    target_y = F.conv2d(target, sy, padding=1)
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def frequency_loss(pred, target):
    pred_fft = torch.fft.fft2(pred, norm="ortho")
    target_fft = torch.fft.fft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def restoration_loss(pred, target):
    pixel = F.l1_loss(pred, target)
    grad = gradient_loss(pred, target)
    freq = frequency_loss(pred, target)
    return pixel + 0.1 * grad + 0.01 * freq, pixel, grad, freq


def main():
    parser = argparse.ArgumentParser(description="Train the integrated image-restoration model.")
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", default="checkpoints/best_integrated_model.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(
        args.lr_dir, args.gt_dir, batch_size=args.batch_size, seed=args.seed
    )

    model = IntegratedRestorationModel().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.6f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    best_psnr = float("-inf")
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0

        for lr, gt in train_loader:
            lr = lr.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(lr)
            loss, _, _, _ = restoration_loss(pred, gt)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += loss.item()

        model.eval()
        val_total = 0.0
        val_pixel = val_grad = val_freq = 0.0
        psnr_values = []

        with torch.no_grad():
            for lr, gt in val_loader:
                lr = lr.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                pred = model(lr).clamp(0.0, 1.0)
                total, pixel, grad, freq = restoration_loss(pred, gt)
                val_total += total.item()
                val_pixel += pixel.item()
                val_grad += grad.item()
                val_freq += freq.item()
                mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
                psnr_values.extend((-10.0 * torch.log10(mse + 1e-10)).cpu().tolist())

        n_train = len(train_loader)
        n_val = len(val_loader)
        train_total /= n_train
        val_total /= n_val
        val_pixel /= n_val
        val_grad /= n_val
        val_freq /= n_val
        mean_psnr = sum(psnr_values) / len(psnr_values)

        scheduler.step()
        print(
            f"Epoch [{epoch + 1:02d}/{args.epochs}] "
            f"Train: {train_total:.6f} Val: {val_total:.6f} "
            f"PSNR: {mean_psnr:.5f} dB LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        print(f"       L1={val_pixel:.6f} Grad={val_grad:.6f} Freq={val_freq:.6f}")

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "psnr": best_psnr,
                    "val_loss": val_total,
                },
                args.output,
            )
            print(f"       NEW BEST MODEL | {best_psnr:.5f} dB")

    elapsed = (time.perf_counter() - start_time) / 60.0
    print("=" * 72)
    print(f"Training complete | Best PSNR: {best_psnr:.5f} dB")
    print(f"Training time: {elapsed:.2f} min")
    print(f"Checkpoint: {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
