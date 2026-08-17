# ============================================================
# FINAL LIVE DEMO
# 4 CONSECUTIVE IMAGES
# Exact 4-image inference benchmark used for the demo.
# ============================================================

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


BEST_PATH = "/kaggle/working/final_integrated_model/best_integrated_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

checkpoint = torch.load(BEST_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

num_parameters = sum(p.numel() for p in model.parameters())
num_parameters_m = num_parameters / 1e6
print(f"Model parameters: {num_parameters_m:.6f} M")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Validation PSNR:", f"{checkpoint['psnr']:.6f} dB")


def calculate_metrics(pred, gt):
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)
    mse = np.mean((pred - gt) ** 2)
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
    ssim_value = ssim(gt, pred, data_range=1.0)
    return mse, psnr, ssim_value


demo_samples = []
with torch.no_grad():
    count = 0
    for lr_batch, gt_batch in val_loader:
        for i in range(lr_batch.shape[0]):
            lr_np = lr_batch[i, 0].cpu().numpy().copy()
            gt_np = gt_batch[i, 0].cpu().numpy().copy()
            demo_samples.append((lr_np, gt_np))
            count += 1
            if count == 4:
                break
        if count == 4:
            break

if len(demo_samples) < 4:
    raise RuntimeError(f"Only {len(demo_samples)} validation images were found. Need at least 4.")

# GPU warm-up before timing.
warmup_tensor = torch.from_numpy(demo_samples[0][0]).float().unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    _ = model(warmup_tensor)
if device.type == "cuda":
    torch.cuda.synchronize()

results = []

for image_number, (lr_np, gt_np) in enumerate(demo_samples, start=1):
    input_tensor = torch.from_numpy(lr_np).float().unsqueeze(0).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000.0
    restored_np = output_tensor[0, 0].detach().cpu().numpy()

    if not np.isfinite(restored_np).all():
        raise RuntimeError(f"Image {image_number} contains NaN or Inf.")
    if restored_np.min() < 0.0 or restored_np.max() > 1.0:
        raise RuntimeError(f"Image {image_number} has values outside [0,1].")

    mse, psnr, ssim_value = calculate_metrics(restored_np, gt_np)
    results.append({
        "image": image_number,
        "input": lr_np,
        "restored": restored_np,
        "gt": gt_np,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_value,
        "time_ms": inference_time_ms
    })

    print(f"Image {image_number}: {inference_time_ms:.3f} ms/image | PSNR: {psnr:.2f} dB | SSIM: {ssim_value:.4f}")

times = [r["time_ms"] for r in results]
psnrs = [r["psnr"] for r in results]
ssims = [r["ssim"] for r in results]
average_time = np.mean(times)
median_time = np.median(times)
average_psnr = np.mean(psnrs)
average_ssim = np.mean(ssims)

print("=" * 80)
print("4-IMAGE INFERENCE SUMMARY")
print("=" * 80)
for i, t in enumerate(times, start=1):
    print(f"Image {i} inference : {t:.3f} ms")
print("-" * 80)
print(f"Average inference   : {average_time:.3f} ms/image")
print(f"Median inference    : {median_time:.3f} ms/image")
print(f"Average PSNR        : {average_psnr:.4f} dB")
print(f"Average SSIM        : {average_ssim:.4f}")

plt.figure(figsize=(15, 12))
for idx, result in enumerate(results):
    plt.subplot(4, 3, idx * 3 + 1)
    plt.imshow(result["input"], cmap="gray")
    plt.title(f"Image {result['image']} — Degraded Input\n{result['input'].shape[0]} × {result['input'].shape[1]}")
    plt.axis("off")

    plt.subplot(4, 3, idx * 3 + 2)
    plt.imshow(result["restored"], cmap="gray", vmin=0, vmax=1)
    plt.title(f"AI Restored Output\nPSNR: {result['psnr']:.2f} dB | SSIM: {result['ssim']:.4f}\nInference: {result['time_ms']:.2f} ms")
    plt.axis("off")

    plt.subplot(4, 3, idx * 3 + 3)
    plt.imshow(result["gt"], cmap="gray", vmin=0, vmax=1)
    plt.title(f"Ground Truth\n{result['gt'].shape[0]} × {result['gt'].shape[1]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

print("=" * 80)
print("FINAL BENCHMARK REPORT")
print("=" * 80)
print(f"Model parameters     : {num_parameters_m:.6f} M")
print("Images processed     : 4")
print(f"Input resolution     : {results[0]['input'].shape[0]} × {results[0]['input'].shape[1]}")
print(f"Output resolution    : {results[0]['restored'].shape[0]} × {results[0]['restored'].shape[1]}")
print(f"Average inference    : {average_time:.3f} ms/image")
print(f"Median inference     : {median_time:.3f} ms/image")
print(f"Average PSNR         : {average_psnr:.4f} dB")
print(f"Average SSIM         : {average_ssim:.4f}")
print("LIVE 4-IMAGE BENCHMARK COMPLETE")
