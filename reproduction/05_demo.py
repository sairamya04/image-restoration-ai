# ============================================================
# FINAL LIVE DEMO
# 4 CONSECUTIVE IMAGES
# ACTUAL INPUT → INTEGRATED MODEL → OUTPUT + GROUND TRUTH
# ============================================================

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import structural_similarity as ssim


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("DEVICE")
print("=" * 80)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 80)


# ============================================================
# 2. LOAD VERIFIED BEST CHECKPOINT
# ============================================================

BEST_PATH = (
    "/kaggle/working/final_integrated_model/"
    "best_integrated_model.pth"
)

print()
print("=" * 80)
print("LOADING BEST CHECKPOINT")
print("=" * 80)

print("Checkpoint:", BEST_PATH)
print("Exists:", os.path.exists(BEST_PATH))

if not os.path.exists(BEST_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{BEST_PATH}"
    )


checkpoint = torch.load(
    BEST_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Checkpoint loaded successfully.")

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Validation PSNR:",
    f"{checkpoint['psnr']:.6f} dB"
)


# ============================================================
# 3. MODEL INFORMATION
# ============================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

num_parameters_m = (
    num_parameters / 1e6
)

print()
print("=" * 80)
print("MODEL INFORMATION")
print("=" * 80)

print(
    f"Model parameters: "
    f"{num_parameters_m:.6f} M"
)

print("=" * 80)


# ============================================================
# 4. METRIC FUNCTION
# ============================================================

def calculate_metrics(pred, gt):

    pred = np.clip(
        pred,
        0.0,
        1.0
    )

    gt = np.clip(
        gt,
        0.0,
        1.0
    )

    mse = np.mean(
        (pred - gt) ** 2
    )

    psnr = (
        10.0
        *
        np.log10(
            1.0 /
            (mse + 1e-10)
        )
    )

    ssim_value = ssim(
        gt,
        pred,
        data_range=1.0
    )

    return (
        mse,
        psnr,
        ssim_value
    )


# ============================================================
# 5. COLLECT FOUR CONSECUTIVE VALIDATION IMAGES
# ============================================================

print()
print("=" * 80)
print("COLLECTING 4 CONSECUTIVE VALIDATION IMAGES")
print("=" * 80)

demo_samples = []

with torch.no_grad():

    count = 0

    for lr_batch, gt_batch in val_loader:

        for i in range(
            lr_batch.shape[0]
        ):

            lr_np = (
                lr_batch[i, 0]
                .cpu()
                .numpy()
                .copy()
            )

            gt_np = (
                gt_batch[i, 0]
                .cpu()
                .numpy()
                .copy()
            )

            demo_samples.append(
                (
                    lr_np,
                    gt_np
                )
            )

            count += 1

            if count == 4:
                break

        if count == 4:
            break


if len(demo_samples) < 4:

    raise RuntimeError(
        f"Only {len(demo_samples)} validation "
        f"images were found. Need at least 4."
    )


print(
    "Images selected:",
    len(demo_samples)
)

for i, (lr_np, gt_np) in enumerate(
    demo_samples,
    start=1
):

    print(
        f"Image {i}: "
        f"Input {lr_np.shape} → "
        f"GT {gt_np.shape}"
    )


# ============================================================
# 6. WARM-UP
#
# The first CUDA inference can include initialization overhead.
# We warm up once before measuring the four images.
# ============================================================

print()
print("=" * 80)
print("GPU WARM-UP")
print("=" * 80)

warmup_tensor = torch.from_numpy(
    demo_samples[0][0]
).float().unsqueeze(
    0
).unsqueeze(
    0
).to(device)


with torch.no_grad():

    _ = model(
        warmup_tensor
    )


if device.type == "cuda":

    torch.cuda.synchronize()


print("Warm-up complete.")


# ============================================================
# 7. SEQUENTIAL SINGLE-IMAGE INFERENCE
# ============================================================

print()
print("=" * 80)
print("LIVE INFERENCE — 4 CONSECUTIVE IMAGES")
print("=" * 80)

results = []


for image_number, (
    lr_np,
    gt_np
) in enumerate(
    demo_samples,
    start=1
):

    # --------------------------------------------------------
    # Prepare ONE image
    # --------------------------------------------------------

    input_tensor = torch.from_numpy(
        lr_np
    ).float().unsqueeze(
        0
    ).unsqueeze(
        0
    ).to(device)


    # --------------------------------------------------------
    # Synchronize BEFORE timing
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.synchronize()


    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # ACTUAL MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        output_tensor = model(
            input_tensor
        )

        output_tensor = torch.clamp(
            output_tensor,
            0.0,
            1.0
        )


    # --------------------------------------------------------
    # Synchronize AFTER inference
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.synchronize()


    # --------------------------------------------------------
    # STOP TIMER
    # --------------------------------------------------------

    end_time = time.perf_counter()


    # --------------------------------------------------------
    # INFERENCE TIME
    # --------------------------------------------------------

    inference_time_ms = (
        end_time
        -
        start_time
    ) * 1000.0


    # --------------------------------------------------------
    # Convert output to NumPy
    # --------------------------------------------------------

    restored_np = (
        output_tensor[0, 0]
        .detach()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if not np.isfinite(
        restored_np
    ).all():

        raise RuntimeError(
            f"Image {image_number} "
            "contains NaN or Inf."
        )


    if restored_np.min() < 0.0:
        raise RuntimeError(
            f"Image {image_number} "
            "has values below 0."
        )


    if restored_np.max() > 1.0:
        raise RuntimeError(
            f"Image {image_number} "
            "has values above 1."
        )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mse, psnr, ssim_value = (
        calculate_metrics(
            restored_np,
            gt_np
        )
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LIVE PRINT
    # --------------------------------------------------------

    print(
        f"Image {image_number}: "
        f"{inference_time_ms:.3f} ms/image | "
        f"PSNR: {psnr:.2f} dB | "
        f"SSIM: {ssim_value:.4f}"
    )


# ============================================================
# 8. INFERENCE SUMMARY
# ============================================================

times = [
    result["time_ms"]
    for result in results
]

psnrs = [
    result["psnr"]
    for result in results
]

ssims = [
    result["ssim"]
    for result in results
]


average_time = np.mean(
    times
)

median_time = np.median(
    times
)

average_psnr = np.mean(
    psnrs
)

average_ssim = np.mean(
    ssims
)


print()
print("=" * 80)
print("4-IMAGE INFERENCE SUMMARY")
print("=" * 80)

for i, t in enumerate(
    times,
    start=1
):

    print(
        f"Image {i} inference : "
        f"{t:.3f} ms"
    )


print("-" * 80)

print(
    f"Average inference   : "
    f"{average_time:.3f} ms/image"
)

print(
    f"Median inference    : "
    f"{median_time:.3f} ms/image"
)

print(
    f"Average PSNR        : "
    f"{average_psnr:.4f} dB"
)

print(
    f"Average SSIM        : "
    f"{average_ssim:.4f}"
)

print("=" * 80)


# ============================================================
# 9. DISPLAY FOUR CONSECUTIVE RESULTS
# ============================================================

plt.figure(
    figsize=(15, 12)
)


for idx, result in enumerate(
    results
):

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    plt.subplot(
        4,
        3,
        idx * 3 + 1
    )

    plt.imshow(
        result["input"],
        cmap="gray"
    )

    plt.title(
        f"Image {result['image']} — Degraded Input\n"
        f"{result['input'].shape[0]} × "
        f"{result['input'].shape[1]}"
    )

    plt.axis("off")


    # --------------------------------------------------------
    # RESTORED
    # --------------------------------------------------------

    plt.subplot(
        4,
        3,
        idx * 3 + 2
    )

    plt.imshow(
        result["restored"],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title(
        f"AI Restored Output\n"
        f"PSNR: {result['psnr']:.2f} dB | "
        f"SSIM: {result['ssim']:.4f}\n"
        f"Inference: {result['time_ms']:.2f} ms"
    )

    plt.axis("off")


    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    plt.subplot(
        4,
        3,
        idx * 3 + 3
    )

    plt.imshow(
        result["gt"],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title(
        f"Ground Truth\n"
        f"{result['gt'].shape[0]} × "
        f"{result['gt'].shape[1]}"
    )

    plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 10. FINAL BENCHMARK REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL BENCHMARK REPORT")
print("=" * 80)

print(
    f"Model parameters     : "
    f"{num_parameters_m:.6f} M"
)

print(
    f"Images processed     : 4"
)

print(
    f"Input resolution     : "
    f"{results[0]['input'].shape[0]} × "
    f"{results[0]['input'].shape[1]}"
)

print(
    f"Output resolution    : "
    f"{results[0]['restored'].shape[0]} × "
    f"{results[0]['restored'].shape[1]}"
)

print(
    f"Average inference    : "
    f"{average_time:.3f} ms/image"
)

print(
    f"Median inference     : "
    f"{median_time:.3f} ms/image"
)

print(
    f"Average PSNR         : "
    f"{average_psnr:.4f} dB"
)

print(
    f"Average SSIM         : "
    f"{average_ssim:.4f}"
)

print("=" * 80)
print("LIVE 4-IMAGE BENCHMARK COMPLETE")
print("=" * 80)