import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from models.integrated_model import IntegratedRestorationModel

SUPPORTED = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
    else:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel image: {path} -> {array.shape}")
    return array


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(array, 0.0, 1.0) * 255.0
    Image.fromarray(image.astype(np.uint8), mode="L").save(path)


def make_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    # Fully convolutional model: preserve the supplied resolution.
    return torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0).to(device)


def find_ground_truth(gt_dir: Path, stem: str):
    for suffix in SUPPORTED:
        candidate = gt_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone SEMICON/KLA image-restoration evaluator."
    )
    parser.add_argument("--input-dir", required=True, help="Directory of degraded test images")
    parser.add_argument("--output-dir", required=True, help="Directory for restored outputs")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_integrated_model.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Optional matching ground-truth directory for local PSNR/SSIM evaluation",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = IntegratedRestorationModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        raise RuntimeError(f"No supported images found in {input_dir}")

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    metrics = []
    total_ms = 0.0

    for path in files:
        array = load_image(path)
        tensor = make_tensor(array, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model(tensor).clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_ms += (time.perf_counter() - start) * 1000.0

        restored = prediction[0, 0].cpu().numpy()
        save_image(restored, output_dir / f"{path.stem}.png")

        if gt_dir is not None:
            gt_path = find_ground_truth(gt_dir, path.stem)
            if gt_path is None:
                print(f"Warning: no ground truth found for {path.name}")
                continue

            gt = load_image(gt_path)
            if gt.shape != restored.shape:
                raise ValueError(
                    f"Shape mismatch for {path.stem}: restored={restored.shape}, gt={gt.shape}"
                )

            mse = float(np.mean((restored - gt) ** 2))
            psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
            ssim = structural_similarity(gt, restored, data_range=1.0)
            metrics.append((psnr, ssim))

    mean_ms = total_ms / len(files)
    print("=" * 64)
    print(f"Processed images : {len(files)}")
    print(f"Mean inference   : {mean_ms:.3f} ms/image")
    if metrics:
        mean_psnr = np.mean([m[0] for m in metrics])
        mean_ssim = np.mean([m[1] for m in metrics])
        print(f"Mean PSNR        : {mean_psnr:.6f} dB")
        print(f"Mean SSIM        : {mean_ssim:.6f}")
    print(f"Outputs saved to : {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
