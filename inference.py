import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained restoration model.")
    parser.add_argument("--input-dir", required=True, help="Directory of degraded images")
    parser.add_argument("--output-dir", required=True, help="Directory for restored images")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_integrated_model.pth",
        help="Path to the trained checkpoint",
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

    total_ms = 0.0
    for path in files:
        array = load_image(path)
        tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            restored = model(tensor).clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_ms += (time.perf_counter() - start) * 1000.0

        save_image(restored[0, 0].cpu().numpy(), output_dir / f"{path.stem}.png")

    print("=" * 64)
    print(f"Processed images : {len(files)}")
    print(f"Mean inference   : {total_ms / len(files):.3f} ms/image")
    print(f"Outputs saved to : {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
