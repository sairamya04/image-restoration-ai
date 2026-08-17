import sys
from pathlib import Path

import numpy as np
import torch

from models.integrated_model import IntegratedRestorationModel


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "best_integrated_model.pth"


def find_checkpoint() -> Path:
    """Find the bundled trained checkpoint without requiring user configuration."""
    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT

    candidates = sorted(
        p for p in MODEL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".pth", ".pt", ".ckpt"}
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "No trained model checkpoint was found in models/. "
            "Include the submitted .pth/.pt/.ckpt weights in models/."
        )
    raise RuntimeError(
        "Multiple model checkpoints found in models/. Keep one submitted "
        "checkpoint or name it best_integrated_model.pth."
    )


def load_model(device: torch.device) -> torch.nn.Module:
    checkpoint_path = find_checkpoint()
    model = IntegratedRestorationModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_npy(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path), dtype=np.float32)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(
            f"Expected grayscale array with shape (H,W) or (H,W,1): "
            f"{path} -> {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Input contains NaN or Inf: {path}")

    return array


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python run.py <input-dir> <output-dir>")

    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.npy"))
    if not input_files:
        raise SystemExit(f"No .npy files found in {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    print(f"Device: {device}")
    print(f"Input files: {len(input_files)}")

    with torch.inference_mode():
        for path in input_files:
            array = load_npy(path)
            tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)

            restored = model(tensor)
            restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
            restored = restored.clamp(0.0, 1.0)

            output = restored[0, 0].detach().cpu().numpy().astype(np.float32)

            if not np.isfinite(output).all():
                raise RuntimeError(f"Model produced NaN or Inf: {path}")
            if output.ndim != 2:
                raise RuntimeError(f"Unexpected output shape for {path}: {output.shape}")
            if output.shape != (array.shape[0] * 2, array.shape[1] * 2):
                raise RuntimeError(
                    f"Unexpected target resolution for {path}: "
                    f"input={array.shape}, output={output.shape}"
                )

            np.save(output_dir / path.name, output)

    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
