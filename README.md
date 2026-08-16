# AI-Based Restoration of Degraded Images for Semiconductor Inspection

A PyTorch restoration and 2× super-resolution pipeline for grayscale semiconductor inspection images. The validated final architecture combines a NAFNet restoration backbone, learned degradation-aware conditioning, and a detail-aware super-resolution head.

## Validated development result

- Input: degraded grayscale image, 128×128 in the supplied training split
- Output: restored grayscale image, 256×256
- Final integrated model: ~6.415 M parameters
- Best validation checkpoint: epoch 16
- Mean validation PSNR: **27.947178 dB**
- Training loss: L1 + 0.1×Sobel gradient loss + 0.01×frequency-magnitude loss

The reported metric is from the project validation split; it is not a claim about the official hidden test set.

## Architecture

```text
Degraded LR image
       |
       +----> NAFNet restoration backbone ----+
       |                                       |
       +----> Degradation analyzer -> 16-D ----+--> gamma/beta
                                                |
                          F' = F(1 + gamma) + beta
                                                |
                                      Detail/SR head
                                                |
                                         256×256 output
```

The degradation-conditioning path starts with zero gamma/beta output so the initial modulation is neutral.

## Repository layout

```text
models/nafnet.py             NAFNet backbone
models/integrated_model.py   final integrated architecture
data/dataset.py              paired .npy dataset loader
train.py                     reproducible training script
evaluate.py                  standalone inference/evaluation script
requirements.txt             runtime dependencies
checkpoints/                 place final trained weights here
outputs/                     place generated restored outputs here
```

## Dataset format

The development dataset used paired NumPy files:

```text
DATASET/
├── NoisyLR/
│   ├── sample_000.npy
│   └── ...
└── GT/
    ├── sample_000.npy
    └── ...
```

`NoisyLR` and `GT` filenames must match. The loader uses a deterministic 90/10 split with seed 42, matching the validated development pipeline.

## Train

```bash
pip install -r requirements.txt
python train.py --lr-dir /path/to/NoisyLR --gt-dir /path/to/GT --output checkpoints/best_integrated_model.pth --epochs 30 --batch-size 8
```

## Evaluate / inference

The benchmark script is a standalone Python program and does not require a Jupyter notebook. It accepts an input directory and an output directory:

```bash
python evaluate.py \
  --input-dir /path/to/test_images \
  --output-dir outputs/test_results \
  --checkpoint checkpoints/best_integrated_model.pth
```

For a paired validation set, additionally provide the ground-truth directory:

```bash
python evaluate.py \
  --input-dir /path/to/NoisyLR \
  --gt-dir /path/to/GT \
  --output-dir outputs/validation_results \
  --checkpoint checkpoints/best_integrated_model.pth
```

Supported input formats are `.npy`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`. The evaluator writes restored images to the requested output directory and reports inference time. When ground truth is supplied, it also reports PSNR and SSIM.

## Trained weights

The final validated weight file is `best_integrated_model.pth`. It must be added to `checkpoints/` or provided through a downloadable release/large-file link before final submission. The repository intentionally does not contain a fabricated or placeholder checkpoint.

## Submission note

SEMICON/KLA's published requirements state that the GitHub repository must include a complete README, standalone evaluation script, training script, downloadable trained weights, restored test outputs, and `requirements.txt`. The benchmark script must run without manual code edits. We therefore keep the runnable Python pipeline separate from the development notebook.

Official requirements: https://i4c.in/hackathon-2026/
