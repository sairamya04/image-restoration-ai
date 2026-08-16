# AI Image Restoration for Semiconductor Inspection

A PyTorch image-restoration pipeline for degraded grayscale inspection imagery.

**SEMICON India Hackathon 2026 — KLA problem statement**

## Approach

```text
Degraded image
      │
      ├──► NAFNet restoration backbone
      │
      └──► Degradation analyzer
                 │
              16-D embedding
                 │
              γ / β modulation
                 │
                 ▼
          Detail + 2× SR head
                 │
                 ▼
          Restored image
```

The model combines:

- NAFNet for image restoration
- degradation-aware conditioning through learned γ/β modulation
- a detail-focused 2× reconstruction head
- L1, gradient and frequency-domain losses

Loss:

```text
L = L1 + 0.1 × Lgradient + 0.01 × Lfrequency
```

## Development result

Measured on the held-out development validation split:

| Metric | Result |
|---|---:|
| Mean PSNR | **27.947178 dB** |
| Best epoch | **16** |
| Parameters | **6.415216 M** |
| Input | **128 × 128** |
| Output | **256 × 256** |

These are development results, **not hidden-test scores**.

## Results & demo

The `outputs/` directory contains three representative restoration samples and the **25-second demonstration video**.

Each visual comparison follows:

```text
Degraded Input → Restored Output → Ground Truth
```

## Repository

```text
image-restoration-ai/
├── models/             # Model architecture
├── data/               # Dataset loader and format
├── outputs/            # Sample results + demo
├── checkpoints/        # Trained weights
├── train.py            # Training pipeline
├── inference.py        # Image restoration
├── evaluate.py         # Evaluation and metrics
├── smoke_test.py       # Model sanity check
├── requirements.txt    # Python dependencies
├── results.md          # Measured results
└── README.md
```

## Dataset

The training pipeline expects paired grayscale `.npy` files with matching filenames:

```text
DATASET/
├── NoisyLR/
└── GT/
```

The dataset is supplied separately and is **not committed to this repository**.

## Training

```bash
python train.py \
  --lr-dir /path/to/DATASET/NoisyLR \
  --gt-dir /path/to/DATASET/GT \
  --output checkpoints/best_integrated_model.pth
```

Default training configuration: 30 epochs, batch size 8, AdamW, cosine learning-rate decay, and deterministic 90/10 train-validation splitting with seed 42.

## Inference

```bash
python inference.py \
  --input-dir /path/to/images \
  --output-dir outputs/restored \
  --checkpoint checkpoints/best_integrated_model.pth
```

## Evaluation

```bash
python evaluate.py \
  --input-dir /path/to/NoisyLR \
  --gt-dir /path/to/GT \
  --output-dir outputs/validation_results \
  --checkpoint checkpoints/best_integrated_model.pth
```

When ground truth is supplied, the evaluator reports PSNR and SSIM and records inference timing.

## Verification

```bash
python smoke_test.py
```

The smoke test verifies model construction, parameter count and tensor dimensions without requiring the training dataset.

## Reproducibility

The repository contains the model definition, dataset loader, training pipeline, inference entry point and evaluation script. The trained checkpoint must be supplied separately if it is not committed to the repository.

Before submission, verify:

- [ ] trained checkpoint is available
- [ ] sample outputs correspond to the submitted checkpoint
- [ ] evaluation runs without source-code edits
- [ ] dependencies are pinned to the verified environment
- [ ] development metrics are clearly separated from official test results
