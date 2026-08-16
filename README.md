# AI-Based Image Restoration for Semiconductor Inspection

A compact PyTorch pipeline for restoring degraded grayscale inspection images. The model combines a NAFNet restoration backbone, a learned degradation-aware conditioning branch, and a detail-focused 2× super-resolution head.

The project was developed for the **SEMICON India Hackathon 2026 — KLA problem statement: AI-Based Restoration of Degraded Images for Semiconductor Inspection**.

## What the model does

```text
Degraded grayscale image
          │
          ├──────────────► Degradation analyzer
          │                       │
          ▼                       ▼
        NAFNet              16-D embedding
          │                       │
          └───────────────► γ / β modulation
                                  │
                         F' = F(1 + γ) + β
                                  │
                           Detail / SR head
                                  │
                                  ▼
                         Restored 2× image
```

The development model was trained on paired degraded/ground-truth grayscale `.npy` images. Its restoration objective combines pixel fidelity, edge structure and frequency content:

`L = L1 + 0.1 × gradient loss + 0.01 × frequency-magnitude loss`

The frequency term compares the magnitude of the 2-D Fourier spectra using an orthonormal FFT. The model is fully convolutional; the submitted evaluator therefore preserves the supplied input resolution rather than resizing every test image to a fixed size.

## Validated development result

These numbers are from our held-out development validation split, **not the official hidden hackathon test set**.

| Item | Result |
|---|---:|
| Input | 128 × 128 grayscale |
| Output | 256 × 256 grayscale |
| Parameters | 6.415216 M |
| Best checkpoint epoch | 16 |
| Mean validation PSNR | **27.947178 dB** |

Representative samples and the short demo video are presentation evidence; the repository's `outputs/` directory is reserved for the actual restored test outputs.

## Repository structure

```text
image-restoration-ai/
├── models/
│   ├── nafnet.py
│   └── integrated_model.py
├── data/
│   └── dataset.py
├── checkpoints/
│   └── best_integrated_model.pth       # add the real weights before submission
├── outputs/                            # actual restored test outputs
├── train.py                            # training from scratch
├── inference.py                        # inference-only entry point
├── evaluate.py                         # standalone benchmark/evaluation script
├── smoke_test.py                       # architecture sanity check
├── requirements.txt
└── README.md
```

## 1. Install

Python 3.12 is recommended for the development environment used for this project.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Before final submission, replace the broad dependency ranges in `requirements.txt` with the exact `pip freeze` captured from the verified training environment, as required by the hackathon.

## 2. Dataset format

The development dataset uses matching filenames:

```text
DATASET/
├── NoisyLR/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
└── GT/
    ├── 000000.npy
    ├── 000001.npy
    └── ...
```

`NoisyLR` and `GT` files must have matching stems. The development pipeline uses a deterministic 90/10 train/validation split with seed `42`.

The supplied degraded arrays may contain values outside `[0, 1]` because the challenge explicitly notes that speckle noise can push intensities beyond the ground-truth range. The evaluator therefore does **not** clip the input before inference; clipping is applied only to the model output.

## 3. Train from scratch

```bash
python train.py \
  --lr-dir /path/to/DATASET/NoisyLR \
  --gt-dir /path/to/DATASET/GT \
  --output checkpoints/best_integrated_model.pth \
  --epochs 30 \
  --batch-size 8
```

The training script records the best validation PSNR and stores the model state, optimizer state, scheduler state, epoch, PSNR and validation loss in the checkpoint.

## 4. Run inference

The inference entry point requires only an input directory, output directory and trained checkpoint:

```bash
python inference.py \
  --input-dir /path/to/test_images \
  --output-dir outputs/test_results \
  --checkpoint checkpoints/best_integrated_model.pth
```

The script accepts `.npy`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` and `.tiff` grayscale inputs and writes restored PNG images.

## 5. Run the standalone evaluator

This is the script intended for reproducible benchmarking:

```bash
python evaluate.py \
  --input-dir /path/to/test_images \
  --output-dir outputs/test_results \
  --checkpoint checkpoints/best_integrated_model.pth
```

It requires no notebook and no manual code edits. It loads the checkpoint, processes every supported image, writes the restored outputs, and reports mean inference time.

For local paired validation, add the ground-truth directory:

```bash
python evaluate.py \
  --input-dir /path/to/DATASET/NoisyLR \
  --gt-dir /path/to/DATASET/GT \
  --output-dir outputs/validation_results \
  --checkpoint checkpoints/best_integrated_model.pth
```

When ground truth is supplied, PSNR and SSIM are also reported.

## 6. Quick architecture check

The model structure can be checked without the dataset or trained weights:

```bash
python smoke_test.py
```

Expected output includes:

```text
Output shape : (1, 1, 256, 256)
Parameters   : 6,415,216 (6.415216 M)
Smoke test passed.
```

## Weights and test outputs

The final submission must contain the **actual trained checkpoint** and the **actual restored outputs produced by that checkpoint**. They are deliberately not replaced with fabricated placeholders.

If the checkpoint is too large for a normal GitHub file, use Git LFS or a downloadable release/storage link, then update the README with the exact download location.

## Reproducibility checklist

Before submitting:

- [ ] `python smoke_test.py` passes on a clean environment.
- [ ] `requirements.txt` contains the exact verified environment (`pip freeze`).
- [ ] `checkpoints/best_integrated_model.pth` or a documented downloadable weight is available.
- [ ] `evaluate.py` runs from the command line without editing the source.
- [ ] The official test outputs are present under `outputs/`.
- [ ] The reported 27.947178 dB figure is clearly labelled as development validation, not hidden-test performance.

## Official problem statement

SEMICON India Hackathon 2026, KLA track:
https://i4c.in/hackathon-2026/

The official submission requirements call for a public repository containing a complete README, standalone evaluation script, training code, downloadable trained weights, restored test outputs and reproducible dependencies. The evaluation script is intended to be run as-is by the benchmarking team.
