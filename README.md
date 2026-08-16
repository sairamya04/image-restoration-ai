# AI Image Restoration for Semiconductor Inspection

PyTorch pipeline for restoring degraded grayscale inspection images.

**SEMICON India Hackathon 2026 · KLA**

## Model

```text
Degraded 128×128 image
        │
        ├── NAFNet restoration backbone
        │
        └── Degradation analyzer
                 │
              16-D code
                 │
             γ / β conditioning
                 │
                 ▼
          Detail + 2× SR head
                 │
                 ▼
          Restored 256×256 image
```

Training loss:

```text
L = L1 + 0.1 × Lgradient + 0.01 × Lfrequency
```

## Development result

| Metric | Value |
|---|---:|
| Mean validation PSNR | **27.915544 dB** |
| Checkpoint epoch | **18** |
| Parameters | **6.415216 M** |
| Input | **128 × 128** |
| Output | **256 × 256** |

These values are from the final trained checkpoint used for this repository's development results. They are not hidden-test scores.

## Results

`outputs/` contains three representative visual comparisons and the 25-second demo.

```text
Degraded Input → AI Restored Output → Ground Truth
```

See [`results.md`](results.md) for the reported sample metrics.

## Repository

```text
image-restoration-ai/
├── data/               # Dataset loader
├── models/             # Restoration architecture
├── outputs/            # Sample results and demo
├── train.py            # Training
├── inference.py        # Inference
├── evaluate.py         # PSNR / SSIM evaluation
├── smoke_test.py       # Architecture check
├── requirements.txt    # Dependencies
├── results.md          # Results record
└── README.md
```

## Dataset

The loader expects paired grayscale NumPy arrays with matching filenames:

```text
DATASET/
├── NoisyLR/
│   ├── 000000.npy
│   └── ...
└── GT/
    ├── 000000.npy
    └── ...
```

The dataset is supplied separately and is not stored in this repository.

## Training

```bash
python train.py \
  --lr-dir /path/to/DATASET/NoisyLR \
  --gt-dir /path/to/DATASET/GT \
  --output best_integrated_model.pth
```

Defaults: 30 epochs, batch size 8, AdamW, cosine learning-rate decay, and seed 42.

## Inference

A trained checkpoint is required separately.

```bash
python inference.py \
  --input-dir /path/to/images \
  --output-dir outputs/restored \
  --checkpoint /path/to/best_integrated_model.pth
```

## Evaluation

```bash
python evaluate.py \
  --input-dir /path/to/NoisyLR \
  --gt-dir /path/to/GT \
  --output-dir outputs/validation_results \
  --checkpoint /path/to/best_integrated_model.pth
```

With ground truth, the evaluator reports mean PSNR, mean SSIM, and mean inference time.

## Quick check

```bash
python smoke_test.py
```

This checks model construction, parameter count, and the expected 128×128 → 256×256 output shape without requiring the dataset or checkpoint.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
