# AI Image Restoration for Semiconductor Inspection

PyTorch pipeline for restoring degraded grayscale inspection images.

**SEMICON India Hackathon 2026 · KLA**

## Pipeline

```text
128×128 degraded input
        ↓
NAFNet restoration backbone
        ↓
Degradation-aware conditioning
        ↓
Detail + 2× reconstruction head
        ↓
256×256 restored output
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

These are development-validation results from the final trained checkpoint. They are not hidden-test scores.

## Results

`outputs/` contains three representative input/restored/ground-truth comparisons and the **25-second demo video**.

See [`results.md`](results.md) for the sample metrics.

## Repository

```text
image-restoration-ai/
├── data/               # Dataset loader and input format
├── models/             # NAFNet and integrated restoration model
├── outputs/            # Sample results and demo video
├── train.py            # Model training
├── inference.py        # Single-run image restoration
├── evaluate.py         # PSNR, SSIM and timing evaluation
├── smoke_test.py       # Fast architecture sanity check
├── requirements.txt    # Python dependencies
├── results.md          # Verified results record
└── README.md
```

## Dataset

Use paired grayscale `.npy` files with matching filenames:

```text
DATASET/
├── NoisyLR/
│   ├── 000000.npy
│   └── ...
└── GT/
    ├── 000000.npy
    └── ...
```

The dataset is provided separately and is not included in this repository.

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

## Train

```bash
python train.py \
  --lr-dir /path/to/DATASET/NoisyLR \
  --gt-dir /path/to/DATASET/GT \
  --output best_integrated_model.pth
```

Default configuration: 30 epochs, batch size 8, AdamW, cosine learning-rate decay, seed 42.

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

With ground truth, this reports mean PSNR, mean SSIM, and mean inference time.

## Smoke test

`smoke_test.py` is a fast architecture check. It does **not** train the model and does **not** require the dataset or checkpoint.

Run:

```bash
python smoke_test.py
```

It verifies:

- model construction and imports
- parameter count
- expected tensor dimensions
- 128×128 input → 256×256 output

A successful run confirms that the repository's model code loads and produces the expected output shape before a full training or inference run.
