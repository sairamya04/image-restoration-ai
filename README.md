# AI Image Restoration for Semiconductor Inspection

PyTorch restoration pipeline for degraded grayscale inspection images.

**SEMICON India Hackathon 2026 · KLA**

## Model

```text
Degraded image
      │
      ├── NAFNet restoration backbone
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

## Validation

| Metric | Value |
|---|---:|
| Mean PSNR | **27.915544 dB** |
| Best epoch | **18** |
| Parameters | **6.415216 M** |
| Input | **128 × 128** |
| Output | **256 × 256** |

Measured on the held-out development validation split for the submitted checkpoint.

## Results

`outputs/` contains three representative comparisons and the 25-second demo.

```text
Degraded Input → Restored Output → Ground Truth
```

See [`results.md`](results.md) for sample metrics.

## Structure

```text
image-restoration-ai/
├── models/             # Architecture
├── data/               # Dataset loader
├── checkpoints/        # Trained weights
├── outputs/            # Results and demo
├── train.py            # Training
├── inference.py        # Inference
├── evaluate.py         # Evaluation
├── smoke_test.py       # Model check
├── requirements.txt    # Dependencies
├── results.md          # Results
└── README.md
```

## Dataset

Paired grayscale NumPy arrays with matching filenames:

```text
DATASET/
├── NoisyLR/
└── GT/
```

The dataset is supplied separately and is not stored in the repository.

## Train

```bash
python train.py \
  --lr-dir /path/to/DATASET/NoisyLR \
  --gt-dir /path/to/DATASET/GT \
  --output checkpoints/best_integrated_model.pth
```

Defaults: 30 epochs, batch size 8, AdamW, cosine learning-rate decay and seed 42.

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

With ground truth, the evaluator reports PSNR and SSIM.

## Quick check

```bash
python smoke_test.py
```

Verifies model construction, parameter count and output dimensions.

## Dependencies

```bash
pip install -r requirements.txt
```
