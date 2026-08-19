# AI Image Restoration for Semiconductor Inspection

PyTorch implementation of an adaptive grayscale image-restoration model for the SEMICON India Hackathon 2026 KLA problem statement.

## Dataset

This solution was developed and trained using the **KLA-provided dataset for the AI-Based Restoration of Degraded Images problem statement**.

The dataset consists of paired degraded (`NoisyLR`) and ground-truth (`GT`) grayscale `.npy` images.

The dataset itself is not included in this repository due to its size and submission constraints.

## Final model

```text
128×128 degraded grayscale .npy
            ↓
       NAFNet backbone
            ↓
 Degradation-aware conditioning
            ↓
   Detail + 2× SR head
            ↓
256×256 restored grayscale .npy
```

The model contains **6,415,216 trainable parameters (6.415216 M)**.

## Required evaluation interface

The evaluator should run exactly:

```bash
python run.py <input-dir> <output-dir>
```

No checkpoint path, internet connection, API key, additional model download, or manual code modification is required.

`run.py`:

- reads every `.npy` file in the input directory
- accepts grayscale arrays shaped `(H, W)` or `(H, W, 1)`
- creates the output directory if necessary
- automatically loads `models/best_model_inference.pth`
- uses an NVIDIA GPU when CUDA is available
- produces one `.npy` output for every input
- preserves each input filename
- produces grayscale `(H, W)` outputs
- constrains output values to `[0, 1]`
- rejects NaN/Inf outputs
- verifies the output resolution is 2× the input resolution

## Submission structure

```text
image-restoration-ai/
├── run.py
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   ├── integrated_model.py
│   ├── nafnet.py
│   └── best_model_inference.pth
└── data/
    └── dataset.py
```

The trained inference checkpoint contains only the model `state_dict`; optimizer and scheduler states are intentionally excluded because they are not required for evaluation.

## Dataset

The KLA/SEMICON dataset is **not bundled with this repository** because of its size. The training pipeline expects paired grayscale `.npy` files with matching filenames:

```text
DATASET/
├── NoisyLR/
│   ├── 000000.npy
│   └── ...
└── GT/
    ├── 000000.npy
    └── ...
```

For training/reproduction, provide the dataset directories explicitly to the training script.

## Reproduction / training

The training code in `train.py` implements the final training configuration:

- 30 epochs
- batch size 8
- AdamW optimizer
- initial learning rate `2e-4`
- cosine annealing schedule
- seed 42
- restoration loss: `L1 + 0.1 × Lgradient + 0.01 × Lfrequency`

Example:

```bash
python train.py \
  --lr-dir /path/to/DATASET/NoisyLR \
  --gt-dir /path/to/DATASET/GT \
  --output checkpoints/best_integrated_model.pth
```

The dataset loader is in `data/dataset.py` and the model implementation is in `models/`.

## Development validation result

| Metric | Value |
|---|---:|
| Mean validation PSNR | **27.915544 dB** |
| Checkpoint epoch | **18** |
| Parameters | **6.415216 M** |
| Input | **128 × 128** |
| Output | **256 × 256** |

These are development-validation results, not hidden-test scores.


