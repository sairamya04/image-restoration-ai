# Sample outputs

This directory contains representative visual results from the trained image-restoration model.

Each comparison should show the same sample in three stages:

```text
Degraded Input  →  AI Restored Output  →  Ground Truth
```

## Recommended samples

Use a small set of **real validation outputs**, not screenshots of the notebook interface. Each sample should be exported as a clean PNG and, where available, labelled with its measured PSNR and SSIM.

Recommended files:

```text
outputs/
├── sample_01.png
├── sample_02.png
└── sample_03.png
```

The strongest currently verified presentation examples include:

| Sample | PSNR | SSIM |
|---|---:|---:|
| Sample 01 | 32.28 dB | 0.8701 |
| Sample 02 | 26.91 dB | 0.8477 |
| Sample 03 | 27.67 dB | 0.5680 |

These numbers should only be retained if the corresponding exported images come from the same inference run. Do not mix metrics from one run with images from another.

## Validation outputs

For a full validation run:

```bash
python evaluate.py \
  --input-dir /path/to/NoisyLR \
  --gt-dir /path/to/GT \
  --output-dir outputs/validation_results \
  --checkpoint checkpoints/best_integrated_model.pth
```

Full generated outputs can be kept under `outputs/validation_results/`; presentation samples belong directly in `outputs/`.

The competition dataset and trained checkpoint are not bundled into this repository unless explicitly released for redistribution.
