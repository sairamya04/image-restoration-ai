# Results

## Development validation

| Metric | Value |
|---|---:|
| Mean PSNR | **27.915544 dB** |
| Checkpoint epoch | **18** |
| Parameters | **6.415216 M** |
| Input | **128 × 128** |
| Output | **256 × 256** |

Measured on the held-out development validation split using the final trained checkpoint.

## Representative samples

| Sample | PSNR | SSIM |
|---|---:|---:|
| Sample 01 | 32.28 dB | 0.8701 |
| Sample 02 | 26.91 dB | 0.8477 |
| Sample 03 | 27.67 dB | 0.5680 |

These metrics are for the corresponding visual samples only.

## Demo

`outputs/semicon_live_demo_25s.mp4` shows the restoration flow from degraded input to AI-restored output with ground-truth comparison.

## Reproduce evaluation

```bash
python evaluate.py \
  --input-dir /path/to/NoisyLR \
  --gt-dir /path/to/GT \
  --output-dir outputs/validation_results \
  --checkpoint /path/to/best_integrated_model.pth
```

The checkpoint is kept outside the repository because of its file size. The source code and evaluation pipeline are included here.
