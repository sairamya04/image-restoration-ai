# Results

## Development validation

| Metric | Value |
|---|---:|
| Mean PSNR | **27.947178 dB** |
| Best epoch | **16** |
| Parameters | **6.415216 M** |
| Input | **128 × 128** |
| Output | **256 × 256** |

These measurements are from the held-out development validation split. They are **not official hidden-test results**.

## Representative samples

| Sample | PSNR | SSIM |
|---|---:|---:|
| Sample 01 | 32.28 dB | 0.8701 |
| Sample 02 | 26.91 dB | 0.8477 |
| Sample 03 | 27.67 dB | 0.5680 |

The sample metrics apply only to their corresponding images.

## Demo

`outputs/demo_25s.mp4` shows the restoration pipeline from degraded input to restored output with ground-truth comparison.

## Reproduction

Use `evaluate.py` with the trained checkpoint and paired ground truth to reproduce PSNR/SSIM measurements. Do not treat presentation samples as the overall validation score.
