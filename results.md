# Results

## Development validation

| Metric | Value |
|---|---:|
| Mean PSNR | **27.915544 dB** |
| Best epoch | **18** |
| Parameters | **6.415216 M** |
| Input | **128 × 128** |
| Output | **256 × 256** |

Measured on the held-out development validation split for the submitted checkpoint.

## Representative samples

| Sample | PSNR | SSIM |
|---|---:|---:|
| Sample 01 | 32.28 dB | 0.8701 |
| Sample 02 | 26.91 dB | 0.8477 |
| Sample 03 | 27.67 dB | 0.5680 |

These values apply to the corresponding samples only.

## Demo

`outputs/demo_25s.mp4` shows degraded input, restored output and ground-truth comparison.

## Evaluation

Run `evaluate.py` with the checkpoint and paired ground truth to reproduce PSNR and SSIM.
