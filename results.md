# Results

## Development validation

The final integrated model was selected using the held-out development validation split.

| Metric | Value |
|---|---:|
| Mean PSNR | **27.947178 dB** |
| Best checkpoint epoch | **16** |
| Parameters | **6.415216 M** |
| Input | 128 × 128 |
| Output | 256 × 256 |

These are development-validation measurements, not official hidden-test scores.

## Representative visual evidence

Two representative validation samples used in the presentation/demo were:

| Sample | PSNR | SSIM |
|---|---:|---:|
| Representative sample A | 32.28 dB | 0.8701 |
| Representative sample B | 26.91 dB | 0.8477 |

These examples are included as qualitative evidence and are **not** reported as the overall validation score.

## Inference

The evaluation script measures inference time with CUDA synchronization around the forward pass. Final submission latency should be reported from the exact evaluator/checkpoint used for benchmarking, rather than from a different demo configuration.

## Benchmark hygiene

The repository does not claim a hidden-test score before the official test set is evaluated. The complete official restored-output folder should be added under `outputs/` after the test data is released.
