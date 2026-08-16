# Results & Demo

Representative outputs from the image-restoration model.

## Samples

Each comparison shows:

**Degraded Input → Restored Output → Ground Truth**

| Sample | PSNR | SSIM |
|---|---:|---:|
| Sample 01 | 32.28 dB | 0.8701 |
| Sample 02 | 26.91 dB | 0.8477 |
| Sample 03 | 27.67 dB | 0.5680 |

These are representative samples, not the overall validation score.

## Demo

`demo_25s.mp4` is the 25-second end-to-end restoration demonstration.

## Files

```text
outputs/
├── sample_01.png
├── sample_02.png
├── sample_03.png
└── demo_25s.mp4
```

The sample images and video are presentation evidence from the project inference pipeline. Full evaluation outputs can be generated with `evaluate.py`.
