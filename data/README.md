# Dataset

The project uses paired low-resolution/degraded (`NoisyLR`) and ground-truth (`GT`) grayscale images stored as NumPy `.npy` arrays.

## Expected directory structure

```text
train/
└── train/
    ├── NoisyLR/
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    └── GT/
        ├── 000000.npy
        ├── 000001.npy
        └── ...
```

Each `NoisyLR/<name>.npy` must have a matching `GT/<name>.npy` file. The loader pairs samples by filename.

## Data format

- Input: single-channel degraded image, typically `128 × 128`
- Target: single-channel ground truth image, typically `256 × 256`
- Storage: NumPy `.npy`
- Value range: floating-point image values expected by the model pipeline

The dataset itself is **not committed to this repository**. It should be provided separately through the competition/Kaggle dataset environment.

## Loader

`dataset.py` provides `SemiconductorDataset` for paired samples and `make_loaders()` for deterministic train/validation splitting.

Default split: **90% training / 10% validation**, with seed `42`.
