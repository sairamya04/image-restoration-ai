# Dataset

Paired degraded and ground-truth grayscale images stored as NumPy `.npy` files.

## Layout

```text
DATASET/
├── NoisyLR/
│   ├── 000000.npy
│   └── ...
└── GT/
    ├── 000000.npy
    └── ...
```

Each `NoisyLR/<name>.npy` is paired with `GT/<name>.npy`.

## Format

- Input: single-channel degraded image, typically 128 × 128
- Target: single-channel ground truth, typically 256 × 256
- Storage: `.npy`
- Pairing: filename stem

The development dataset contains 3,200 paired samples and uses a deterministic 90/10 train-validation split with seed `42`.

The dataset is supplied separately and is not stored in the repository.

## Loader

`dataset.py` creates the paired dataset and DataLoaders used by `train.py`.
