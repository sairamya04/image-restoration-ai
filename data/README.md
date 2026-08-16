# Dataset

Paired grayscale NumPy arrays are used for training and validation.

## Expected layout

```text
DATASET/
├── NoisyLR/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
└── GT/
    ├── 000000.npy
    ├── 000001.npy
    └── ...
```

Each `NoisyLR/<name>.npy` must have a matching `GT/<name>.npy`.

## Format

- Input: single-channel degraded image, typically `128 × 128`
- Target: single-channel ground truth, typically `256 × 256`
- Storage: `.npy`
- Pairing: filename stem

The development dataset contained **3,200 paired samples** and used a deterministic **90/10 train-validation split** with seed `42`.

The dataset is not included in this repository. It is supplied separately through the competition/Kaggle environment.

## Loader

`dataset.py` loads the paired arrays and creates the train/validation DataLoaders used by `train.py`.
