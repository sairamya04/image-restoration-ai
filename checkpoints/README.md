# Model Checkpoint

This directory stores the trained model weights used for inference and evaluation.

## Expected file

```text
best_integrated_model.pth
```

The checkpoint must contain a PyTorch `model_state_dict` compatible with the model implementation in `models/integrated_model.py`.

## Validated result

The best validated development checkpoint was obtained at **epoch 16** with a mean validation **PSNR of 27.947178 dB**.

> The binary checkpoint is not included in the repository yet. It will be added separately once the validated weight file is available. No placeholder weights are used.
