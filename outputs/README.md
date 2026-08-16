# Restored outputs

This directory is reserved for the actual restored images produced by the final checkpoint on the official test set.

For local validation, run:

```bash
python evaluate.py \
  --input-dir /path/to/NoisyLR \
  --gt-dir /path/to/GT \
  --output-dir outputs/validation_results \
  --checkpoint checkpoints/best_integrated_model.pth
```

Do not use presentation-only screenshots as a substitute for the restored test-output files.
