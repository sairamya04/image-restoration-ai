# ============================================================
# DATASET + TRAIN / VALIDATION LOADERS
# Exact dataset loader used in the training notebook.
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

GT_DIR = Path("/kaggle/input/datasets/sairamyaru/semicon-training/train/train/GT")
LR_DIR = Path("/kaggle/input/datasets/sairamyaru/semicon-training/train/train/NoisyLR")

print("=" * 70)
print("DATASET SETUP")
print("=" * 70)
print("GT directory exists:", GT_DIR.exists())
print("LR directory exists:", LR_DIR.exists())

class SemiconductorDataset(Dataset):
    def __init__(self, lr_dir, gt_dir):
        self.lr_dir = Path(lr_dir)
        self.gt_dir = Path(gt_dir)
        self.names = sorted([p.stem for p in self.lr_dir.glob("*.npy") if (self.gt_dir / p.name).exists()])
        print("Matching image pairs:", len(self.names))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = np.load(self.lr_dir / f"{name}.npy")
        gt = np.load(self.gt_dir / f"{name}.npy")
        lr = torch.from_numpy(lr).float().unsqueeze(0)
        gt = torch.from_numpy(gt).float().unsqueeze(0)
        return lr, gt

full_dataset = SemiconductorDataset(LR_DIR, GT_DIR)
print("Total samples:", len(full_dataset))

generator = torch.Generator().manual_seed(42)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

print("Training samples  :", len(train_dataset))
print("Validation samples:", len(val_dataset))

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

lr_batch, gt_batch = next(iter(train_loader))
print("=" * 70)
print("LOADER VERIFICATION")
print("=" * 70)
print("LR batch shape:", tuple(lr_batch.shape))
print("GT batch shape:", tuple(gt_batch.shape))
print("LR range:", lr_batch.min().item(), "to", lr_batch.max().item())
print("GT range:", gt_batch.min().item(), "to", gt_batch.max().item())
print("=" * 70)
print("train_loader:", "train_loader" in globals())
print("val_loader:", "val_loader" in globals())
