from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SemiconductorDataset(Dataset):
    """Paired NoisyLR/GT .npy dataset used by the validated training pipeline."""
    def __init__(self, lr_dir, gt_dir):
        self.lr_dir=Path(lr_dir); self.gt_dir=Path(gt_dir)
        self.names=sorted([p.stem for p in self.lr_dir.glob('*.npy') if (self.gt_dir/p.name).exists()])
    def __len__(self): return len(self.names)
    def __getitem__(self,idx):
        name=self.names[idx]
        lr=torch.from_numpy(np.load(self.lr_dir/f'{name}.npy')).float().unsqueeze(0)
        gt=torch.from_numpy(np.load(self.gt_dir/f'{name}.npy')).float().unsqueeze(0)
        return lr,gt


def make_loaders(lr_dir, gt_dir, batch_size=8, val_fraction=0.1, seed=42, num_workers=2):
    dataset=SemiconductorDataset(lr_dir,gt_dir)
    train_size=int((1-val_fraction)*len(dataset)); val_size=len(dataset)-train_size
    generator=torch.Generator().manual_seed(seed)
    train_ds,val_ds=random_split(dataset,[train_size,val_size],generator=generator)
    kwargs=dict(batch_size=batch_size,num_workers=num_workers,pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds,shuffle=True,**kwargs), DataLoader(val_ds,shuffle=False,**kwargs)
