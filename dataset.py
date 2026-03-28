"""
Data pipeline for BraTS 2021 FLAIR-to-T1 synthesis.
Optimized for NVIDIA L4 (23GB VRAM).
Paper-compliant: MONAI ExtractMidSlice, 256x256, [-1,1] normalization, 80/20 split.
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, Resized, MapTransform
)
from monai.data import PersistentDataset as MONAIPersistentDataset
from monai.utils import set_determinism


class ExtractMidSlice(MapTransform):
    """Extract the central axial slice from a 3D volume -> 2D."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            mid_idx = img.shape[-1] // 2
            d[key] = img[..., mid_idx]
        return d


class NormalizeToRange(MapTransform):
    """Normalize to [-1, 1] range for stable GAN training (paper-compliant)."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            vmin, vmax = img.min(), img.max()
            if vmax - vmin > 0:
                d[key] = 2.0 * (img - vmin) / (vmax - vmin) - 1.0
            else:
                d[key] = torch.zeros_like(img)
        return d


class RepeatChannel(MapTransform):
    """Repeat single channel to 3 channels for ResNet/VGG compatibility."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if img.shape[0] == 1:
                d[key] = img.repeat(3, 1, 1)
        return d


class ToFloat32(MapTransform):
    """Ensure float32 tensors."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key], dtype=torch.float32)
            else:
                d[key] = d[key].float()
        return d


def _get_sample_list(root_dir):
    """Return list of {image, label} dicts for all valid FLAIR/T1 pairs."""
    samples = []
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Data directory not found: {root_dir}")
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        flair = os.path.join(folder_path, f"{folder}_flair.nii.gz")
        t1 = os.path.join(folder_path, f"{folder}_t1.nii.gz")
        if os.path.exists(flair) and os.path.exists(t1):
            samples.append({"image": flair, "label": t1})
    return samples


class BraTSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._find_valid_samples()

    def _find_valid_samples(self):
        samples = []
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")
        for folder in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            flair = os.path.join(folder_path, f"{folder}_flair.nii.gz")
            t1 = os.path.join(folder_path, f"{folder}_t1.nii.gz")
            if os.path.exists(flair) and os.path.exists(t1):
                samples.append({"image": flair, "label": t1, "subject_id": folder})
        print(f"Found {len(samples)} valid FLAIR/T1 pairs in {self.root_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {"image": self.samples[idx]["image"],
                  "label": self.samples[idx]["label"]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_transforms():
    """Paper-compliant transform pipeline."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0),
                 mode=("bilinear", "bilinear")),
        ExtractMidSlice(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=(256, 256),
                mode=("bilinear", "bilinear")),
        ToFloat32(keys=["image", "label"]),
        NormalizeToRange(keys=["image", "label"]),
        RepeatChannel(keys=["image", "label"]),
    ])


def create_dataloaders(root_dir, batch_size=4, seed=42, num_workers=4,
                       val_ratio=0.2, cache_dir=None):
    """
    Create train/val dataloaders with deterministic 80/20 split.

    cache_dir (str|None): If set, uses MONAI PersistentDataset — preprocesses
    each NIfTI once and caches tensors to disk. From epoch 2 onward, data
    loading is near-instant (reads cached .pt files instead of full 3D NIfTI
    resampling). Recommended for multi-epoch training on L4.
    First run will be slower while the cache is being built.
    """
    set_determinism(seed=seed)
    transforms = get_transforms()

    samples = _get_sample_list(root_dir)
    n = len(samples)
    n_val = int(n * val_ratio)
    n_train = n - n_val

    # Deterministic split
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    if cache_dir:
        # PersistentDataset: cache preprocessed tensors to disk.
        # Survives restarts — cache is reused across all runs.
        train_cache = os.path.join(cache_dir, 'train')
        val_cache = os.path.join(cache_dir, 'val')
        os.makedirs(train_cache, exist_ok=True)
        os.makedirs(val_cache, exist_ok=True)
        print(f"PersistentDataset cache: {cache_dir}")
        print(f"  First run: builds cache (slow). Subsequent epochs: instant reads.")
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        train_ds = MONAIPersistentDataset(
            data=train_samples, transform=transforms, cache_dir=train_cache
        )
        val_ds = MONAIPersistentDataset(
            data=val_samples, transform=transforms, cache_dir=val_cache
        )
    else:
        dataset = BraTSDataset(root_dir=root_dir, transform=transforms)
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)

    print(f"Split: {n_train} train / {n_val} val (total {n})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader, train_indices, val_indices


class BraTS2023Dataset(Dataset):
    """
    BraTS 2023 GLI Challenge dataset for external validation.
    File naming convention (differs from BraTS 2021):
      {subject}-t2f.nii.gz  → FLAIR input
      {subject}-t1n.nii.gz  → T1 native target
    Directory layout expected:
      root_dir/
        BraTS-GLI-XXXXX-XXX/
          BraTS-GLI-XXXXX-XXX-t2f.nii.gz
          BraTS-GLI-XXXXX-XXX-t1n.nii.gz
          ...
    root_dir can also be the parent of the ASNR-MICCAI-... folder;
    the class will descend automatically.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._find_valid_samples()

    def _resolve_subjects_dir(self):
        """Walk one level deeper if subjects aren't at root_dir directly."""
        candidates = [self.root_dir]
        for entry in sorted(os.listdir(self.root_dir)):
            full = os.path.join(self.root_dir, entry)
            if os.path.isdir(full):
                candidates.append(full)
                break  # only try first subdir
        for path in candidates:
            # Check whether any subfolder has a -t2f file
            for sub in sorted(os.listdir(path))[:5]:
                sub_path = os.path.join(path, sub)
                if os.path.isdir(sub_path):
                    probe = os.path.join(sub_path, f"{sub}-t2f.nii.gz")
                    if os.path.exists(probe):
                        return path
        return self.root_dir  # fallback

    def _find_valid_samples(self):
        subjects_dir = self._resolve_subjects_dir()
        samples = []
        for folder in sorted(os.listdir(subjects_dir)):
            folder_path = os.path.join(subjects_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            flair = os.path.join(folder_path, f"{folder}-t2f.nii.gz")
            t1 = os.path.join(folder_path, f"{folder}-t1n.nii.gz")
            if os.path.exists(flair) and os.path.exists(t1):
                samples.append({"image": flair, "label": t1, "subject_id": folder})
        print(f"Found {len(samples)} valid FLAIR/T1 pairs in BraTS 2023 ({subjects_dir})")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {"image": self.samples[idx]["image"],
                  "label": self.samples[idx]["label"]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def create_brats2023_loader(root_dir, batch_size=4, num_workers=4):
    """
    Dataloader for BraTS 2023 GLI Challenge external validation.
    Uses the same get_transforms() pipeline as BraTS 2021 for consistency.
    """
    dataset = BraTS2023Dataset(root_dir=root_dir, transform=get_transforms())
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader


if __name__ == '__main__':
    root = "/home/atchu2504/training/data"
    tl, vl, _, _ = create_dataloaders(root, batch_size=4, num_workers=0)
    batch = next(iter(tl))
    print(f"Image: {batch['image'].shape}, range [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
    print(f"Label: {batch['label'].shape}, range [{batch['label'].min():.2f}, {batch['label'].max():.2f}]")
    print(f"Train batches: {len(tl)}, Val batches: {len(vl)}")

    # Test BraTS 2023 loader
    val23_root = "/home/atchu2504/training/validation"
    loader23 = create_brats2023_loader(val23_root, batch_size=2, num_workers=0)
    batch23 = next(iter(loader23))
    print(f"\nBraTS 2023 — Image: {batch23['image'].shape}, "
          f"range [{batch23['image'].min():.2f}, {batch23['image'].max():.2f}]")
