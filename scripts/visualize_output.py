"""
Visualize model output: FLAIR → T1 with Grad-CAM attention overlay
Single data point visualization for paper figures.
"""
import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from models import ResNet9Generator
from dataset import create_dataloaders

# ── config ───────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(ROOT_DIR, 'outputs', 'resnet9_v6', 'resnet9', 'checkpoints', 'best_gen_weights.pth')
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
CACHE_DIR  = os.path.join(ROOT_DIR, 'cache')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'samples')
device     = torch.device('cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── GradCAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)
        out.mean().backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        for i in range(cam.size(0)):
            m = cam[i].max()
            if m > 0:
                cam[i] /= m
        return cam, out

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading model from: {CHECKPOINT}")
gen  = ResNet9Generator(in_channels=3, out_channels=3).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# torch.compile saves weights with '_orig_mod.' prefix — strip it
if 'gen' in ckpt:
    state_dict = ckpt['gen']  # V6 checkpoint format
elif 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt
    
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

gen.load_state_dict(state_dict)
gen.eval()
print(f"✓ Loaded — epoch {ckpt['epoch']}, best SSIM {ckpt['best_ssim']:.4f}")

# target = last Conv2d in ResNet9Generator (output 7x7 conv, index 26)
target_layer = gen.model[26]
grad_cam = GradCAM(gen, target_layer)

# ── load single data point ────────────────────────────────────────────────────
_, val_loader, _, _ = create_dataloaders(
    DATA_DIR, batch_size=1, seed=42, num_workers=0, cache_dir=CACHE_DIR)
batch   = next(iter(val_loader))
flair   = batch['image'].to(device)
t1_real = batch['label'].to(device)
flair.requires_grad_(True)

# ── run GradCAM ───────────────────────────────────────────────────────────────
cam, t1_fake = grad_cam(flair)

def to01(t):
    return ((t.detach() + 1) / 2).clamp(0, 1)

flair_01 = to01(flair)
fake_01  = to01(t1_fake)
real_01  = to01(t1_real)

# ── extract numpy arrays ──────────────────────────────────────────────────────
f = flair_01[0, 0].cpu().numpy()
r = real_01[0, 0].cpu().numpy()
g = fake_01[0, 0].cpu().numpy()
h = cam[0, 0].cpu().numpy()

psnr = psnr_fn(r, g, data_range=1.0)
sval = ssim_fn(r, g, data_range=1.0)

print(f"✓ Metrics — PSNR: {psnr:.2f} dB, SSIM: {sval:.4f}")

# ── visualise (1 row, 5 columns) ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

axes[0].imshow(f, cmap='gray')
axes[0].set_title('FLAIR (Input)', fontsize=14, fontweight='bold')

axes[1].imshow(r, cmap='gray')
axes[1].set_title('Real T1 (Ground Truth)', fontsize=14, fontweight='bold')

axes[2].imshow(g, cmap='gray')
axes[2].set_title(f'Generated T1\nPSNR={psnr:.2f} dB  SSIM={sval:.4f}', fontsize=14, fontweight='bold')

axes[3].imshow(h, cmap='jet')
axes[3].set_title('Grad-CAM Saliency', fontsize=14, fontweight='bold')

axes[4].imshow(f, cmap='gray')
axes[4].imshow(h, cmap='jet', alpha=0.5)
axes[4].set_title('Attention Overlay', fontsize=14, fontweight='bold')

for ax in axes:
    ax.axis('off')

plt.suptitle(
    f"FLAIR → T1 Synthesis  |  ResNet-9 Generator  |  Epoch {ckpt['epoch']}",
    fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# ── save output ───────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'flair_t1_visualization_final.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved → {out_path}")

# ── also save individual images for paper flexibility ─────────────────────────
for name, img, cmap in [
    ('input_flair', f, 'gray'),
    ('real_t1', r, 'gray'),
    ('generated_t1', g, 'gray'),
    ('gradcam', h, 'jet'),
]:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# Overlay separately
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(f, cmap='gray')
ax.imshow(h, cmap='jet', alpha=0.5)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'attention_overlay.png'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"✓ Individual images saved to {OUTPUT_DIR}/")
print("Done!")
