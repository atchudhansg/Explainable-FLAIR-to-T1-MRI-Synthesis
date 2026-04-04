"""
Generate Grad-CAM Heatmap for Paper Figure
Produces a single heatmap image with colorbar like Fig. 3 in the paper.
"""
import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from models import ResNet9Generator
from dataset import create_dataloaders

# ── config ───────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(ROOT_DIR, 'outputs', 'resnet9_v6', 'resnet9', 'checkpoints', 'best_model.pth')
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
CACHE_DIR  = os.path.join(ROOT_DIR, 'cache')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'outputs', 'samples', 'gradcam_saliency.png')
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

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
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam, out

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading model from: {CHECKPOINT}")
gen  = ResNet9Generator(in_channels=3, out_channels=3).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

state_dict = ckpt['gen']
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

gen.load_state_dict(state_dict)
gen.eval()
print(f"✓ Loaded — epoch {ckpt['epoch']}, best SSIM {ckpt['best_ssim']:.4f}")

# target = last Conv2d in ResNet9Generator (output 7x7 conv, index 26)
target_layer = gen.model[26]
grad_cam = GradCAM(gen, target_layer)

# ── load single data point ────────────────────────────────────────────────────
# Try different samples - index 0-249 available in validation set
SAMPLE_INDEX = 15  # Change this to try different samples

_, val_loader, _, _ = create_dataloaders(
    DATA_DIR, batch_size=1, seed=42, num_workers=0, cache_dir=CACHE_DIR)

# Skip to desired sample
for i, batch in enumerate(val_loader):
    if i == SAMPLE_INDEX:
        break

flair = batch['image'].to(device)
flair.requires_grad_(True)

print(f"✓ Using validation sample #{SAMPLE_INDEX}")

# ── run GradCAM ───────────────────────────────────────────────────────────────
cam, _ = grad_cam(flair)
heatmap = cam[0, 0].cpu().numpy()

# Upscale to 4K resolution (3840x3840) using high-quality interpolation
from PIL import Image
heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
heatmap_4k = heatmap_pil.resize((2048, 2048), Image.LANCZOS)
heatmap = np.array(heatmap_4k).astype(np.float32) / 255.0

print(f"✓ Grad-CAM computed — shape: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

# ── create figure like paper (with colorbar) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 12))

im = ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1, interpolation='nearest')
ax.set_title('GradCAM Heatmap', fontsize=18, fontweight='bold')
ax.axis('off')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved → {OUTPUT_PATH}")
print("Done!")
