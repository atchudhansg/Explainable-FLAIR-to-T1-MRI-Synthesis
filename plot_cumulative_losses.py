#!/usr/bin/env python3
"""
Plot cumulative training losses across all 6 versions (V1-V6, 600 epochs total)
Replicates the style of the original GAN training loss plot
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define version paths and their corresponding epoch ranges
versions = [
    ('/Users/Work/Documents/PROJECTS/Paper Submission/final-paper-submission/outputs/resnet9_v1/resnet9/training_report.json', 0, 100),     # V1: epochs 1-100
    ('/Users/Work/Documents/PROJECTS/Paper Submission/final-paper-submission/outputs/resnet9_v2/resnet9/training_report.json', 100, 200),      # V2: epochs 101-200
    ('/Users/Work/Documents/PROJECTS/Paper Submission/final-paper-submission/outputs/resnet9_v3/resnet9/training_report.json', 200, 300),  # V3: epochs 201-300
    ('/Users/Work/Documents/PROJECTS/Paper Submission/final-paper-submission/outputs/resnet9_v4/resnet9/training_report.json', 300, 400),  # V4: epochs 301-400
    ('/Users/Work/Documents/PROJECTS/Paper Submission/final-paper-submission/outputs/resnet9_v5/resnet9/training_report.json', 400, 500),  # V5: epochs 401-500
    ('/Users/Work/Documents/PROJECTS/Paper Submission/final-paper-submission/outputs/resnet9_v6/resnet9/training_report.json', 500, 600),  # V6: epochs 501-600
]

# Aggregate losses
all_g_loss = []
all_d_loss = []
all_epochs = []

for json_path, start_epoch, end_epoch in versions:
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        history = data.get('history', {})
        g_loss = history.get('train_g_loss', [])
        d_loss = history.get('train_d_loss', [])
        
        # Take up to 100 epochs from this version
        n_epochs = min(len(g_loss), end_epoch - start_epoch)
        
        all_g_loss.extend(g_loss[:n_epochs])
        all_d_loss.extend(d_loss[:n_epochs])
        all_epochs.extend(range(start_epoch + 1, start_epoch + n_epochs + 1))
        
        print(f"✓ Loaded {n_epochs} epochs from {Path(json_path).parent.name}")
    
    except Exception as e:
        print(f"✗ Error loading {json_path}: {e}")

print(f"\n✓ Total epochs loaded: {len(all_epochs)}")

# Create the plot in the style of the original
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-darkgrid')

# Plot Generator Loss (solid blue line)
plt.plot(all_epochs, all_g_loss, 'b-', linewidth=2, label='Generator Loss')

# Plot Discriminator Loss (dashed orange line)
plt.plot(all_epochs, all_d_loss, 'orange', linestyle='--', linewidth=2, label='Discriminator Loss')

# For the third line (Discriminator T1 Loss), we'll use a smoothed version of D loss
# since we don't have separate real/fake discriminator losses in the JSON
# This approximates the original plot's appearance
d_loss_smooth = np.convolve(all_d_loss, np.ones(5)/5, mode='same')
plt.plot(all_epochs, d_loss_smooth, 'g-.', linewidth=2, label='Discriminator Loss (smoothed)', alpha=0.8)

plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('GAN Training Losses (600 Epochs across V1-V6)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 600)

# Set y-axis limit based on data
max_loss = max(max(all_g_loss[:50]), max(all_d_loss[:50]))  # Use first 50 epochs for scale
plt.ylim(0, min(max_loss * 1.1, 4.0))

plt.tight_layout()

# Save the plot
output_path = 'outputs/cumulative_loss_plot_v1_v6.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_path}")

plt.show()
print("✓ Done!")
