"""
Grad-CAM analysis + comparative study evaluation.
Run after training all models.

Usage:
  python evaluate.py --model resnet9 --checkpoint outputs/resnet9/checkpoints/best_model.pth
  python evaluate.py --compare   # Runs comparative study across all trained models
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from pytorch_msssim import ssim as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import get_model_pair, count_parameters
from dataset import create_dataloaders, create_brats2023_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='resnet9', choices=['resnet9', 'unet'])
    p.add_argument('--checkpoint', type=str, default='')
    p.add_argument('--data_dir', type=str, default='/home/atchu2504/training/data')
    p.add_argument('--output_dir', type=str, default='/home/atchu2504/training/outputs')
    p.add_argument('--compare', action='store_true', help='Run comparative study')
    p.add_argument('--external_val', action='store_true',
                   help='Run external validation on BraTS 2023')
    p.add_argument('--external_val_dir', type=str,
                   default='/home/atchu2504/training/validation',
                   help='Path to BraTS 2023 GLI Challenge data')
    p.add_argument('--gradcam_n', type=int, default=100, help='Number of Grad-CAM samples')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """Grad-CAM for the generator's last conv layer."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Find the last Conv2d in the generator
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        if target_layer is None:
            raise ValueError("No Conv2d found in model")

        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        # Use mean of output as target for backprop
        target = output.mean()
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None, output

        # Grad-CAM computation
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        # Upsample to input size
        cam = nn.functional.interpolate(cam, size=input_tensor.shape[2:],
                                         mode='bilinear', align_corners=False)
        return cam, output


def compute_gradcam_metrics(cam_tensor):
    """Compute interpretability metrics for a single Grad-CAM heatmap."""
    cam = cam_tensor.squeeze().cpu().numpy()
    cam_flat = cam.flatten()

    # Entropy
    cam_prob = cam_flat / (cam_flat.sum() + 1e-8)
    cam_prob = cam_prob[cam_prob > 0]
    entropy = -np.sum(cam_prob * np.log2(cam_prob + 1e-10))

    # Spatial variance
    spatial_var = np.var(cam_flat)

    # Peak activation
    peak_act = cam.max()

    # High activation ratio (>0.5 threshold)
    high_ratio = (cam > 0.5).sum() / cam.size

    # Mean and std
    mean_act = cam.mean()
    std_act = cam.std()

    return {
        'entropy': entropy,
        'spatial_variance': spatial_var,
        'peak_activation': peak_act,
        'high_activation_ratio': high_ratio,
        'mean_activation': mean_act,
        'std_activation': std_act,
    }


def run_gradcam_analysis(args):
    """Full Grad-CAM analysis on n samples."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen, _ = get_model_pair(args.model)
    ckpt_path = args.checkpoint or os.path.join(args.output_dir, args.model,
                                                 'checkpoints', 'best_model.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(ckpt['gen'])
    gen = gen.to(device)
    gen.eval()

    _, val_loader, _, _ = create_dataloaders(
        args.data_dir, batch_size=1, seed=args.seed, num_workers=2
    )

    gradcam = GradCAM(gen)
    all_metrics = []
    save_dir = os.path.join(args.output_dir, args.model, 'gradcam')
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for batch in tqdm(val_loader, desc="Grad-CAM Analysis", total=args.gradcam_n):
        if count >= args.gradcam_n:
            break
        flair = batch['image'].to(device)
        flair.requires_grad_(True)
        t1_real = batch['label'].to(device)

        cam, fake_t1 = gradcam(flair)
        if cam is None:
            continue

        metrics = compute_gradcam_metrics(cam)

        # Error correlation
        error_map = torch.abs((fake_t1.detach() + 1) / 2 - (t1_real + 1) / 2)
        error_flat = error_map.mean(dim=1).squeeze().cpu().numpy().flatten()
        cam_flat = cam.squeeze().cpu().numpy().flatten()
        if len(error_flat) == len(cam_flat):
            corr = np.corrcoef(cam_flat, error_flat)[0, 1]
            metrics['error_correlation'] = corr if not np.isnan(corr) else 0.0

        all_metrics.append(metrics)

        # Save a few sample visualizations
        if count < 10:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(flair[0, 0].detach().cpu().numpy(), cmap='gray')
            axes[0].set_title('FLAIR Input'); axes[0].axis('off')
            axes[1].imshow(t1_real[0, 0].cpu().numpy(), cmap='gray')
            axes[1].set_title('Real T1'); axes[1].axis('off')
            axes[2].imshow(fake_t1[0, 0].detach().cpu().numpy(), cmap='gray')
            axes[2].set_title('Generated T1'); axes[2].axis('off')
            axes[3].imshow(flair[0, 0].detach().cpu().numpy(), cmap='gray')
            axes[3].imshow(cam[0, 0].cpu().numpy(), cmap='jet', alpha=0.5)
            axes[3].set_title('Grad-CAM Overlay'); axes[3].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'gradcam_sample_{count}.png'), dpi=150)
            plt.close()

        count += 1

    # Aggregate metrics
    print(f"\n{'='*60}")
    print(f"  Grad-CAM Interpretability Metrics (n={len(all_metrics)})")
    print(f"{'='*60}")

    summary = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        mean = np.mean(values)
        std = np.std(values)
        summary[key] = {'mean': mean, 'std': std}
        print(f"  {key}: {mean:.4f} ± {std:.4f}")

    with open(os.path.join(save_dir, 'gradcam_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {save_dir}/")
    return summary


# ============================================================
# COMPARATIVE STUDY
# ============================================================

def run_comparative_study(args):
    """Compare all trained models on the same validation set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader, _, _ = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed, num_workers=4
    )

    results = {}
    models_to_compare = ['resnet9', 'unet']

    for model_name in models_to_compare:
        ckpt_path = os.path.join(args.output_dir, model_name, 'checkpoints', 'best_model.pth')
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {model_name}: no checkpoint found at {ckpt_path}")
            continue

        gen, _ = get_model_pair(model_name)
        ckpt = torch.load(ckpt_path, map_location=device)
        gen.load_state_dict(ckpt['gen'])
        gen = gen.to(device)
        gen.eval()

        # Load training report for timing info
        report_path = os.path.join(args.output_dir, model_name, 'training_report.json')
        training_time = 0
        peak_gpu = 0
        if os.path.exists(report_path):
            with open(report_path) as f:
                rpt = json.load(f)
                training_time = rpt.get('total_training_time_hours', 0)
                peak_gpu = rpt.get('peak_gpu_memory_gb', 0)

        all_psnr, all_ssim, all_mae, all_rmse = [], [], [], []
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

        for batch in tqdm(val_loader, desc=f"  Evaluating {model_name}"):
            flair = batch['image'].to(device, non_blocking=True)
            t1_real = batch['label'].to(device, non_blocking=True)
            with torch.no_grad(), autocast():
                fake_t1 = gen(flair)
            gen_01 = (fake_t1.float() + 1) / 2.0
            tgt_01 = (t1_real.float() + 1) / 2.0
            ssim_v = compute_ssim(gen_01, tgt_01, data_range=1.0, size_average=True).item()
            gen_np = gen_01.cpu().numpy()
            tgt_np = tgt_01.cpu().numpy()
            for i in range(gen_np.shape[0]):
                all_psnr.append(psnr(tgt_np[i], gen_np[i], data_range=1.0))
            all_ssim.append(ssim_v)
            all_mae.append(torch.mean(torch.abs(gen_01 - tgt_01)).item())
            all_rmse.append(torch.sqrt(torch.mean((gen_01 - tgt_01)**2)).item())

            pred_pos = (gen_01 > 0.1).float()
            true_pos = (tgt_01 > 0.1).float()
            total_tp += (pred_pos * true_pos).sum().item()
            total_fp += (pred_pos * (1 - true_pos)).sum().item()
            total_fn += ((1 - pred_pos) * true_pos).sum().item()
            total_tn += ((1 - pred_pos) * (1 - true_pos)).sum().item()

        prec = total_tp / (total_tp + total_fp + 1e-8)
        rec = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        label_map = {'resnet9': 'Proposed (ResNet-9)', 'unet': 'Pix2Pix (U-Net)'}
        results[model_name] = {
            'label': label_map.get(model_name, model_name),
            'psnr': np.mean(all_psnr), 'ssim': np.mean(all_ssim),
            'mae': np.mean(all_mae), 'rmse': np.mean(all_rmse),
            'precision': prec, 'recall': rec, 'f1': f1,
            'params_M': count_parameters(gen) / 1e6,
            'training_time_h': training_time, 'peak_gpu_gb': peak_gpu,
            'interpretability': 'Grad-CAM' if model_name == 'resnet9' else 'None',
        }

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  COMPARATIVE STUDY")
    print(f"{'='*80}")
    header = f"{'Method':<25} {'SSIM':>8} {'PSNR':>8} {'MAE':>8} {'RMSE':>8} {'F1':>8} {'Params':>8} {'XAI':>10}"
    print(header)
    print("-" * 80)
    for name, r in results.items():
        print(f"{r['label']:<25} {r['ssim']:>8.4f} {r['psnr']:>8.2f} "
              f"{r['mae']:>8.4f} {r['rmse']:>8.4f} {r['f1']:>8.4f} "
              f"{r['params_M']:>7.2f}M {r['interpretability']:>10}")
    print("="*80)

    # Save
    comp_path = os.path.join(args.output_dir, 'comparative_study.json')
    with open(comp_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {comp_path}")

    # Comparison bar chart
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    names = [r['label'] for r in results.values()]
    for ax, metric, title in zip(axes,
        ['ssim', 'psnr', 'mae', 'f1'],
        ['SSIM ↑', 'PSNR (dB) ↑', 'MAE ↓', 'F1 Score ↑']):
        vals = [r[metric] for r in results.values()]
        colors = ['#4CAF50' if 'Proposed' in n else '#2196F3' for n in names]
        ax.bar(names, vals, color=colors)
        ax.set_title(title); ax.set_ylabel(title.split(' ')[0])
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'comparative_chart.png'), dpi=200)
    plt.close()
    return results


# ============================================================
# EXTERNAL VALIDATION: BraTS 2023
# ============================================================

def _compute_val_metrics(gen, loader, device):
    """Shared validation loop returning all per-sample lists."""
    gen.eval()
    all_psnr, all_ssim, all_mae, all_rmse, all_mse = [], [], [], [], []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for batch in tqdm(loader, desc="  Evaluating"):
        flair = batch['image'].to(device, non_blocking=True)
        t1_real = batch['label'].to(device, non_blocking=True)
        with torch.no_grad(), autocast():
            fake_t1 = gen(flair)

        gen_01 = (fake_t1.float() + 1) / 2.0
        tgt_01 = (t1_real.float() + 1) / 2.0

        from pytorch_msssim import ssim as compute_ssim_fn
        ssim_v = compute_ssim_fn(gen_01, tgt_01, data_range=1.0, size_average=True).item()

        gen_np = gen_01.cpu().numpy()
        tgt_np = tgt_01.cpu().numpy()
        for i in range(gen_np.shape[0]):
            all_psnr.append(psnr(tgt_np[i], gen_np[i], data_range=1.0))
        all_ssim.append(ssim_v)
        all_mae.append(torch.mean(torch.abs(gen_01 - tgt_01)).item())
        all_rmse.append(torch.sqrt(torch.mean((gen_01 - tgt_01) ** 2)).item())
        all_mse.append(torch.mean((gen_01 - tgt_01) ** 2).item())

        pred_pos = (gen_01 > 0.1).float()
        true_pos = (tgt_01 > 0.1).float()
        total_tp += (pred_pos * true_pos).sum().item()
        total_fp += (pred_pos * (1 - true_pos)).sum().item()
        total_fn += ((1 - pred_pos) * true_pos).sum().item()
        total_tn += ((1 - pred_pos) * (1 - true_pos)).sum().item()

    prec = total_tp / (total_tp + total_fp + 1e-8)
    rec = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return {
        'psnr': np.mean(all_psnr), 'ssim': np.mean(all_ssim),
        'mae': np.mean(all_mae), 'rmse': np.mean(all_rmse), 'mse': np.mean(all_mse),
        'precision': prec, 'recall': rec, 'f1': f1,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn,
        'psnr_all': all_psnr, 'ssim_all': all_ssim,
        'mae_all': all_mae, 'rmse_all': all_rmse,
    }


def _bootstrap_ci(values, n_boot=1000, ci=0.95):
    values = np.array(values)
    boot_means = [np.mean(np.random.choice(values, size=len(values), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(values), lo, hi


def run_external_validation(args):
    """
    Evaluate the trained model on BraTS 2023 GLI Challenge data.
    Produces full metrics table, bootstrap CIs, confusion matrix plot,
    and a side-by-side comparison chart vs. BraTS 2021 internal results
    (if training_report.json is available).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join(args.output_dir, args.model, 'external_validation')
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    gen, _ = get_model_pair(args.model)
    ckpt_path = args.checkpoint or os.path.join(
        args.output_dir, args.model, 'checkpoints', 'best_model.pth'
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(ckpt['gen'])
    gen = gen.to(device)
    gen.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # BraTS 2023 loader
    ext_loader = create_brats2023_loader(
        args.external_val_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )
    n_subjects = len(ext_loader.dataset)
    print(f"External validation subjects: {n_subjects}")

    # Evaluate
    results = _compute_val_metrics(gen, ext_loader, device)

    # Bootstrap CIs
    print(f"\n{'='*60}")
    print(f"  EXTERNAL VALIDATION — BraTS 2023 GLI (n={n_subjects})")
    print(f"{'='*60}")
    ci_results = {}
    for metric_name, values in [
        ('PSNR', results['psnr_all']),
        ('SSIM', results['ssim_all']),
        ('MAE',  results['mae_all']),
        ('RMSE', results['rmse_all']),
    ]:
        mean, lo, hi = _bootstrap_ci(values, n_boot=1000)
        ci_results[metric_name] = {'mean': mean, 'ci_low': lo, 'ci_high': hi}
        print(f"  {metric_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\n  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"  MSE:       {results['mse']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {results['tp']:.0f}  FP: {results['fp']:.0f}")
    print(f"    FN: {results['fn']:.0f}  TN: {results['tn']:.0f}")
    print(f"{'='*60}")

    # Save JSON
    report = {
        'model': args.model,
        'checkpoint': ckpt_path,
        'n_subjects': n_subjects,
        'dataset': 'BraTS 2023 GLI Challenge',
        'metrics': {
            'psnr': results['psnr'], 'ssim': results['ssim'],
            'mae': results['mae'], 'rmse': results['rmse'], 'mse': results['mse'],
            'precision': results['precision'],
            'recall': results['recall'], 'f1': results['f1'],
        },
        'confusion_matrix': {
            'tp': results['tp'], 'fp': results['fp'],
            'fn': results['fn'], 'tn': results['tn'],
        },
        'bootstrap_ci_95': ci_results,
    }
    with open(os.path.join(save_dir, 'external_validation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Confusion matrix plot
    cm = np.array([[results['tn'], results['fp']],
                   [results['fn'], results['tp']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nHealthy', 'Predicted\nLesion'])
    ax.set_yticklabels(['Actual\nHealthy', 'Actual\nLesion'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,.0f}", ha='center', va='center',
                    fontsize=12, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title('Confusion Matrix — BraTS 2023 (Pixel-level)')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_brats2023.png'), dpi=200)
    plt.close()

    # Compare internal (BraTS 2021) vs external (BraTS 2023) if report available
    internal_report_path = os.path.join(args.output_dir, args.model, 'training_report.json')
    if os.path.exists(internal_report_path):
        with open(internal_report_path) as f:
            int_rpt = json.load(f)
        int_metrics = int_rpt.get('final_metrics', {})
        metrics_to_plot = ['psnr', 'ssim', 'mae', 'rmse']
        labels = ['PSNR (dB)', 'SSIM', 'MAE', 'RMSE']
        int_vals = [int_metrics.get(m, 0) for m in metrics_to_plot]
        ext_vals = [results[m] for m in metrics_to_plot]

        x = np.arange(len(metrics_to_plot))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width/2, int_vals, width, label='BraTS 2021 (Internal)', color='#2196F3')
        ax.bar(x + width/2, ext_vals, width, label='BraTS 2023 (External)', color='#4CAF50')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title('Internal vs External Validation — Proposed Method')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        for i, (iv, ev) in enumerate(zip(int_vals, ext_vals)):
            ax.text(i - width/2, iv, f'{iv:.4f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, ev, f'{ev:.4f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'internal_vs_external.png'), dpi=200)
        plt.close()
        print(f"  Comparison plot saved.")

    print(f"\nExternal validation results saved to {save_dir}/")
    return report


if __name__ == '__main__':
    args = parse_args()
    if args.compare:
        run_comparative_study(args)
    elif args.external_val:
        run_external_validation(args)
    else:
        run_gradcam_analysis(args)
