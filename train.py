"""
FLAIR-to-T1 MRI Synthesis — Full Training Pipeline
Paper: "Explainable FLAIR-to-T1 MRI Synthesis: Interpretable Residual GANs"
Optimized for NVIDIA L4 (23GB VRAM)

Usage:
  python train.py --model resnet9 --epochs 20 --batch_size 4
  python train.py --model unet --epochs 20 --batch_size 4    # Vanilla Pix2Pix comparison
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from pytorch_msssim import ssim as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import get_model_pair, count_parameters
from dataset import create_dataloaders, create_brats2023_loader

# ============================================================
# CONFIG
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='resnet9', choices=['resnet9', 'unet'])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--beta1', type=float, default=0.5)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--lambda_l1', type=float, default=100.0)
    p.add_argument('--lambda_ssim', type=float, default=10.0)
    p.add_argument('--data_dir', type=str, default='/home/atchu2504/training/data')
    p.add_argument('--output_dir', type=str, default='/home/atchu2504/training/outputs')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_every', type=int, default=5)
    # torch.compile (PyTorch 2.x, ~20-30% GPU throughput gain)
    p.add_argument('--compile', action='store_true',
                   help='Compile models with torch.compile for faster GPU execution')
    # Resume from checkpoint
    p.add_argument('--resume', action='store_true',
                   help='Resume from latest periodic checkpoint in output_dir/model/checkpoints/')
    # PersistentDataset cache (big speedup from epoch 2 onward)
    p.add_argument('--cache_dir', type=str,
                   default='/home/atchu2504/training/cache',
                   help='Directory for MONAI PersistentDataset cache. Set to "" to disable.')
    # BraTS 2023 external validation (separate from internal BraTS 2021 val split)
    p.add_argument('--external_val_dir', type=str,
                   default='/home/atchu2504/training/validation',
                   help='Path to BraTS 2023 GLI Challenge data for external validation')
    p.add_argument('--skip_external_val', action='store_true',
                   help='Skip external BraTS 2023 validation')
    return p.parse_args()


# ============================================================
# METRICS
# ============================================================

def compute_metrics_batch(generated, target):
    """Compute PSNR, SSIM, MAE, RMSE on a batch. Images in [-1,1]."""
    # Rescale to [0,1] for metrics
    gen_01 = (generated + 1.0) / 2.0
    tgt_01 = (target + 1.0) / 2.0

    # SSIM (pytorch_msssim works on [0,1])
    ssim_val = compute_ssim(gen_01, tgt_01, data_range=1.0, size_average=True).item()

    # Per-image PSNR
    gen_np = gen_01.detach().cpu().numpy()
    tgt_np = tgt_01.detach().cpu().numpy()
    psnr_vals = []
    for i in range(gen_np.shape[0]):
        p = psnr(tgt_np[i], gen_np[i], data_range=1.0)
        psnr_vals.append(p)

    mae = torch.mean(torch.abs(gen_01 - tgt_01)).item()
    rmse = torch.sqrt(torch.mean((gen_01 - tgt_01) ** 2)).item()
    mse = torch.mean((gen_01 - tgt_01) ** 2).item()

    return {
        'psnr': np.mean(psnr_vals),
        'ssim': ssim_val,
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'psnr_list': psnr_vals,
    }


def compute_pixel_classification(generated, target, threshold=0.1):
    """Compute precision, recall, F1 for pixel-level classification."""
    gen_01 = (generated + 1.0) / 2.0
    tgt_01 = (target + 1.0) / 2.0
    pred_pos = (gen_01 > threshold).float()
    true_pos_mask = (tgt_01 > threshold).float()

    tp = (pred_pos * true_pos_mask).sum().item()
    fp = (pred_pos * (1 - true_pos_mask)).sum().item()
    fn = ((1 - pred_pos) * true_pos_mask).sum().item()
    tn = ((1 - pred_pos) * (1 - true_pos_mask)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    """Compute bootstrap 95% confidence interval."""
    values = np.array(values)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(values), lower, upper


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(gen, disc, train_loader, opt_g, opt_d, scaler_g, scaler_d,
                    criterion_gan, lambda_l1, lambda_ssim, device, epoch):
    gen.train(); disc.train()
    metrics = {'g_loss': [], 'd_loss': [], 'g_adv': [], 'g_l1': [], 'g_ssim': []}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch in pbar:
        flair = batch['image'].to(device, non_blocking=True)
        t1_real = batch['label'].to(device, non_blocking=True)

        # --------------- Discriminator ---------------
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            fake_t1 = gen(flair)
            # Real
            pred_real = disc(flair, t1_real)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            # Fake
            pred_fake = disc(flair, fake_t1.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5

        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # --------------- Generator ---------------
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            pred_fake_g = disc(flair, fake_t1)
            loss_g_adv = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))
            loss_g_l1 = nn.L1Loss()(fake_t1, t1_real)
            # SSIM loss (needs [0,1])
            gen_01 = (fake_t1 + 1) / 2.0
            tgt_01 = (t1_real + 1) / 2.0
            ssim_val = compute_ssim(gen_01, tgt_01, data_range=1.0, size_average=True)
            loss_g_ssim = 1.0 - ssim_val
            loss_g = loss_g_adv + lambda_l1 * loss_g_l1 + lambda_ssim * loss_g_ssim

        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        metrics['g_loss'].append(loss_g.item())
        metrics['d_loss'].append(loss_d.item())
        metrics['g_adv'].append(loss_g_adv.item())
        metrics['g_l1'].append(loss_g_l1.item())
        metrics['g_ssim'].append(loss_g_ssim.item())

        pbar.set_postfix(G=f"{loss_g.item():.4f}", D=f"{loss_d.item():.4f}")

    return {k: np.mean(v) for k, v in metrics.items()}


@torch.no_grad()
def validate(gen, val_loader, device):
    gen.eval()
    all_psnr, all_ssim, all_mae, all_rmse, all_mse = [], [], [], [], []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for batch in tqdm(val_loader, desc="  [Val]", leave=False):
        flair = batch['image'].to(device, non_blocking=True)
        t1_real = batch['label'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            fake_t1 = gen(flair)

        m = compute_metrics_batch(fake_t1.float(), t1_real.float())
        all_psnr.extend(m['psnr_list'])
        all_ssim.append(m['ssim'])
        all_mae.append(m['mae'])
        all_rmse.append(m['rmse'])
        all_mse.append(m['mse'])

        cls = compute_pixel_classification(fake_t1.float(), t1_real.float())
        total_tp += cls['tp']; total_fp += cls['fp']
        total_fn += cls['fn']; total_tn += cls['tn']

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'psnr': np.mean(all_psnr), 'ssim': np.mean(all_ssim),
        'mae': np.mean(all_mae), 'rmse': np.mean(all_rmse),
        'mse': np.mean(all_mse),
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn,
        'psnr_all': all_psnr, 'ssim_all': all_ssim,
        'mae_all': all_mae, 'rmse_all': all_rmse,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # speeds up fixed-size input training
    print(f"\n{'='*60}")
    print(f"  FLAIR -> T1 Synthesis Training")
    print(f"  Model: {args.model.upper()}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Lambda L1: {args.lambda_l1}, Lambda SSIM: {args.lambda_ssim}")
    print(f"{'='*60}\n")

    # Output dirs
    run_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'samples'), exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(os.path.join(run_dir, 'tb_logs'))

    # Data — use PersistentDataset cache if cache_dir is set
    cache_dir = args.cache_dir if args.cache_dir else None
    train_loader, val_loader, train_idx, val_idx = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed,
        num_workers=args.num_workers, cache_dir=cache_dir
    )

    # Models
    gen, disc = get_model_pair(args.model)
    gen, disc = gen.to(device), disc.to(device)
    print(f"Generator params: {count_parameters(gen)/1e6:.2f}M")
    print(f"Discriminator params: {count_parameters(disc)/1e6:.2f}M")

    # torch.compile: fuses ops, gives 20-30% speedup on PyTorch 2.x (L4/Ampere)
    if args.compile:
        print("  Compiling models with torch.compile (one-time warmup on first batch)...")
        gen = torch.compile(gen)
        disc = torch.compile(disc)
        print("  Compilation ready.\n")
    else:
        print()

    # Optimizers (paper-compliant)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Mixed precision scalers
    scaler_g = GradScaler('cuda')
    scaler_d = GradScaler('cuda')

    criterion_gan = nn.BCEWithLogitsLoss()

    # Training history
    history = {
        'train_g_loss': [], 'train_d_loss': [],
        'train_g_adv': [], 'train_g_l1': [], 'train_g_ssim': [],
        'val_psnr': [], 'val_ssim': [], 'val_mae': [], 'val_rmse': [],
        'val_mse': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
    }

    best_ssim = 0
    start_epoch = 0

    # ---- Resume from checkpoint ----
    if args.resume:
        import glob as _glob
        ckpt_dir = os.path.join(run_dir, 'checkpoints')
        periodic = sorted(_glob.glob(os.path.join(ckpt_dir, 'epoch_*.pth')))
        resume_path = periodic[-1] if periodic else os.path.join(ckpt_dir, 'best_model.pth')
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            gen.load_state_dict(ckpt['gen'])
            disc.load_state_dict(ckpt['disc'])
            if 'opt_g' in ckpt:
                opt_g.load_state_dict(ckpt['opt_g'])
            if 'opt_d' in ckpt:
                opt_d.load_state_dict(ckpt['opt_d'])
            start_epoch = ckpt['epoch'] + 1
            best_ssim = ckpt.get('best_ssim', 0)
            # Reload history if report exists
            report_path = os.path.join(run_dir, 'training_report.json')
            if os.path.exists(report_path):
                with open(report_path) as f:
                    prev = json.load(f)
                history = prev.get('history', history)
            print(f"\nResumed from {resume_path}")
            print(f"  Starting at epoch {start_epoch + 1}, best SSIM so far: {best_ssim:.4f}\n")
        else:
            print(f"WARNING: --resume set but no checkpoint found at {ckpt_dir}. Starting fresh.")

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_m = train_one_epoch(
            gen, disc, train_loader, opt_g, opt_d, scaler_g, scaler_d,
            criterion_gan, args.lambda_l1, args.lambda_ssim, device, epoch
        )

        # Validate
        val_m = validate(gen, val_loader, device)

        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        # Log
        for k in ['g_loss', 'd_loss', 'g_adv', 'g_l1', 'g_ssim']:
            history[f'train_{k}'].append(train_m[k])
        for k in ['psnr', 'ssim', 'mae', 'rmse', 'mse', 'precision', 'recall', 'f1']:
            history[f'val_{k}'].append(val_m[k])

        # TensorBoard
        writer.add_scalars('Loss/Train', {
            'G_total': train_m['g_loss'], 'D': train_m['d_loss'],
            'G_adv': train_m['g_adv'], 'G_L1': train_m['g_l1'],
            'G_SSIM': train_m['g_ssim'],
        }, epoch)
        writer.add_scalars('Metrics/Val', {
            'PSNR': val_m['psnr'], 'SSIM': val_m['ssim'],
            'MAE': val_m['mae'], 'RMSE': val_m['rmse'],
        }, epoch)
        writer.add_scalars('Classification/Val', {
            'Precision': val_m['precision'], 'Recall': val_m['recall'],
            'F1': val_m['f1'],
        }, epoch)
        writer.add_scalar('System/PeakGPU_GB', peak_mem, epoch)
        writer.add_scalar('System/EpochTime_s', epoch_time, epoch)

        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s, GPU: {peak_mem:.2f}GB)")
        print(f"  Train: G={train_m['g_loss']:.4f} D={train_m['d_loss']:.4f} "
              f"Adv={train_m['g_adv']:.4f} L1={train_m['g_l1']:.4f} SSIM={train_m['g_ssim']:.4f}")
        print(f"  Val:   PSNR={val_m['psnr']:.2f} SSIM={val_m['ssim']:.4f} "
              f"MAE={val_m['mae']:.4f} RMSE={val_m['rmse']:.4f}")
        print(f"         P={val_m['precision']:.4f} R={val_m['recall']:.4f} F1={val_m['f1']:.4f}")

        # Save best
        if val_m['ssim'] > best_ssim:
            best_ssim = val_m['ssim']
            torch.save({
                'epoch': epoch, 'gen': gen.state_dict(), 'disc': disc.state_dict(),
                'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(),
                'best_ssim': best_ssim,
            }, os.path.join(run_dir, 'checkpoints', 'best_model.pth'))
            print(f"  ★ New best SSIM: {best_ssim:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'gen': gen.state_dict(), 'disc': disc.state_dict(),
            }, os.path.join(run_dir, 'checkpoints', f'epoch_{epoch+1}.pth'))

        # Save sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            gen.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                s_flair = sample_batch['image'][:4].to(device)
                s_t1 = sample_batch['label'][:4].to(device)
                with torch.amp.autocast('cuda'):
                    s_fake = gen(s_flair)
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                for i in range(min(4, s_flair.shape[0])):
                    axes[0, i].imshow(s_flair[i, 0].cpu().numpy(), cmap='gray')
                    axes[0, i].set_title('FLAIR'); axes[0, i].axis('off')
                    axes[1, i].imshow(s_t1[i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title('Real T1'); axes[1, i].axis('off')
                    axes[2, i].imshow(s_fake[i, 0].cpu().float().numpy(), cmap='gray')
                    axes[2, i].set_title('Gen T1'); axes[2, i].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, 'samples', f'epoch_{epoch+1}.png'), dpi=150)
                plt.close()

    total_time = time.time() - start_time
    peak_gpu = torch.cuda.max_memory_allocated() / 1e9
    alloc_gpu = torch.cuda.memory_allocated() / 1e9

    # ============================================================
    # FINAL REPORT
    # ============================================================
    final_val = validate(gen, val_loader, device)

    # Bootstrap CIs
    print("\n" + "="*60)
    print("  FINAL RESULTS (with 95% Bootstrap CIs)")
    print("="*60)

    ci_results = {}
    for metric_name, values in [
        ('PSNR', final_val['psnr_all']),
        ('SSIM', final_val['ssim_all']),
        ('MAE', final_val['mae_all']),
        ('RMSE', final_val['rmse_all']),
    ]:
        mean, lo, hi = bootstrap_ci(values, n_boot=1000)
        ci_results[metric_name] = {'mean': mean, 'ci_low': lo, 'ci_high': hi}
        print(f"  {metric_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\n  Precision: {final_val['precision']:.4f}")
    print(f"  Recall:    {final_val['recall']:.4f}")
    print(f"  F1 Score:  {final_val['f1']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {final_val['tp']:.0f}  FP: {final_val['fp']:.0f}")
    print(f"    FN: {final_val['fn']:.0f}  TN: {final_val['tn']:.0f}")
    print(f"\n  Training Time: {total_time/3600:.2f} hours")
    print(f"  Peak GPU Memory: {peak_gpu:.2f} GB")
    print(f"  Allocated GPU Memory: {alloc_gpu:.2f} GB")
    print(f"  Mixed Precision: FP16/FP32")
    print(f"  Epochs: {args.epochs}")
    print("="*60)

    # Save all stats as JSON
    report = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lambda_l1': args.lambda_l1,
        'lambda_ssim': args.lambda_ssim,
        'total_training_time_hours': total_time / 3600,
        'peak_gpu_memory_gb': peak_gpu,
        'allocated_gpu_memory_gb': alloc_gpu,
        'generator_params_M': count_parameters(gen) / 1e6,
        'discriminator_params_M': count_parameters(disc) / 1e6,
        'final_metrics': {
            'psnr': final_val['psnr'], 'ssim': final_val['ssim'],
            'mae': final_val['mae'], 'rmse': final_val['rmse'],
            'mse': final_val['mse'],
            'precision': final_val['precision'],
            'recall': final_val['recall'], 'f1': final_val['f1'],
        },
        'confusion_matrix': {
            'tp': final_val['tp'], 'fp': final_val['fp'],
            'fn': final_val['fn'], 'tn': final_val['tn'],
        },
        'bootstrap_ci': ci_results,
        'history': history,
    }
    with open(os.path.join(run_dir, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {run_dir}/training_report.json")

    # ============================================================
    # PLOTS
    # ============================================================
    plot_dir = os.path.join(run_dir, 'plots')

    # 1. Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_x = range(1, args.epochs + 1)
    ax1.plot(epochs_x, history['train_g_loss'], label='Generator', linewidth=2)
    ax1.plot(epochs_x, history['train_d_loss'], label='Discriminator', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs_x, history['train_g_adv'], label='Adversarial', linewidth=1.5)
    ax2.plot(epochs_x, history['train_g_l1'], label='L1', linewidth=1.5)
    ax2.plot(epochs_x, history['train_g_ssim'], label='SSIM', linewidth=1.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title('Generator Loss Components')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'), dpi=200)
    plt.close()

    # 2. Validation metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, key, title in zip(axes.flat,
        ['val_psnr', 'val_ssim', 'val_mae', 'val_rmse'],
        ['PSNR (dB)', 'SSIM', 'MAE', 'RMSE']):
        ax.plot(epochs_x, history[key], linewidth=2, color='#2196F3')
        ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'validation_metrics.png'), dpi=200)
    plt.close()

    # 3. Confusion matrix
    cm = np.array([[final_val['tn'], final_val['fp']],
                    [final_val['fn'], final_val['tp']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nHealthy', 'Predicted\nLesion'])
    ax.set_yticklabels(['Actual\nHealthy', 'Actual\nLesion'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,.0f}", ha='center', va='center',
                    fontsize=12, color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_title('Confusion Matrix (Pixel-level)')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()

    # 4. Precision/Recall/F1 over epochs
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_x, history['val_precision'], label='Precision', linewidth=2)
    ax.plot(epochs_x, history['val_recall'], label='Recall', linewidth=2)
    ax.plot(epochs_x, history['val_f1'], label='F1 Score', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Score'); ax.set_title('Classification Metrics')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'classification_metrics.png'), dpi=200)
    plt.close()

    # ============================================================
    # EXTERNAL VALIDATION: BraTS 2023 GLI Challenge
    # ============================================================
    if not args.skip_external_val and args.external_val_dir:
        print(f"\n{'='*60}")
        print(f"  EXTERNAL VALIDATION: BraTS 2023 GLI Challenge")
        print(f"  Path: {args.external_val_dir}")
        print(f"{'='*60}")
        try:
            ext_loader = create_brats2023_loader(
                args.external_val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            ext_val = validate(gen, ext_loader, device)

            # Bootstrap CIs on external set
            print("\n  External validation results (95% Bootstrap CIs):")
            ext_ci = {}
            for metric_name, values in [
                ('PSNR',  ext_val['psnr_all']),
                ('SSIM',  ext_val['ssim_all']),
                ('MAE',   ext_val['mae_all']),
                ('RMSE',  ext_val['rmse_all']),
            ]:
                mean, lo, hi = bootstrap_ci(values, n_boot=1000)
                ext_ci[metric_name] = {'mean': mean, 'ci_low': lo, 'ci_high': hi}
                print(f"  {metric_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

            print(f"\n  Precision: {ext_val['precision']:.4f}")
            print(f"  Recall:    {ext_val['recall']:.4f}")
            print(f"  F1 Score:  {ext_val['f1']:.4f}")
            print(f"  MSE:       {ext_val['mse']:.4f}")
            print(f"  Subjects:  {len(ext_loader.dataset)}")

            # Append to report
            report['external_validation_brats2023'] = {
                'n_subjects': len(ext_loader.dataset),
                'psnr': ext_val['psnr'], 'ssim': ext_val['ssim'],
                'mae': ext_val['mae'], 'rmse': ext_val['rmse'],
                'mse': ext_val['mse'],
                'precision': ext_val['precision'],
                'recall': ext_val['recall'], 'f1': ext_val['f1'],
                'confusion_matrix': {
                    'tp': ext_val['tp'], 'fp': ext_val['fp'],
                    'fn': ext_val['fn'], 'tn': ext_val['tn'],
                },
                'bootstrap_ci': ext_ci,
            }
            with open(os.path.join(run_dir, 'training_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n  Report updated with external validation results.")

            # External validation bar chart (internal vs external)
            metrics_compare = ['PSNR', 'SSIM', 'MAE', 'RMSE']
            internal_vals = [
                ci_results['PSNR']['mean'], ci_results['SSIM']['mean'],
                ci_results['MAE']['mean'],  ci_results['RMSE']['mean'],
            ]
            external_vals = [
                ext_ci['PSNR']['mean'], ext_ci['SSIM']['mean'],
                ext_ci['MAE']['mean'],  ext_ci['RMSE']['mean'],
            ]
            x = np.arange(len(metrics_compare))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - width/2, internal_vals, width, label='BraTS 2021 (Internal)', color='#2196F3')
            ax.bar(x + width/2, external_vals, width, label='BraTS 2023 (External)', color='#4CAF50')
            ax.set_xticks(x); ax.set_xticklabels(metrics_compare)
            ax.set_title('Internal vs External Validation')
            ax.legend(); ax.grid(True, alpha=0.3, axis='y')
            for i, (iv, ev) in enumerate(zip(internal_vals, external_vals)):
                ax.text(i - width/2, iv, f'{iv:.4f}', ha='center', va='bottom', fontsize=8)
                ax.text(i + width/2, ev, f'{ev:.4f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'internal_vs_external_validation.png'), dpi=200)
            plt.close()
            print(f"  Plot saved to {plot_dir}/internal_vs_external_validation.png")

        except Exception as e:
            print(f"  WARNING: External validation failed — {e}")
            print(f"  Skipping external validation. Training results are unaffected.")

    writer.close()
    print(f"\nAll plots saved to {plot_dir}/")
    print(f"TensorBoard logs: tensorboard --logdir {run_dir}/tb_logs")
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
