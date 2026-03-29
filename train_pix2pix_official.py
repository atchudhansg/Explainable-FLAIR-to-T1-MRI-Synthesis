"""
Official Pix2Pix Training Pipeline — FLAIR -> T1 MRI Synthesis
Per Isola et al. 2017 (arXiv:1611.07004)

Benchmark baseline using the ORIGINAL Pix2Pix architecture:
  - U-Net Generator (with skip connections)
  - PatchGAN Discriminator (70x70 receptive field)
  - Adversarial + L1 reconstruction loss

This is the VANILLA Pix2Pix for comparison, NOT our proposed ResNet-9 method.

Usage:
  python train_pix2pix_official.py --epochs 50 --batch_size 12
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from pytorch_msssim import ssim as compute_ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import UNetGenerator, PatchGANDiscriminator, count_parameters
from dataset import create_dataloaders, create_brats2023_loader


# ============================================================
# CONFIG
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--beta1', type=float, default=0.5)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--lambda_l1', type=float, default=100.0,
                   help='L1 reconstruction loss weight (paper default: 100)')
    p.add_argument('--data_dir', type=str, default='/home/atchu2504/training/data')
    p.add_argument('--output_dir', type=str, default='/home/atchu2504/training/outputs')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--cache_dir', type=str,
                   default='/home/atchu2504/training/cache')
    p.add_argument('--external_val_dir', type=str,
                   default='/home/atchu2504/training/validation')
    p.add_argument('--skip_external_val', action='store_true')
    return p.parse_args()


# ============================================================
# METRICS (SSIM & MAE only for benchmark)
# ============================================================

def compute_metrics_batch(generated, target):
    """Compute SSIM and MAE only for benchmark comparison."""
    gen_01 = (generated + 1.0) / 2.0
    tgt_01 = (target + 1.0) / 2.0
    ssim_val = compute_ssim(gen_01, tgt_01, data_range=1.0, size_average=True).item()
    mae = torch.mean(torch.abs(gen_01 - tgt_01)).item()
    return {'ssim': ssim_val, 'mae': mae}


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    values = np.array(values)
    boot_means = [np.mean(np.random.choice(values, size=len(values), replace=True))
                  for _ in range(n_boot)]
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(values), lower, upper


# ============================================================
# PIX2PIX TRAINING (Official: Adversarial + L1)
# ============================================================

def train_one_epoch(gen, disc, train_loader, opt_g, opt_d,
                    scaler_g, scaler_d, lambda_l1, device, epoch):
    """Official Pix2Pix training: Adversarial + L1 reconstruction."""
    gen.train()
    disc.train()
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    metrics = {'g_loss': [], 'd_loss': [], 'g_adv': [], 'g_l1': []}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch in pbar:
        real_flair = batch['image'].to(device, non_blocking=True)
        real_t1 = batch['label'].to(device, non_blocking=True)

        # ── Generator ──────────────────────────────────────────
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            fake_t1 = gen(real_flair)

            # Adversarial loss
            pred_fake = disc(real_flair, fake_t1)
            loss_g_adv = criterion_gan(pred_fake, torch.ones_like(pred_fake))

            # L1 reconstruction loss
            loss_g_l1 = criterion_l1(fake_t1, real_t1) * lambda_l1

            loss_g = loss_g_adv + loss_g_l1

        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        # ── Discriminator ──────────────────────────────────────
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            # Real pairs
            pred_real = disc(real_flair, real_t1)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            # Fake pairs (detached)
            pred_fake = disc(real_flair, fake_t1.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            loss_d = (loss_d_real + loss_d_fake) * 0.5

        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        metrics['g_loss'].append(loss_g.item())
        metrics['d_loss'].append(loss_d.item())
        metrics['g_adv'].append(loss_g_adv.item())
        metrics['g_l1'].append(loss_g_l1.item())

        pbar.set_postfix(G=f"{loss_g.item():.3f}", D=f"{loss_d.item():.3f}")

    return {k: np.mean(v) for k, v in metrics.items()}


@torch.no_grad()
def validate(gen, val_loader, device):
    """Validate generator — SSIM and MAE only for benchmark."""
    gen.eval()
    all_ssim, all_mae = [], []

    for batch in tqdm(val_loader, desc="  [Val]", leave=False):
        flair = batch['image'].to(device, non_blocking=True)
        t1_real = batch['label'].to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            fake_t1 = gen(flair)
        m = compute_metrics_batch(fake_t1.float(), t1_real.float())
        all_ssim.append(m['ssim'])
        all_mae.append(m['mae'])

    return {
        'ssim': np.mean(all_ssim), 'mae': np.mean(all_mae),
        'ssim_all': all_ssim, 'mae_all': all_mae,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'='*60}")
    print(f"  Official Pix2Pix: FLAIR -> T1 Synthesis")
    print(f"  Architecture: U-Net Generator + PatchGAN Discriminator")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Lambda L1: {args.lambda_l1}")
    print(f"{'='*60}\n")

    # Output dirs
    run_dir = os.path.join(args.output_dir, 'pix2pix_official')
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'samples'), exist_ok=True)

    writer = SummaryWriter(os.path.join(run_dir, 'tb_logs'))

    # Data
    cache_dir = args.cache_dir if args.cache_dir else None
    train_loader, val_loader, train_idx, val_idx = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed,
        num_workers=args.num_workers, cache_dir=cache_dir
    )

    # Models (Official Pix2Pix architecture)
    gen = UNetGenerator(in_channels=3, out_channels=3).to(device)
    disc = PatchGANDiscriminator(in_channels=6).to(device)  # 6 = concat(FLAIR, T1)

    print(f"Generator (U-Net) params: {count_parameters(gen)/1e6:.2f}M")
    print(f"Discriminator (PatchGAN) params: {count_parameters(disc)/1e6:.2f}M")
    print(f"Total: {(count_parameters(gen)+count_parameters(disc))/1e6:.2f}M\n")

    if args.compile:
        print("  Compiling models with torch.compile...")
        gen = torch.compile(gen)
        disc = torch.compile(disc)
        print("  Compilation ready.\n")

    # Optimizers (separate, per Pix2Pix paper)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    scaler_g = GradScaler('cuda')
    scaler_d = GradScaler('cuda')

    # History
    history = {
        'train_g_loss': [], 'train_d_loss': [],
        'train_g_adv': [], 'train_g_l1': [],
        'val_ssim': [], 'val_mae': [],
    }

    best_ssim = 0
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_m = train_one_epoch(
            gen, disc, train_loader, opt_g, opt_d,
            scaler_g, scaler_d, args.lambda_l1, device, epoch
        )

        # Validate
        val_m = validate(gen, val_loader, device)

        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        # Log history
        for k in ['g_loss', 'd_loss', 'g_adv', 'g_l1']:
            history[f'train_{k}'].append(train_m[k])
        for k in ['ssim', 'mae']:
            history[f'val_{k}'].append(val_m[k])

        # TensorBoard
        writer.add_scalars('Loss/Train', {
            'G_total': train_m['g_loss'], 'D': train_m['d_loss'],
            'G_adv': train_m['g_adv'], 'G_L1': train_m['g_l1'],
        }, epoch)
        writer.add_scalars('Metrics/Val', {
            'SSIM': val_m['ssim'], 'MAE': val_m['mae'],
        }, epoch)
        writer.add_scalar('System/PeakGPU_GB', peak_mem, epoch)
        writer.add_scalar('System/EpochTime_s', epoch_time, epoch)

        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s, GPU: {peak_mem:.2f}GB)")
        print(f"  Train: G={train_m['g_loss']:.4f} D={train_m['d_loss']:.4f} "
              f"Adv={train_m['g_adv']:.4f} L1={train_m['g_l1']:.4f}")
        print(f"  Val:   ★ SSIM={val_m['ssim']:.4f} MAE={val_m['mae']:.4f} ★")

        # Save best (based on SSIM)
        if val_m['ssim'] > best_ssim:
            best_ssim = val_m['ssim']
            torch.save({
                'epoch': epoch, 'gen': gen.state_dict(), 'disc': disc.state_dict(),
                'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(),
                'best_ssim': best_ssim,
            }, os.path.join(run_dir, 'checkpoints', 'best_model.pth'))
            print(f"  * New best SSIM: {best_ssim:.4f}")

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

    # ============================================================
    # FINAL REPORT
    # ============================================================
    final_val = validate(gen, val_loader, device)

    print(f"\n{'='*80}")
    print(f"  ★★★ OFFICIAL PIX2PIX BENCHMARK RESULTS (for paper comparison) ★★★")
    print(f"{'='*80}")

    ci_results = {}
    for metric_name, values in [
        ('SSIM', final_val['ssim_all']), ('MAE', final_val['mae_all']),
    ]:
        mean, lo, hi = bootstrap_ci(values, n_boot=1000)
        ci_results[metric_name] = {'mean': mean, 'ci_low': lo, 'ci_high': hi}
        print(f"  ★★★ {metric_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\n  Training: {total_time/3600:.2f}h | Peak GPU: {peak_gpu:.2f}GB | Best SSIM: {best_ssim:.4f}")
    print(f"{'='*80}")

    # Save report JSON
    report = {
        'model': 'pix2pix_official',
        'architecture': 'U-Net Generator + PatchGAN Discriminator',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lambda_l1': args.lambda_l1,
        'total_training_time_hours': total_time / 3600,
        'peak_gpu_memory_gb': peak_gpu,
        'gen_params_M': count_parameters(gen) / 1e6,
        'disc_params_M': count_parameters(disc) / 1e6,
        'best_ssim': best_ssim,
        'final_metrics': {
            'ssim': final_val['ssim'],
            'mae': final_val['mae'],
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
    epochs_x = range(1, args.epochs + 1)

    # 1. Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_x, history['train_g_loss'], label='Generator', linewidth=2)
    ax1.plot(epochs_x, history['train_d_loss'], label='Discriminator', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Pix2Pix Training Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs_x, history['train_g_adv'], label='Adversarial', linewidth=1.5)
    ax2.plot(epochs_x, history['train_g_l1'], label='L1 Recon', linewidth=1.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title('Generator Loss Components')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'), dpi=200)
    plt.close()

    # 2. Validation metrics (SSIM & MAE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs_x, history['val_ssim'], linewidth=2, color='#FF5722')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('SSIM'); ax1.set_title('SSIM')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs_x, history['val_mae'], linewidth=2, color='#2196F3')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE'); ax2.set_title('MAE')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'validation_metrics.png'), dpi=200)
    plt.close()

    # ============================================================
    # EXTERNAL VALIDATION
    # ============================================================
    if not args.skip_external_val and args.external_val_dir:
        print(f"\n{'='*60}")
        print(f"  EXTERNAL VALIDATION: BraTS 2023 GLI Challenge")
        print(f"{'='*60}")
        try:
            ext_loader = create_brats2023_loader(
                args.external_val_dir, batch_size=args.batch_size,
                num_workers=args.num_workers)
            ext_val = validate(gen, ext_loader, device)

            ext_ci = {}
            for metric_name, values in [
                ('SSIM', ext_val['ssim_all']), ('MAE', ext_val['mae_all']),
            ]:
                mean, lo, hi = bootstrap_ci(values, n_boot=1000)
                ext_ci[metric_name] = {'mean': mean, 'ci_low': lo, 'ci_high': hi}
                print(f"  ★ {metric_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

            print(f"  Subjects:  {len(ext_loader.dataset)}")

            report['external_validation_brats2023'] = {
                'n_subjects': len(ext_loader.dataset),
                'ssim': ext_val['ssim'], 'mae': ext_val['mae'],
                'bootstrap_ci': ext_ci,
            }
            with open(os.path.join(run_dir, 'training_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
            print(f"  Report updated.")
        except Exception as e:
            print(f"  WARNING: External validation failed — {e}")

    # Export clean generator weights
    best_ckpt = torch.load(os.path.join(run_dir, 'checkpoints', 'best_model.pth'),
                           map_location='cpu', weights_only=False)
    gen_sd = {k.replace('_orig_mod.', ''): v for k, v in best_ckpt['gen'].items()}
    torch.save({'gen': gen_sd, 'epoch': best_ckpt['epoch'], 'best_ssim': best_ckpt['best_ssim']},
               os.path.join(run_dir, 'checkpoints', 'best_gen_weights.pth'))
    print(f"\nClean generator weights exported to {run_dir}/checkpoints/best_gen_weights.pth")

    writer.close()
    print(f"\nAll plots saved to {plot_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
