"""
AttentionGAN Training Pipeline — FLAIR <-> T1 MRI Synthesis
Per Tang et al. 2019 (arXiv:1903.12296)
Official implementation: https://github.com/Ha0Tang/AttentionGAN

Benchmark baseline for paper comparison.
Key features:
  - Attention-guided generators (content + attention masks)
  - Cycle consistency loss (like CycleGAN)
  - Identity loss
  - Attention mechanism: out = content * attention + input * (1 - attention)

Usage:
  python train_attentiongan.py --epochs 50 --batch_size 12
"""
import os, sys, json, time, argparse
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
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

from models import get_attentiongan_models, ImageBuffer, count_parameters
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
    p.add_argument('--lambda_cycle', type=float, default=10.0,
                   help='Cycle consistency loss weight')
    p.add_argument('--lambda_identity', type=float, default=5.0,
                   help='Identity loss weight')
    p.add_argument('--lr_decay_start', type=int, default=25,
                   help='Epoch to start linear LR decay')
    p.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'data'))
    p.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR, 'outputs'))
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--cache_dir', type=str,
                   default=os.path.join(ROOT_DIR, 'cache'))
    p.add_argument('--external_val_dir', type=str,
                   default=os.path.join(ROOT_DIR, 'validation'))
    p.add_argument('--skip_external_val', action='store_true')
    p.add_argument('--simple', action='store_true', default=True,
                   help='Use simplified single-attention generator (default: True)')
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
# ATTENTIONGAN TRAINING
# ============================================================

def train_one_epoch(g_ab, g_ba, d_a, d_b, train_loader, opt_g, opt_d,
                    scaler_g, scaler_d, lambda_cycle, lambda_identity,
                    device, epoch, buf_a, buf_b):
    """AttentionGAN training: cycle consistency + identity + adversarial."""
    g_ab.train(); g_ba.train(); d_a.train(); d_b.train()
    criterion = nn.MSELoss()  # LSGAN
    l1 = nn.L1Loss()

    metrics = {'g_loss': [], 'd_loss': [], 'cycle': [], 'identity': [],
               'adv_ab': [], 'adv_ba': []}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch in pbar:
        real_a = batch['image'].to(device, non_blocking=True)  # FLAIR
        real_b = batch['label'].to(device, non_blocking=True)  # T1

        # ── Generators ──────────────────────────────────────
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            # Identity loss
            loss_idt = torch.tensor(0.0, device=device)
            if lambda_identity > 0:
                idt_b = g_ab(real_b)
                idt_a = g_ba(real_a)
                loss_idt = (l1(idt_b, real_b) + l1(idt_a, real_a)) * lambda_identity

            # GAN loss
            fake_b = g_ab(real_a)  # FLAIR -> fake T1
            pred_fake_b = d_b(fake_b)
            loss_gan_ab = criterion(pred_fake_b, torch.ones_like(pred_fake_b))

            fake_a = g_ba(real_b)  # T1 -> fake FLAIR
            pred_fake_a = d_a(fake_a)
            loss_gan_ba = criterion(pred_fake_a, torch.ones_like(pred_fake_a))

            # Cycle consistency
            recon_a = g_ba(fake_b)
            recon_b = g_ab(fake_a)
            loss_cycle = (l1(recon_a, real_a) + l1(recon_b, real_b)) * lambda_cycle

            loss_g = loss_gan_ab + loss_gan_ba + loss_cycle + loss_idt

        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        # ── Discriminators ───────────────────────────────────
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            fake_a_buf = buf_a.push_and_pop(fake_a.detach())
            loss_d_a = (criterion(d_a(real_a), torch.ones_like(d_a(real_a))) +
                        criterion(d_a(fake_a_buf), torch.zeros_like(d_a(fake_a_buf)))) * 0.5

            fake_b_buf = buf_b.push_and_pop(fake_b.detach())
            loss_d_b = (criterion(d_b(real_b), torch.ones_like(d_b(real_b))) +
                        criterion(d_b(fake_b_buf), torch.zeros_like(d_b(fake_b_buf)))) * 0.5

            loss_d = loss_d_a + loss_d_b

        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        metrics['g_loss'].append(loss_g.item())
        metrics['d_loss'].append(loss_d.item())
        metrics['cycle'].append(loss_cycle.item())
        metrics['identity'].append(loss_idt.item())
        metrics['adv_ab'].append(loss_gan_ab.item())
        metrics['adv_ba'].append(loss_gan_ba.item())

        pbar.set_postfix(G=f"{loss_g.item():.3f}", D=f"{loss_d.item():.3f}")

    return {k: np.mean(v) for k, v in metrics.items()}


@torch.no_grad()
def validate(gen, val_loader, device):
    """Validate G_AB (FLAIR->T1) — SSIM and MAE only for benchmark."""
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
    print(f"  AttentionGAN: FLAIR <-> T1 Synthesis")
    print(f"  Per Tang et al. 2019 (arXiv:1903.12296)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Lambda Cycle: {args.lambda_cycle}, Lambda Identity: {args.lambda_identity}")
    print(f"  Generator: {'Simple (single attention)' if args.simple else 'Full (multi-attention)'}")
    print(f"{'='*60}\n")

    # Output dirs
    run_dir = os.path.join(args.output_dir, 'attentiongan')
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

    # Models
    g_ab, g_ba, d_a, d_b = get_attentiongan_models(simple=args.simple)
    g_ab, g_ba = g_ab.to(device), g_ba.to(device)
    d_a, d_b = d_a.to(device), d_b.to(device)

    print(f"G_AB (Attention) params: {count_parameters(g_ab)/1e6:.2f}M")
    print(f"G_BA (Attention) params: {count_parameters(g_ba)/1e6:.2f}M")
    print(f"D_A params: {count_parameters(d_a)/1e6:.2f}M")
    print(f"D_B params: {count_parameters(d_b)/1e6:.2f}M")
    print(f"Total: {(count_parameters(g_ab)+count_parameters(g_ba)+count_parameters(d_a)+count_parameters(d_b))/1e6:.2f}M\n")

    if args.compile:
        print("  Compiling models with torch.compile...")
        g_ab = torch.compile(g_ab)
        g_ba = torch.compile(g_ba)
        d_a = torch.compile(d_a)
        d_b = torch.compile(d_b)
        print("  Compilation ready.\n")

    # Optimizers
    import itertools
    opt_g = torch.optim.Adam(
        itertools.chain(g_ab.parameters(), g_ba.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(
        itertools.chain(d_a.parameters(), d_b.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2))

    scaler_g = GradScaler('cuda')
    scaler_d = GradScaler('cuda')

    # Image replay buffers
    buf_a = ImageBuffer(max_size=50)
    buf_b = ImageBuffer(max_size=50)

    # LR scheduler
    def lr_lambda(epoch):
        if epoch < args.lr_decay_start:
            return 1.0
        decay_epochs = args.epochs - args.lr_decay_start
        if decay_epochs <= 0:
            return 1.0
        return max(0.0, 1.0 - (epoch - args.lr_decay_start) / decay_epochs)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)

    # History
    history = {
        'train_g_loss': [], 'train_d_loss': [],
        'train_cycle': [], 'train_identity': [],
        'train_adv_ab': [], 'train_adv_ba': [],
        'val_ssim': [], 'val_mae': [],
    }

    best_ssim = 0
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_m = train_one_epoch(
            g_ab, g_ba, d_a, d_b, train_loader, opt_g, opt_d,
            scaler_g, scaler_d, args.lambda_cycle, args.lambda_identity,
            device, epoch, buf_a, buf_b
        )

        sched_g.step()
        sched_d.step()

        val_m = validate(g_ab, val_loader, device)

        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        # Log history
        for k in ['g_loss', 'd_loss', 'cycle', 'identity', 'adv_ab', 'adv_ba']:
            history[f'train_{k}'].append(train_m[k])
        for k in ['ssim', 'mae']:
            history[f'val_{k}'].append(val_m[k])

        # TensorBoard
        writer.add_scalars('Loss/Train', {
            'G_total': train_m['g_loss'], 'D': train_m['d_loss'],
            'Cycle': train_m['cycle'], 'Identity': train_m['identity'],
        }, epoch)
        writer.add_scalars('Metrics/Val', {
            'SSIM': val_m['ssim'], 'MAE': val_m['mae'],
        }, epoch)
        writer.add_scalar('System/PeakGPU_GB', peak_mem, epoch)

        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s, GPU: {peak_mem:.2f}GB)")
        print(f"  Train: G={train_m['g_loss']:.4f} D={train_m['d_loss']:.4f} "
              f"Cycle={train_m['cycle']:.4f} Idt={train_m['identity']:.4f}")
        print(f"  Val:   ★ SSIM={val_m['ssim']:.4f} MAE={val_m['mae']:.4f} ★")

        # Save best
        if val_m['ssim'] > best_ssim:
            best_ssim = val_m['ssim']
            torch.save({
                'epoch': epoch, 'g_ab': g_ab.state_dict(), 'g_ba': g_ba.state_dict(),
                'd_a': d_a.state_dict(), 'd_b': d_b.state_dict(),
                'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(),
                'best_ssim': best_ssim,
            }, os.path.join(run_dir, 'checkpoints', 'best_model.pth'))
            print(f"  * New best SSIM: {best_ssim:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'g_ab': g_ab.state_dict(), 'g_ba': g_ba.state_dict(),
                'd_a': d_a.state_dict(), 'd_b': d_b.state_dict(),
            }, os.path.join(run_dir, 'checkpoints', f'epoch_{epoch+1}.pth'))

        # Save samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            g_ab.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                s_flair = sample_batch['image'][:4].to(device)
                s_t1 = sample_batch['label'][:4].to(device)
                with torch.amp.autocast('cuda'):
                    s_fake = g_ab(s_flair)
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
    final_val = validate(g_ab, val_loader, device)

    print(f"\n{'='*80}")
    print(f"  ★★★ ATTENTIONGAN BENCHMARK RESULTS (for paper comparison) ★★★")
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
        'model': 'attentiongan',
        'architecture': 'Attention-guided Generator + PatchGAN Discriminator',
        'simple_generator': args.simple,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lambda_cycle': args.lambda_cycle,
        'lambda_identity': args.lambda_identity,
        'total_training_time_hours': total_time / 3600,
        'peak_gpu_memory_gb': peak_gpu,
        'g_ab_params_M': count_parameters(g_ab) / 1e6,
        'g_ba_params_M': count_parameters(g_ba) / 1e6,
        'd_a_params_M': count_parameters(d_a) / 1e6,
        'd_b_params_M': count_parameters(d_b) / 1e6,
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

    # Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_x, history['train_g_loss'], label='Generator', linewidth=2)
    ax1.plot(epochs_x, history['train_d_loss'], label='Discriminator', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('AttentionGAN Training Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs_x, history['train_cycle'], label='Cycle', linewidth=1.5)
    ax2.plot(epochs_x, history['train_identity'], label='Identity', linewidth=1.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title('Generator Loss Components')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'), dpi=200)
    plt.close()

    # Validation metrics
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
            ext_val = validate(g_ab, ext_loader, device)

            ext_ci = {}
            for metric_name, values in [
                ('SSIM', ext_val['ssim_all']), ('MAE', ext_val['mae_all']),
            ]:
                mean, lo, hi = bootstrap_ci(values, n_boot=1000)
                ext_ci[metric_name] = {'mean': mean, 'ci_low': lo, 'ci_high': hi}
                print(f"  ★ {metric_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

            print(f"  Subjects: {len(ext_loader.dataset)}")

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

    # Export clean G_AB weights
    best_ckpt = torch.load(os.path.join(run_dir, 'checkpoints', 'best_model.pth'),
                           map_location='cpu', weights_only=False)
    gen_sd = {k.replace('_orig_mod.', ''): v for k, v in best_ckpt['g_ab'].items()}
    torch.save({'gen': gen_sd, 'epoch': best_ckpt['epoch'], 'best_ssim': best_ckpt['best_ssim']},
               os.path.join(run_dir, 'checkpoints', 'best_gen_weights.pth'))
    print(f"\nClean G_AB weights exported to {run_dir}/checkpoints/best_gen_weights.pth")

    writer.close()
    print(f"\nAll plots saved to {plot_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
