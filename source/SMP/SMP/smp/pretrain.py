"""Diffusion prior pretraining for SMP motion windows."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from SMP.smp.dataset import MotionWindowDataset
from SMP.smp.model import DiffusionDenoiser
from SMP.smp.pretrain_cfg import PretrainCfg
from SMP.smp.scheduler import DDPMScheduler
from SMP.smp.utils import count_parameters, seed_everything


class Ema:
    """Exponential moving average shadow model."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        src = model.state_dict()
        dst = self.shadow.state_dict()
        for key, src_value in src.items():
            dst_value = dst[key]
            if dst_value.is_floating_point():
                dst_value.mul_(self.decay).add_(src_value.detach(), alpha=1.0 - self.decay)
            else:
                dst_value.copy_(src_value)


def diffusion_loss(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    num_noise_samples: int,
) -> torch.Tensor:
    """DDPM epsilon-prediction L1 loss."""
    batch_size = x_0.shape[0]
    num_samples = num_noise_samples
    x_0_exp = x_0[:, None].expand(batch_size, num_samples, *x_0.shape[1:])
    x_0_exp = x_0_exp.reshape(batch_size * num_samples, *x_0.shape[1:])
    timesteps = scheduler.sample_timesteps(batch_size * num_samples, x_0.device)
    noise = torch.randn_like(x_0_exp)
    x_t = scheduler.add_noise(x_0_exp, noise, timesteps)
    return F.l1_loss(model(x_t, timesteps), noise)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: DiffusionDenoiser,
    dataset: MotionWindowDataset,
    cfg: PretrainCfg,
    optimizer: torch.optim.Optimizer | None = None,
    ema: Ema | None = None,
) -> None:
    data: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "q_low": dataset.q_low,
        "q_high": dataset.q_high,
        "cfg": {
            **vars(cfg),
            "feature_dim": dataset.feature_dim,
            "window_size": dataset.window_size,
        },
    }
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    if ema is not None:
        data["model_ema"] = ema.shadow.state_dict()
    torch.save(data, path)


def pretrain(cfg: PretrainCfg) -> Path:
    """Run SMP diffusion pretraining and return the final checkpoint path."""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    dataset = MotionWindowDataset(cfg.data_dir, norm_stats_file=cfg.norm_stats_file)

    n_train = int(len(dataset) * cfg.train_split)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory)

    model = DiffusionDenoiser(
        feature_dim=dataset.feature_dim,
        window_size=dataset.window_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    scheduler = DDPMScheduler(num_timesteps=cfg.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = Ema(model, decay=cfg.ema_decay) if cfg.use_ema else None

    print(
        f"Dataset: {len(dataset)} windows, train={n_train}, val={n_val}, "
        f"window_size={dataset.window_size}, feature_dim={dataset.feature_dim}"
    )
    print(f"Denoiser: {count_parameters(model):,} params on {device}")

    save_dir = Path(cfg.log_dir) / cfg.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = None
    if cfg.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
        tb_writer.add_text(
            "config",
            "\n".join(f"{key}: {value}" for key, value in sorted(vars(cfg).items())),
            0,
        )
        tb_writer.add_scalar("dataset/num_windows", len(dataset), 0)
        tb_writer.add_scalar("dataset/train_windows", n_train, 0)
        tb_writer.add_scalar("dataset/val_windows", n_val, 0)
        tb_writer.add_scalar("dataset/window_size", dataset.window_size, 0)
        tb_writer.add_scalar("dataset/feature_dim", dataset.feature_dim, 0)
        tb_writer.add_scalar("model/num_parameters", count_parameters(model), 0)

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.name, config=vars(cfg))

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = torch.zeros((), device=device)
        n_batches = 0

        for batch in train_loader:
            x_0 = batch.to(device, non_blocking=pin_memory)
            loss = diffusion_loss(model, scheduler, x_0, cfg.num_noise_samples)
            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += loss.detach()
            n_batches += 1

        train_loss = (total_loss / max(n_batches, 1)).item()
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", train_loss, epoch)
            tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % cfg.log_interval == 0:
            eval_model = ema.shadow if ema is not None else model
            val_loss = validate(eval_model, scheduler, val_loader, device, pin_memory, cfg.num_noise_samples)
            print(f"Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f}")
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss, epoch)
                tb_writer.flush()
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})

        if epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs - 1:
            save_checkpoint(save_dir / f"checkpoint_{epoch:05d}.pt", epoch, model, dataset, cfg, optimizer, ema)

    final_path = save_dir / "pretrained.pt"
    save_checkpoint(final_path, cfg.num_epochs, model, dataset, cfg, ema=ema)
    print(f"Saved final checkpoint to {final_path}")
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    return final_path


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    val_loader: DataLoader,
    device: torch.device,
    pin_memory: bool,
    num_noise_samples: int,
) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    n_batches = 0
    for batch in val_loader:
        x_0 = batch.to(device, non_blocking=pin_memory)
        total += diffusion_loss(model, scheduler, x_0, num_noise_samples)
        n_batches += 1
    return (total / max(n_batches, 1)).item()
