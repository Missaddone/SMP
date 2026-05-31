"""Configuration for SMP diffusion pretraining."""

from __future__ import annotations

from dataclasses import dataclass

from SMP_catchball.smp.utils import detect_device


@dataclass
class PretrainCfg:
    data_dir: str = "datasets/npz"
    norm_stats_file: str = "datasets/norm_stats.npz"
    train_split: float = 0.9

    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.0

    num_timesteps: int = 50
    num_noise_samples: int = 10

    use_ema: bool = False
    ema_decay: float = 0.9999

    batch_size: int = 1024
    num_epochs: int = 2000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    name: str = "pretrain"
    log_interval: int = 10
    save_interval: int = 100
    log_dir: str = "logs/pretrain"
    use_wandb: bool = False
    wandb_project: str = "smp"

    device: str = ""
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.device:
            self.device = detect_device()
