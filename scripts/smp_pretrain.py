"""Train an SMP diffusion prior from pre-windowed NPZ motion data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "SMP"))

from SMP.smp.pretrain import pretrain
from SMP.smp.pretrain_cfg import PretrainCfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=PretrainCfg.data_dir)
    parser.add_argument("--norm-stats-file", default=PretrainCfg.norm_stats_file)
    parser.add_argument("--log-dir", default=PretrainCfg.log_dir)
    parser.add_argument("--name", default=PretrainCfg.name)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=PretrainCfg.batch_size)
    parser.add_argument("--num-epochs", type=int, default=PretrainCfg.num_epochs)
    parser.add_argument("--num-timesteps", type=int, default=PretrainCfg.num_timesteps)
    parser.add_argument("--num-noise-samples", type=int, default=PretrainCfg.num_noise_samples)
    parser.add_argument("--d-model", type=int, default=PretrainCfg.d_model)
    parser.add_argument("--nhead", type=int, default=PretrainCfg.nhead)
    parser.add_argument("--num-layers", type=int, default=PretrainCfg.num_layers)
    parser.add_argument("--dropout", type=float, default=PretrainCfg.dropout)
    parser.add_argument("--lr", type=float, default=PretrainCfg.lr)
    parser.add_argument("--weight-decay", type=float, default=PretrainCfg.weight_decay)
    parser.add_argument("--train-split", type=float, default=PretrainCfg.train_split)
    parser.add_argument("--save-interval", type=int, default=PretrainCfg.save_interval)
    parser.add_argument("--log-interval", type=int, default=PretrainCfg.log_interval)
    parser.add_argument("--seed", type=int, default=PretrainCfg.seed)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=PretrainCfg.ema_decay)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    cfg = PretrainCfg(**vars(args))
    pretrain(cfg)


if __name__ == "__main__":
    main()
