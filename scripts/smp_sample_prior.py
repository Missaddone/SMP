"""Sample denormalized motion windows from a trained SMP prior checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "SMP"))

from SMP.smp.utils import load_denoiser


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output", default="outputs/smp_prior_samples.npz")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(args.ckpt_path, device)

    x_t = torch.randn(args.num_samples, window_size, feature_dim, device=device)
    for timestep in reversed(range(scheduler.num_timesteps)):
        t = torch.full((args.num_samples,), timestep, dtype=torch.long, device=device)
        eps = model(x_t, t)
        x_t = scheduler.step(eps, x_t, timestep)

    windows = (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        windows=windows.cpu().numpy().astype(np.float32),
        feature_dim=np.array([feature_dim], dtype=np.int32),
        window_size=np.array([window_size], dtype=np.int32),
    )
    print(f"Saved {args.num_samples} samples to {output}: windows={tuple(windows.shape)}")


if __name__ == "__main__":
    main()
