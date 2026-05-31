"""Compute q01/q99 feature statistics for SMP NPZ motion windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/npz")
    parser.add_argument("--output", default="datasets/norm_stats.npz")
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    args = parser.parse_args()

    npz_files = sorted(Path(args.input_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files in {args.input_dir}")

    chunks = []
    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=False) as data:
            windows = data["windows"]
        chunks.append(windows.reshape(-1, windows.shape[-1]))
        print(f"{npz_file.name}: windows={windows.shape[0]}, feature_dim={windows.shape[-1]}")

    all_frames = np.concatenate(chunks, axis=0).astype(np.float64)
    q_low = np.percentile(all_frames, args.q_low * 100, axis=0).astype(np.float32)
    q_high = np.percentile(all_frames, args.q_high * 100, axis=0).astype(np.float32)
    tiny = (q_high - q_low) < 1e-6
    if tiny.any():
        q_high[tiny] = q_low[tiny] + 1.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, q_low=q_low, q_high=q_high)
    print(f"Saved {output}: feature_dim={q_low.shape[0]}")


if __name__ == "__main__":
    main()
