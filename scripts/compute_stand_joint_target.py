"""Compute a 29-D G1 standing joint target from stand motion data.

Inputs can be either:
  - CSV files with columns root_pos(3), root_quat(4), joint_pos(29)
  - SMP NPZ files with a ``windows`` array whose feature layout contains
    joint_pos at columns [9:38]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_joint_pos(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            windows = data["windows"].astype(np.float64, copy=False)
        if windows.ndim != 3 or windows.shape[-1] < 38:
            raise ValueError(f"{path}: expected windows shape (N, W, F>=38), got {windows.shape}")
        return windows[:, :, 9:38].reshape(-1, 29)

    if path.suffix.lower() == ".csv":
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
        except ValueError:
            named = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
            if named.dtype.names is None:
                raise
            arr = np.column_stack([named[name] for name in named.dtype.names])
        if arr.ndim == 1:
            arr = arr[None]
        if arr.shape[1] < 36:
            raise ValueError(f"{path}: expected at least 36 columns, got {arr.shape[1]}")
        return arr[:, 7:36]

    raise ValueError(f"{path}: unsupported file type")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Stand CSV/NPZ files or directories containing them.")
    parser.add_argument("--stat", choices=("median", "mean"), default="median")
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.npz")))
            files.extend(sorted(path.glob("*.csv")))
        else:
            files.append(path)
    if not files:
        raise FileNotFoundError("No CSV/NPZ files found.")

    joint_chunks = [_load_joint_pos(path) for path in files]
    joint_pos = np.concatenate(joint_chunks, axis=0)
    target = np.median(joint_pos, axis=0) if args.stat == "median" else np.mean(joint_pos, axis=0)

    print("STANDING_JOINT_TARGET: tuple[float, ...] = (")
    for value in target:
        print(f"    {value:.6f},")
    print(")")


if __name__ == "__main__":
    main()
