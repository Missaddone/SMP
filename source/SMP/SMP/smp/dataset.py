"""Motion-window dataset for SMP diffusion pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionWindowDataset(Dataset[torch.Tensor]):
    """Load pre-windowed NPZ files with a ``windows`` array of shape ``(N, W, F)``."""

    def __init__(self, data_dir: str | Path, norm_stats_file: str | Path | None = None) -> None:
        npz_files = sorted(Path(data_dir).glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files found in {data_dir}")

        chunks: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None
        for npz_file in npz_files:
            with np.load(npz_file, allow_pickle=False) as npz:
                windows = npz["windows"].astype(np.float32, copy=False)
            if windows.ndim != 3:
                raise ValueError(f"{npz_file.name}: expected windows shape (N, W, F), got {windows.shape}")
            shape = (int(windows.shape[1]), int(windows.shape[2]))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(f"{npz_file.name}: shape {shape} mismatches first file shape {expected_shape}")
            chunks.append(windows)

        assert expected_shape is not None
        self.window_size, self.feature_dim = expected_shape
        data = np.concatenate(chunks, axis=0)

        if norm_stats_file is not None and Path(norm_stats_file).is_file():
            stats = np.load(norm_stats_file, allow_pickle=False)
            self.q_low = stats["q_low"].astype(np.float32)
            self.q_high = stats["q_high"].astype(np.float32)
        else:
            flat = data.reshape(-1, self.feature_dim)
            self.q_low = np.percentile(flat, 1, axis=0).astype(np.float32)
            self.q_high = np.percentile(flat, 99, axis=0).astype(np.float32)

        span = self.q_high - self.q_low
        tiny = span < 1e-6
        if tiny.any():
            self.q_high[tiny] = self.q_low[tiny] + 1.0

        data = 2.0 * (data - self.q_low) / (self.q_high - self.q_low) - 1.0
        self.windows = torch.from_numpy(data.astype(np.float32, copy=False))

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        q_low = torch.from_numpy(self.q_low).to(x.device, x.dtype)
        q_high = torch.from_numpy(self.q_high).to(x.device, x.dtype)
        return (x + 1.0) / 2.0 * (q_high - q_low) + q_low

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]
