"""Utilities for SMP RL: denoiser loader, diff-normalizer, and feature buffer."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from SMP.smp.model import DiffusionDenoiser
from SMP.smp.scheduler import DDPMScheduler


def detect_device() -> str:
  """Auto-detect the best available torch device."""
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


def count_parameters(module: nn.Module) -> int:
  """Return the number of parameters in a module."""
  return sum(p.numel() for p in module.parameters())


def seed_everything(seed: int, deterministic: bool = False) -> None:
  """Seed Python, NumPy, and PyTorch RNGs."""
  import os
  import random

  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _install_numpy_pickle_compat() -> None:
  """Allow checkpoints pickled with NumPy 2.x to load under NumPy 1.x.

  EN:
    Some pretrained checkpoints were pickled with NumPy 2.x module names such
    as ``numpy._core``. Isaac Sim may load a NumPy build where only
    ``numpy.core`` exists, especially when GUI/headless modes alter sys.path.
    Registering these aliases keeps torch.load independent of that runtime
    detail.

  中文：
    有些预训练 checkpoint 是用 NumPy 2.x 保存的，pickle 里会记录
    ``numpy._core`` 这类模块名。Isaac Sim 在 GUI/headless 不同模式下可能
    加载到只提供 ``numpy.core`` 的 NumPy。这里补充模块别名，让 torch.load
    不受运行时 NumPy 路径差异影响。
  """
  if importlib.util.find_spec("numpy._core") is not None:
    return

  numpy_core = importlib.import_module("numpy.core")
  sys.modules.setdefault("numpy._core", numpy_core)
  for name in ("multiarray", "numeric", "fromnumeric", "shape_base", "umath"):
    try:
      sys.modules.setdefault(f"numpy._core.{name}", importlib.import_module(f"numpy.core.{name}"))
    except ModuleNotFoundError:
      pass


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  out = q.clone()
  out[..., 1:] = -out[..., 1:]
  return out


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
  w1, x1, y1, z1 = q.unbind(dim=-1)
  w2, x2, y2, z2 = r.unbind(dim=-1)
  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  q_xyz = q[..., 1:]
  q_w = q[..., :1]
  t = 2.0 * torch.cross(q_xyz, v, dim=-1)
  return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_apply(quat_conjugate(q), v)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(dim=-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  half_yaw = 0.5 * yaw
  out = torch.zeros_like(q)
  out[..., 0] = torch.cos(half_yaw)
  out[..., 3] = torch.sin(half_yaw)
  return out


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(dim=-1)
  ww = w * w
  xx = x * x
  yy = y * y
  zz = z * z
  wx = w * x
  wy = w * y
  wz = w * z
  xy = x * y
  xz = x * z
  yz = y * z
  return torch.stack(
    [
      ww + xx - yy - zz,
      2.0 * (xy - wz),
      2.0 * (xz + wy),
      2.0 * (xy + wz),
      ww - xx + yy - zz,
      2.0 * (yz - wx),
      2.0 * (xz - wy),
      2.0 * (yz + wx),
      ww - xx - yy + zz,
    ],
    dim=-1,
  ).reshape(q.shape[:-1] + (3, 3))


def load_denoiser(
  ckpt_path: str,
  device: torch.device | str,
) -> tuple[DiffusionDenoiser, DDPMScheduler, torch.Tensor, torch.Tensor, int, int]:
  """Load a frozen pretrained denoiser checkpoint → ``(model, scheduler, q_low,
  q_high, feature_dim, window_size)``."""
  device = torch.device(device)

  _install_numpy_pickle_compat()
  ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
  cfg = ckpt["cfg"]
  feature_dim = int(cfg["feature_dim"])
  window_size = int(cfg["window_size"])

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=int(cfg.get("d_model", 256)),
    nhead=int(cfg.get("nhead", 8)),
    num_layers=int(cfg.get("num_layers", 2)),
    dropout=float(cfg.get("dropout", 0.0)),
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  model.requires_grad_(False)

  scheduler = DDPMScheduler(
    num_timesteps=int(cfg.get("num_timesteps", 50)),
  ).to(device)

  q_low = torch.from_numpy(np.asarray(ckpt["q_low"], dtype=np.float32)).to(device)
  q_high = torch.from_numpy(np.asarray(ckpt["q_high"], dtype=np.float32)).to(device)

  return model, scheduler, q_low, q_high, feature_dim, window_size


class DiffNormalizer:
  """Count-based running mean per diffusion timestep: equal-weighted, so it
  freezes as the count grows — a stable SDS-MSE reference, unlike a drifting EMA."""

  def __init__(
    self,
    num_timesteps: int,
    device: torch.device,
    min_value: float = 1e-4,
    max_count: int = 50_000_000,
  ) -> None:
    self.min_value = min_value
    self.max_count = max_count
    self.mean = torch.ones(num_timesteps, device=device)
    self.count = torch.zeros(num_timesteps, device=device, dtype=torch.long)

  def update_and_normalize(self, t: int, mse_per_env: torch.Tensor) -> torch.Tensor:
    """Record MSE values for timestep ``t``; return them divided by the mean."""
    if self.count[t] > self.max_count:
      # Freeze once stable (and avoid count overflow).
      return mse_per_env / self.mean[t].clamp(min=self.min_value)
    n = mse_per_env.numel()
    batch_mean = mse_per_env.mean()
    old_count = self.count[t].item()
    new_count = old_count + n
    if old_count == 0:
      self.mean[t] = batch_mean
    else:
      w_old = old_count / new_count
      w_new = n / new_count
      self.mean[t] = w_old * self.mean[t] + w_new * batch_mean
    self.count[t] = new_count
    return mse_per_env / self.mean[t].clamp(min=self.min_value)


class MotionFeatureBuffer:
  """Rolling per-env buffer of the last ``window_size`` kinematic samples;
  ``compute_features()`` returns a window anchored at the LAST frame's yaw-only
  local frame, layout (matching ``scripts/csv_to_npz.py``):

      ``[root_pos(3), root_rot(6), joint_pos(J), ee_pos(E*3),
         root_lin_vel(3), root_ang_vel(3)]``

  ``joint_vel`` is stored for symmetry but excluded from the output.  Positions
  use the caller's frame (SMP RL feeds env-origin-relative)."""

  def __init__(
    self,
    num_envs: int,
    window_size: int,
    num_joints: int,
    num_ee: int,
    device: torch.device | str,
  ) -> None:
    self.num_envs = num_envs
    self.window_size = window_size
    self.num_joints = num_joints
    self.num_ee = num_ee
    self.device = torch.device(device)

    self.root_pos_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_quat_w = torch.zeros(num_envs, window_size, 4, device=self.device)
    self.root_quat_w[..., 0] = 1.0
    self.root_lin_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_ang_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.ee_pos_w = torch.zeros(num_envs, window_size, num_ee, 3, device=self.device)
    self.joint_pos = torch.zeros(num_envs, window_size, num_joints, device=self.device)
    self.joint_vel = torch.zeros(num_envs, window_size, num_joints, device=self.device)

  def reset(
    self,
    env_ids: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    ee_pos_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Fill all W slots of ``env_ids`` with a pre-sampled trajectory."""
    if env_ids.numel() == 0:
      return
    self.root_pos_w[env_ids] = root_pos_w
    self.root_quat_w[env_ids] = root_quat_w
    self.root_lin_vel_w[env_ids] = root_lin_vel_w
    self.root_ang_vel_w[env_ids] = root_ang_vel_w
    self.ee_pos_w[env_ids] = ee_pos_w
    self.joint_pos[env_ids] = joint_pos
    self.joint_vel[env_ids] = joint_vel

  def update(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    ee_pos_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Shift left by one and append the new frame at index W-1."""
    self.root_pos_w = torch.roll(self.root_pos_w, shifts=-1, dims=1)
    self.root_quat_w = torch.roll(self.root_quat_w, shifts=-1, dims=1)
    self.root_lin_vel_w = torch.roll(self.root_lin_vel_w, shifts=-1, dims=1)
    self.root_ang_vel_w = torch.roll(self.root_ang_vel_w, shifts=-1, dims=1)
    self.ee_pos_w = torch.roll(self.ee_pos_w, shifts=-1, dims=1)
    self.joint_pos = torch.roll(self.joint_pos, shifts=-1, dims=1)
    self.joint_vel = torch.roll(self.joint_vel, shifts=-1, dims=1)
    self.root_pos_w[:, -1] = root_pos_w
    self.root_quat_w[:, -1] = root_quat_w
    self.root_lin_vel_w[:, -1] = root_lin_vel_w
    self.root_ang_vel_w[:, -1] = root_ang_vel_w
    self.ee_pos_w[:, -1] = ee_pos_w
    self.joint_pos[:, -1] = joint_pos
    self.joint_vel[:, -1] = joint_vel

  def compute_features(self) -> torch.Tensor:
    """Return features ``(num_envs, W, 3+6+J+E*3+3+3)``, all anchored to the LAST
    frame's yaw-only local frame (layout in the class docstring)."""
    N = self.num_envs
    W = self.window_size
    E = self.num_ee

    anchor_pos_T = self.root_pos_w[:, -1]
    anchor_quat_T = self.root_quat_w[:, -1]
    yaw_T = yaw_quat(anchor_quat_T)
    heading_inv_T = quat_conjugate(yaw_T)
    heading_inv_T_W = heading_inv_T[:, None, :].expand(N, W, 4)
    yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

    root_offset = self.root_pos_w - anchor_pos_T[:, None, :]
    root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(
      N, W, 3
    )
    root_pos_local = root_pos_local.clone()
    root_pos_local[..., 2] = self.root_pos_w[..., 2]

    # 6D rot is stacked [col0, col2] = [rotated-x-axis, rotated-z-axis].
    root_rot_local_quat = quat_mul(
      heading_inv_T_W.reshape(-1, 4),
      self.root_quat_w.reshape(-1, 4),
    ).reshape(N, W, 4)
    root_rot_mat = matrix_from_quat(root_rot_local_quat.reshape(-1, 4)).reshape(
      N, W, 3, 3
    )
    root_rot_6d = torch.cat([root_rot_mat[..., :, 0], root_rot_mat[..., :, 2]], dim=-1)

    ee_offset_w = self.ee_pos_w - self.root_pos_w[:, :, None, :]
    yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
    ee_pos_local = quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(
      N, W, E * 3
    )

    lin_vel_local = quat_apply_inverse(
      yaw_T_W, self.root_lin_vel_w.reshape(-1, 3)
    ).reshape(N, W, 3)
    ang_vel_local = quat_apply_inverse(
      yaw_T_W, self.root_ang_vel_w.reshape(-1, 3)
    ).reshape(N, W, 3)

    return torch.cat(
      [
        root_pos_local,
        root_rot_6d,
        self.joint_pos,
        ee_pos_local,
        lin_vel_local,
        ang_vel_local,
      ],
      dim=-1,
    )
