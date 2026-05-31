# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from SMP_catchball.smp.utils import DiffNormalizer, MotionFeatureBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def _data_attr(data, *names: str) -> torch.Tensor:
    for name in names:
        if hasattr(data, name):
            return getattr(data, name)
    raise AttributeError(f"None of these articulation data fields exist: {names}")


def _update_smp_buffer_from_sim(env: ManagerBasedRLEnv) -> None:
    robot: Articulation = env.scene["robot"]
    data = robot.data
    origins = env.scene.env_origins
    ee_ids = env._smp_ee_indexes
    buffer: MotionFeatureBuffer = env._smp_buffer

    root_pos_w = _data_attr(data, "root_link_pos_w", "root_pos_w") - origins
    root_quat_w = _data_attr(data, "root_link_quat_w", "root_quat_w")
    root_lin_vel_w = _data_attr(data, "root_link_lin_vel_w", "root_lin_vel_w")
    root_ang_vel_w = _data_attr(data, "root_link_ang_vel_w", "root_ang_vel_w")
    body_pos_w = _data_attr(data, "body_link_pos_w", "body_pos_w")
    ee_pos_w = body_pos_w[:, ee_ids] - origins[:, None, :]

    buffer.update(
        root_pos_w,
        root_quat_w,
        root_lin_vel_w,
        root_ang_vel_w,
        ee_pos_w,
        data.joint_pos,
        data.joint_vel,
    )


def smp_guidance_reward(
    env: ManagerBasedRLEnv,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 4.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute the frozen-prior SDS guidance reward for the current G1 motion window."""
    if not hasattr(env, "_smp_bundle"):
        raise RuntimeError("SMP prior is not initialized. Add init_smp_state as a startup event.")

    model, scheduler, q_low, q_high, _, _ = env._smp_bundle
    normalizer: DiffNormalizer = env._smp_normalizer
    buffer: MotionFeatureBuffer = env._smp_buffer
    _update_smp_buffer_from_sim(env)

    features = buffer.compute_features()
    x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
    num_envs = x_0.shape[0]
    device = x_0.device

    total_err = torch.zeros(num_envs, device=device)
    total_raw = torch.zeros(num_envs, device=device)
    with torch.no_grad():
        for t_scalar in fixed_timesteps:
            if not 0 <= t_scalar < scheduler.num_timesteps:
                raise ValueError(f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})")
            t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
            noise = torch.randn_like(x_0)
            x_t = scheduler.add_noise(x_0, noise, t)
            eps_hat = model(x_t, t)
            mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
            total_raw += mse_per_env
            if normalize:
                total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
            else:
                total_err += mse_per_env

    env._smp_raw_err = total_raw / len(fixed_timesteps)
    err = total_err / len(fixed_timesteps)
    return torch.exp(-err * ws)
