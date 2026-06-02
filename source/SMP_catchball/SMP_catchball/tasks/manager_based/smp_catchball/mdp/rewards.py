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

from SMP_catchball.smp.feature_to_state import G1_JOINT_NAMES, NUM_JOINTS
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
    if not hasattr(env, "_smp_joint_indexes"):
        joint_ids, joint_names = robot.find_joints(list(G1_JOINT_NAMES), preserve_order=True)
        if len(joint_ids) != NUM_JOINTS:
            raise RuntimeError(f"Expected SMP joints {G1_JOINT_NAMES}, but got {joint_names}.")
        env._smp_joint_indexes = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    joint_ids = env._smp_joint_indexes
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
        data.joint_pos[:, joint_ids],
        data.joint_vel[:, joint_ids],
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


def task_smp_product(
    env: ManagerBasedRLEnv,
    task_terms: tuple[tuple[callable, float, dict], ...],
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
) -> torch.Tensor:
    """Multiplicative task reward gated by SMP guidance."""
    task = sum(weight * func(env, **kwargs) for func, weight, kwargs in task_terms)
    return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)


def forward_task_smp_product(
    env: ManagerBasedRLEnv,
    command_name: str = "steering",
    vel_err_scale: float = 0.5,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
) -> torch.Tensor:
    """Forward task reward gated by the frozen SMP guidance reward.

    EN:
      Hydra/OmegaConf cannot serialize Python function objects in config
      params. The generic ``task_smp_product`` accepts callables, but training
      configs need primitive values only; this wrapper keeps Forward trainable
      through Hydra.

    中文：
      Hydra/OmegaConf 不能在配置参数里序列化 Python function。通用的
      ``task_smp_product`` 可以接收 callable，但训练配置必须只包含基础
      类型；这个 wrapper 让 Forward 任务可以通过 Hydra 正常训练。
    """
    task = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
    return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)


def _root_lin_vel_w(data) -> torch.Tensor:
    return data.root_link_lin_vel_w if hasattr(data, "root_link_lin_vel_w") else data.root_lin_vel_w


def _heading_w(data) -> torch.Tensor:
    if hasattr(data, "heading_w"):
        return data.heading_w
    quat = data.root_link_quat_w if hasattr(data, "root_link_quat_w") else data.root_quat_w
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def steering_target_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    vel_err_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded world-frame xy velocity, zeroing backwards motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    root_vel_xy = _root_lin_vel_w(asset.data)[:, :2]
    tar_vel = command.tar_speed.unsqueeze(-1) * command.tar_dir_w
    vel_err = ((tar_vel - root_vel_xy) ** 2).sum(dim=-1)
    proj_speed = (command.tar_dir_w * root_vel_xy).sum(dim=-1)
    reward = torch.exp(-vel_err_scale * vel_err)
    return torch.where(proj_speed < 0.0, torch.zeros_like(reward), reward)


def steering_face_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment between commanded face direction and robot heading."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    heading_w = _heading_w(asset.data)
    char_face_w = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
    return (command.face_dir_w * char_face_w).sum(dim=-1).clamp_min(0.0)
