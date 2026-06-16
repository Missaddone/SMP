# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, wrap_to_pi

from SMP.smp.feature_to_state import G1_JOINT_NAMES, NUM_JOINTS
from SMP.smp.utils import DiffNormalizer, MotionFeatureBuffer

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
    env_mask: torch.Tensor | None = None,
    prior_name: str = "moving",
) -> torch.Tensor:
    """Compute the frozen-prior SDS guidance reward for the current G1 motion window."""
    if not hasattr(env, "_smp_bundle"):
        raise RuntimeError("SMP prior is not initialized. Add init_smp_state as a startup event.")

    if hasattr(env, "_smp_prior_bundles"):
        if prior_name not in env._smp_prior_bundles:
            raise RuntimeError(f"SMP prior '{prior_name}' is not initialized.")
        model, scheduler, q_low, q_high, _, _ = env._smp_prior_bundles[prior_name]
        normalizer: DiffNormalizer = env._smp_prior_normalizers[prior_name]
    else:
        model, scheduler, q_low, q_high, _, _ = env._smp_bundle
        normalizer: DiffNormalizer = env._smp_normalizer
    buffer: MotionFeatureBuffer = env._smp_buffer
    _update_smp_buffer_from_sim(env)

    features = buffer.compute_features()
    x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
    all_num_envs = x_0.shape[0]
    device = x_0.device
    if env_mask is not None:
        env_mask = env_mask.to(device=device, dtype=torch.bool)
        if env_mask.numel() != all_num_envs:
            raise ValueError(f"env_mask length {env_mask.numel()} does not match num_envs {all_num_envs}.")
        if not env_mask.any():
            env._smp_raw_err = torch.zeros(all_num_envs, device=device)
            return torch.ones(all_num_envs, device=device)
        active_mask = env_mask
        x_0_active = x_0[active_mask]
    else:
        active_mask = None
        x_0_active = x_0

    num_envs = x_0_active.shape[0]

    total_err = torch.zeros(num_envs, device=device)
    total_raw = torch.zeros(num_envs, device=device)
    with torch.no_grad():
        for t_scalar in fixed_timesteps:
            if not 0 <= t_scalar < scheduler.num_timesteps:
                raise ValueError(f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})")
            t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
            noise = torch.randn_like(x_0_active)
            x_t = scheduler.add_noise(x_0_active, noise, t)
            eps_hat = model(x_t, t)
            mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
            total_raw += mse_per_env
            if normalize:
                total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
            else:
                total_err += mse_per_env

    if active_mask is None:
        env._smp_raw_err = total_raw / len(fixed_timesteps)
    else:
        env._smp_raw_err = torch.zeros(all_num_envs, device=device)
        env._smp_raw_err[active_mask] = total_raw / len(fixed_timesteps)
    err = total_err / len(fixed_timesteps)
    reward = torch.exp(-err * ws)
    if active_mask is None:
        return reward
    full_reward = torch.ones(all_num_envs, device=device)
    full_reward[active_mask] = reward
    return full_reward


def task_smp_product(
    env: ManagerBasedRLEnv,
    task_terms: tuple[tuple[callable, float, dict], ...],
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
) -> torch.Tensor:
    """Multiplicative task reward gated by SMP guidance."""
    task = sum(weight * func(env, **kwargs) for func, weight, kwargs in task_terms)
    return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)


def _command_moving_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return True for envs whose effective reference speed is outside the deadzone."""
    command = env.command_manager.get_term(command_name)
    speed_deadzone = getattr(command.cfg, "speed_deadzone", 0.0)
    return command.tar_speed >= speed_deadzone


def _root_ang_vel_w(data) -> torch.Tensor:
    return data.root_link_ang_vel_w if hasattr(data, "root_link_ang_vel_w") else data.root_ang_vel_w


def standing_still_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_std: float = 0.15,
    lin_vel_scale: float = 2.0,
    ang_vel_scale: float = 0.5,
    action_scale: float = 0.1,
) -> torch.Tensor:
    """Reward quiet upright standing without using the SMP motion prior.

    EN: Use an additive blend instead of multiplying all terms. This keeps a
    usable learning signal when GSI starts an env from a moving pose.
    中文：这里不用各项相乘，而是加权求和。这样即使 GSI 把环境初始化到运动
    姿态，站立分支也不会因为某一项接近 0 而完全没梯度信号。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    if not hasattr(env, "_smp_joint_indexes"):
        joint_ids, joint_names = asset.find_joints(list(G1_JOINT_NAMES), preserve_order=True)
        if len(joint_ids) != NUM_JOINTS:
            raise RuntimeError(f"Expected SMP joints {G1_JOINT_NAMES}, but got {joint_names}.")
        env._smp_joint_indexes = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    joint_ids = env._smp_joint_indexes

    root_lin_vel_xy = _root_lin_vel_w(data)[:, :2]
    root_ang_vel = _root_ang_vel_w(data)
    joint_err = data.joint_pos[:, joint_ids] - data.default_joint_pos[:, joint_ids]
    pose = torch.exp(-torch.mean((joint_err / joint_std) ** 2, dim=-1))
    lin_quiet = torch.exp(-lin_vel_scale * torch.sum(root_lin_vel_xy**2, dim=-1))
    ang_quiet = torch.exp(-ang_vel_scale * torch.sum(root_ang_vel**2, dim=-1))
    upright = (-data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    action = env.action_manager.action
    action_regularization = torch.exp(-action_scale * torch.mean(action**2, dim=-1))
    return 0.30 * lin_quiet + 0.20 * ang_quiet + 0.25 * upright + 0.15 * pose + 0.10 * action_regularization


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


def steering_task_smp_product(
    env: ManagerBasedRLEnv,
    command_name: str = "steering",
    vel_err_scale: float = 1.0,
    velocity_weight: float = 1.5,
    face_weight: float = 0.5,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
) -> torch.Tensor:
    """Steering task reward gated by SMP guidance.

    EN: Matches the original SMP steering task: velocity tracking and facing
    direction alignment are blended before being multiplied by the SMP prior.
    中文：对齐原 SMP steering 任务：速度跟踪和朝向对齐先加权融合，再乘以
    SMP prior 风格项。
    """
    velocity = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
    face = steering_face_direction(env, command_name=command_name)
    # print("velocity reward:", velocity)
    # print("face reward:", face)
    task = velocity_weight * velocity + face_weight * face
    style = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
    # print("task reward:", task)
    # print("style reward:", style)
    return task * style


def steering_modified_task_smp_product(
    env: ManagerBasedRLEnv,
    command_name: str = "steering",
    vel_err_scale: float = 1.0,
    velocity_weight: float = 1.0,
    face_weight: float = 0.5,
    deadzone_stand_weight: float = 0.5,
    deadzone_lin_vel_penalty_weight: float = 2.0,
    deadzone_joint_vel_penalty_weight: float = 0.05,
    deadzone_action_penalty_weight: float = 0.05,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Steering reward with extra quiet-standing constraints in the deadzone.

    EN: The base task is always ``velocity_weight * velocity + face_weight *
    face``. For commands inside ``speed_deadzone`` we add a standing bonus and
    subtract stronger penalties for root xy velocity, joint velocity and action.
    The final layout is still ``task * style``.

    中文：基础 task 始终保持 ``velocity_weight * velocity + face_weight *
    face`` 不变。只有命令落在 ``speed_deadzone`` 内时，额外加入站立奖励，并
    对 root 水平线速度、关节速度和 action 加更强惩罚。最终结构仍然是
    ``task * style``。
    """
    moving_mask = _command_moving_mask(env, command_name)
    velocity = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
    face = steering_face_direction(env, command_name=command_name)
    task = velocity_weight * velocity + face_weight * face

    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_smp_joint_indexes"):
        joint_ids, joint_names = asset.find_joints(list(G1_JOINT_NAMES), preserve_order=True)
        if len(joint_ids) != NUM_JOINTS:
            raise RuntimeError(f"Expected SMP joints {G1_JOINT_NAMES}, but got {joint_names}.")
        env._smp_joint_indexes = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    joint_ids = env._smp_joint_indexes

    root_vel_xy = _root_lin_vel_w(asset.data)[:, :2]
    root_speed_sq = torch.sum(root_vel_xy**2, dim=-1)
    joint_vel_sq = torch.mean(asset.data.joint_vel[:, joint_ids] ** 2, dim=-1)
    action_sq = torch.mean(env.action_manager.action**2, dim=-1)
    deadzone_extra = deadzone_stand_weight * standing_still_reward(env)
    deadzone_extra = deadzone_extra - deadzone_lin_vel_penalty_weight * root_speed_sq
    deadzone_extra = deadzone_extra - deadzone_joint_vel_penalty_weight * joint_vel_sq
    deadzone_extra = deadzone_extra - deadzone_action_penalty_weight * action_sq

    task = torch.where(moving_mask, task, torch.clamp(task + deadzone_extra, min=0.0))
    return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)


def steering_modified_stand_branch_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "steering",
    vel_err_scale: float = 1.0,
    velocity_weight: float = 1.0,
    face_weight: float = 0.5,
    deadzone_stand_weight: float = 0.5,
    deadzone_lin_vel_penalty_weight: float = 2.0,
    deadzone_joint_vel_penalty_weight: float = 0.05,
    deadzone_action_penalty_weight: float = 0.05,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Use SMP style only for moving commands, and pure standing reward in the deadzone.

    EN: For commands outside ``speed_deadzone``, keep the original steering
    layout ``(velocity_weight * velocity + face_weight * face) * SMP``. For
    commands inside ``speed_deadzone``, do not use the SMP prior; instead use a
    quiet-standing reward with explicit penalties for root xy velocity, joint
    velocity, and action magnitude.

    中文：速度命令在死区外时，仍然使用原 steering 的
    ``(velocity_weight * velocity + face_weight * face) * SMP``。速度命令在
    死区内时，不使用 SMP prior，而是切换到站立静止奖励，并显式惩罚 root
    水平速度、关节速度和 action 幅值。
    """
    moving_mask = _command_moving_mask(env, command_name)
    velocity = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
    face = steering_face_direction(env, command_name=command_name)
    task = velocity_weight * velocity + face_weight * face

    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_smp_joint_indexes"):
        joint_ids, joint_names = asset.find_joints(list(G1_JOINT_NAMES), preserve_order=True)
        if len(joint_ids) != NUM_JOINTS:
            raise RuntimeError(f"Expected SMP joints {G1_JOINT_NAMES}, but got {joint_names}.")
        env._smp_joint_indexes = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    joint_ids = env._smp_joint_indexes

    root_vel_xy = _root_lin_vel_w(asset.data)[:, :2]
    root_speed_sq = torch.sum(root_vel_xy**2, dim=-1)
    joint_vel_sq = torch.mean(asset.data.joint_vel[:, joint_ids] ** 2, dim=-1)
    action_sq = torch.mean(env.action_manager.action**2, dim=-1)
    stand = deadzone_stand_weight * standing_still_reward(env)
    stand = stand - deadzone_lin_vel_penalty_weight * root_speed_sq
    stand = stand - deadzone_joint_vel_penalty_weight * joint_vel_sq
    stand = stand - deadzone_action_penalty_weight * action_sq
    stand = torch.clamp(stand, min=0.0)

    moving = task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws, env_mask=moving_mask)
    return torch.where(moving_mask, moving, stand)


def steering_doubleprior_task_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "steering",
    vel_err_scale: float = 1.0,
    velocity_weight: float = 1.0,
    face_weight: float = 0.5,
    deadzone_stand_weight: float = 0.5,
    deadzone_lin_vel_penalty_weight: float = 2.0,
    deadzone_joint_vel_penalty_weight: float = 0.05,
    deadzone_action_penalty_weight: float = 0.05,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    moving_ws: float = 6.0,
    stand_ws: float = 6.0,
    moving_prior_name: str = "moving",
    stand_prior_name: str = "stand",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Steering reward with separate moving and standing SMP priors.

    EN: Moving commands use the locomotion prior. Deadzone commands use a
    standing task reward multiplied by the standing prior. GSI still comes from
    the moving prior.
    中文：运动命令使用 locomotion prior；死区命令使用站立任务奖励乘 standing
    prior。GSI 仍由 moving prior 负责。
    """
    moving_mask = _command_moving_mask(env, command_name)
    velocity = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
    face = steering_face_direction(env, command_name=command_name)
    moving_task = velocity_weight * velocity + face_weight * face

    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_smp_joint_indexes"):
        joint_ids, joint_names = asset.find_joints(list(G1_JOINT_NAMES), preserve_order=True)
        if len(joint_ids) != NUM_JOINTS:
            raise RuntimeError(f"Expected SMP joints {G1_JOINT_NAMES}, but got {joint_names}.")
        env._smp_joint_indexes = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    joint_ids = env._smp_joint_indexes

    root_vel_xy = _root_lin_vel_w(asset.data)[:, :2]
    root_speed_sq = torch.sum(root_vel_xy**2, dim=-1)
    joint_vel_sq = torch.mean(asset.data.joint_vel[:, joint_ids] ** 2, dim=-1)
    action_sq = torch.mean(env.action_manager.action**2, dim=-1)
    stand_task = deadzone_stand_weight * standing_still_reward(env)
    stand_task = stand_task - deadzone_lin_vel_penalty_weight * root_speed_sq
    stand_task = stand_task - deadzone_joint_vel_penalty_weight * joint_vel_sq
    stand_task = stand_task - deadzone_action_penalty_weight * action_sq
    stand_task = torch.clamp(stand_task, min=0.0)

    moving_style = smp_guidance_reward(
        env,
        fixed_timesteps=fixed_timesteps,
        ws=moving_ws,
        env_mask=moving_mask,
        prior_name=moving_prior_name,
    )
    stand_style = smp_guidance_reward(
        env,
        fixed_timesteps=fixed_timesteps,
        ws=stand_ws,
        env_mask=~moving_mask,
        prior_name=stand_prior_name,
    )
    moving_reward = moving_task * moving_style
    stand_reward = stand_task * stand_style
    return torch.where(moving_mask, moving_reward, stand_reward)


def _virtual_head_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    head_pos_in_torso: tuple[float, float, float] = (0.0, 0.0, 0.43),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world position and velocity of the getup task's virtual head site."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_getup_head_body_indexes"):
        body_ids, body_names = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected exactly one getup head parent body, got {body_names}.")
        env._getup_head_body_indexes = torch.tensor(body_ids, dtype=torch.long, device=env.device)

    body_id = env._getup_head_body_indexes
    body_pos = _data_attr(asset.data, "body_link_pos_w", "body_pos_w")[:, body_id].squeeze(1)
    body_quat = _data_attr(asset.data, "body_link_quat_w", "body_quat_w")[:, body_id].squeeze(1)
    body_lin_vel = _data_attr(asset.data, "body_link_lin_vel_w", "body_lin_vel_w")[:, body_id].squeeze(1)
    body_ang_vel = _data_attr(asset.data, "body_link_ang_vel_w", "body_ang_vel_w")[:, body_id].squeeze(1)

    offset_b = torch.tensor(head_pos_in_torso, dtype=body_pos.dtype, device=body_pos.device).expand_as(body_pos)
    offset_w = quat_apply(body_quat, offset_b)
    head_pos = body_pos + offset_w
    head_vel = body_lin_vel + torch.cross(body_ang_vel, offset_w, dim=-1)
    return head_pos, head_vel


def track_head_height(
    env: ManagerBasedRLEnv,
    target_height: float = 1.2,
    scale: float = 6.0,
) -> torch.Tensor:
    """Reward the virtual head reaching target height, with no overshoot penalty."""
    head_pos, _ = _virtual_head_state(env)
    shortfall = torch.clamp(head_pos[:, 2] - target_height, max=0.0)
    return torch.exp(-scale * shortfall * shortfall)


def upward_velocity(
    env: ManagerBasedRLEnv,
    target_velocity: float = 0.25,
    head_height_threshold: float = 0.6,
    scale: float = 100.0,
) -> torch.Tensor:
    """Reward upward virtual-head velocity until the head has reached a useful height."""
    head_pos, head_vel = _virtual_head_state(env)
    shortfall = torch.clamp(head_vel[:, 2] - target_velocity, max=0.0)
    shaped = torch.exp(-scale * shortfall * shortfall)
    return torch.where(head_pos[:, 2] < head_height_threshold, shaped, torch.ones_like(shaped))


def getup_task_smp_product(
    env: ManagerBasedRLEnv,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    ws: float = 6.0,
) -> torch.Tensor:
    """Getup task reward gated by the frozen SMP guidance reward."""
    task = 0.7 * upward_velocity(
        env,
        target_velocity=0.25,
        head_height_threshold=0.9,
        scale=100.0,
    ) + 0.3 * track_head_height(env, target_height=1.1, scale=1.0)
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
    speed_deadzone = getattr(command.cfg, "speed_deadzone", 0.0)
    # print("target_speed:", command.tar_speed)
    # print("root_speed:", torch.linalg.norm(root_vel_xy, dim=-1))
    deadzone_speed_bias = -0.1
    tar_speed = torch.where(
        command.tar_speed < speed_deadzone,
        torch.full_like(command.tar_speed, deadzone_speed_bias),
        command.tar_speed,
    )
    tar_vel = tar_speed.unsqueeze(-1) * command.tar_dir_w
    vel_err = ((tar_vel - root_vel_xy) ** 2).sum(dim=-1)
    proj_speed = (command.tar_dir_w * root_vel_xy).sum(dim=-1)
    # 非负reward, 误差越小reward接近1，误差越大reward接近0
    reward = torch.exp(-vel_err_scale * vel_err)
    # 如果投影速度小于0，直接设置为0
    # return torch.where(proj_speed < 0.0, torch.zeros_like(reward), reward)
    # 速度不同号这设置为0，同时防止在0附近抖动
    eps = 1e-6
    wrong_direction = (torch.abs(tar_speed) > eps) & (torch.abs(proj_speed) > eps) & (tar_speed * proj_speed < 0.0)

    return torch.where(
        wrong_direction,
        torch.zeros_like(reward),
        reward,
    )


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
