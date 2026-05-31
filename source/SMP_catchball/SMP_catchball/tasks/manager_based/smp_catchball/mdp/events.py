from __future__ import annotations

import torch

from SMP_catchball.smp.feature_to_state import EE_BODY_NAMES, NUM_EE, NUM_JOINTS, rot6d_to_quat, slice_features
from SMP_catchball.smp.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser, quat_apply, quat_mul, yaw_quat


def init_smp_state(
    env,
    env_ids: torch.Tensor | None = None,
    ckpt_path: str = "",
    gsi_buffer_size: int = 4096,
    gsi_batch_size: int = 1024,
) -> None:
    """Load the frozen SMP prior, allocate buffers, and pre-sample the GSI pool."""
    del env_ids
    if not ckpt_path:
        raise RuntimeError("init_smp_state requires a non-empty `ckpt_path`.")

    model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(ckpt_path, env.device)
    expected_dim = 3 + 6 + NUM_JOINTS + NUM_EE * 3 + 3 + 3
    if feature_dim != expected_dim:
        raise ValueError(f"SMP prior feature_dim={feature_dim}, expected {expected_dim} for G1.")

    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)
    if len(body_ids) != NUM_EE:
        raise RuntimeError(f"Expected SMP end-effectors {EE_BODY_NAMES}, but got body ids {body_ids}.")

    env._smp_bundle = (model, scheduler, q_low, q_high, feature_dim, window_size)
    env._smp_ee_indexes = torch.tensor(body_ids, dtype=torch.long, device=env.device)
    env._smp_buffer = MotionFeatureBuffer(
        num_envs=env.num_envs,
        window_size=window_size,
        num_joints=NUM_JOINTS,
        num_ee=NUM_EE,
        device=env.device,
    )
    env._smp_normalizer = DiffNormalizer(scheduler.num_timesteps, torch.device(env.device))
    env._smp_gsi_head = 0

    if gsi_buffer_size > 0:
        chunks: list[torch.Tensor] = []
        for start in range(0, gsi_buffer_size, gsi_batch_size):
            batch_size = min(gsi_batch_size, gsi_buffer_size - start)
            chunks.append(_ddpm_sample(env, batch_size))
        env._smp_gsi_pool = torch.cat(chunks, dim=0)
        gsi_reset(env)


def _control_dt(env) -> float:
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    return float(env.cfg.sim.dt) * float(env.cfg.decimation)


@torch.no_grad()
def _ddpm_sample(env, num_samples: int) -> torch.Tensor:
    """Sample denormalized SMP motion windows from the frozen prior."""
    model, scheduler, q_low, q_high, feature_dim, window_size = env._smp_bundle
    x_t = torch.randn(num_samples, window_size, feature_dim, device=env.device)
    for t_int in reversed(range(scheduler.num_timesteps)):
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=env.device)
        eps = model(x_t, t)
        x_t = scheduler.step(eps, x_t, t_int)
    return (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low


def _prime_sim_and_buffer(env, env_ids: torch.Tensor, window: torch.Tensor) -> None:
    """Write the sampled window's last frame to sim and fill the SMP feature buffer."""
    num_envs, window_size, _ = window.shape
    parts = slice_features(window)
    root_pos_local = parts["root_pos"]
    root_rot_6d = parts["root_rot"]
    joint_pos = parts["joint_pos"]
    ee_pos_local = parts["ee_pos"].reshape(num_envs, window_size, NUM_EE, 3)
    root_lin_vel_local = parts["root_lin_vel"]
    root_ang_vel_local = parts["root_ang_vel"]

    dt = _control_dt(env)
    if window_size > 1:
        joint_vel = torch.zeros_like(joint_pos)
        joint_vel[:, :-1] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / dt
        joint_vel[:, -1] = joint_vel[:, -2]
    else:
        joint_vel = torch.zeros_like(joint_pos)

    robot = env.scene["robot"]
    default_root = robot.data.default_root_state[env_ids].clone()
    default_pos = default_root[:, 0:3]
    default_quat = default_root[:, 3:7]
    yaw_t = yaw_quat(default_quat)
    yaw_t_w = yaw_t[:, None, :].expand(num_envs, window_size, 4).reshape(-1, 4)

    local_xy = root_pos_local.clone()
    local_xy[..., 2] = 0.0
    world_offset_xy = quat_apply(yaw_t_w, local_xy.reshape(-1, 3)).reshape(num_envs, window_size, 3)
    pelvis_pos_w = world_offset_xy.clone()
    pelvis_pos_w[..., 0] += default_pos[:, None, 0]
    pelvis_pos_w[..., 1] += default_pos[:, None, 1]
    pelvis_pos_w[..., 2] = root_pos_local[..., 2]

    root_rot_local_quat = rot6d_to_quat(root_rot_6d.reshape(-1, 6)).reshape(num_envs, window_size, 4)
    pelvis_quat_w = quat_mul(yaw_t_w, root_rot_local_quat.reshape(-1, 4)).reshape(num_envs, window_size, 4)

    lin_vel_w = quat_apply(yaw_t_w, root_lin_vel_local.reshape(-1, 3)).reshape(num_envs, window_size, 3)
    ang_vel_w = quat_apply(yaw_t_w, root_ang_vel_local.reshape(-1, 3)).reshape(num_envs, window_size, 3)

    yaw_t_ee = yaw_t[:, None, None, :].expand(num_envs, window_size, NUM_EE, 4).reshape(-1, 4)
    ee_offset_w = quat_apply(yaw_t_ee, ee_pos_local.reshape(-1, 3)).reshape(num_envs, window_size, NUM_EE, 3)
    ee_pos_w = ee_offset_w + pelvis_pos_w[:, :, None, :]

    origins = env.scene.env_origins[env_ids]
    last_root_state = torch.cat(
        [pelvis_pos_w[:, -1] + origins, pelvis_quat_w[:, -1], lin_vel_w[:, -1], ang_vel_w[:, -1]],
        dim=-1,
    )
    robot.write_root_state_to_sim(last_root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos[:, -1], joint_vel[:, -1], env_ids=env_ids)
    env._smp_buffer.reset(env_ids, pelvis_pos_w, pelvis_quat_w, lin_vel_w, ang_vel_w, ee_pos_w, joint_pos, joint_vel)


@torch.no_grad()
def gsi_reset(env, env_ids: torch.Tensor | None = None) -> None:
    """Generative State Initialization reset event."""
    if not hasattr(env, "_smp_gsi_pool"):
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return
    pool = env._smp_gsi_pool
    indices = torch.randint(0, pool.shape[0], (env_ids.numel(),), device=env.device)
    _prime_sim_and_buffer(env, env_ids, pool[indices])


@torch.no_grad()
def gsi_refresh(
    env,
    env_ids: torch.Tensor | None = None,
    num_samples: int = 1024,
    step_interval: int = 2400,
) -> None:
    """Periodically refresh part of the GSI sample pool."""
    del env_ids
    if not hasattr(env, "_smp_gsi_pool"):
        return
    step = int(getattr(env, "common_step_counter", 0))
    if step == 0 or (step % step_interval) != 0:
        return
    pool = env._smp_gsi_pool
    num_samples = min(num_samples, pool.shape[0])
    new_windows = _ddpm_sample(env, num_samples)
    head = int(getattr(env, "_smp_gsi_head", 0))
    end = head + num_samples
    if end <= pool.shape[0]:
        pool[head:end] = new_windows
    else:
        first = pool.shape[0] - head
        pool[head:] = new_windows[:first]
        pool[: end - pool.shape[0]] = new_windows[first:]
    env._smp_gsi_head = end % pool.shape[0]
