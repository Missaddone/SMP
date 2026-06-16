from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from SMP.smp.feature_to_state import (
    EE_BODY_NAMES,
    G1_JOINT_NAMES,
    NUM_EE,
    NUM_JOINTS,
    rot6d_to_quat,
    slice_features,
)
from SMP.smp.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser, quat_apply, quat_mul, yaw_quat


@torch.no_grad()
def randomize_encoder_bias(
    env,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float] = (-0.015, 0.015),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize per-joint encoder bias, matching the original SMP domain randomization.

    EN: The bias is stored on the environment and consumed by the local
    observation/action terms. The simulator joint state itself is left unchanged.
    中文：偏置保存在环境对象上，由本地 observation/action term 使用；仿真里的
    真实关节状态本身不被修改。
    """
    robot = env.scene[asset_cfg.name]
    if not hasattr(env, "_smp_encoder_bias"):
        env._smp_encoder_bias = torch.zeros(env.num_envs, robot.num_joints, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    joint_ids = asset_cfg.joint_ids
    low, high = bias_range
    if isinstance(joint_ids, slice):
        num_joints = env._smp_encoder_bias[:, joint_ids].shape[1]
        env._smp_encoder_bias[env_ids, joint_ids] = torch.empty(
            env_ids.numel(), num_joints, device=env.device
        ).uniform_(low, high)
    else:
        joint_ids_tensor = torch.as_tensor(joint_ids, dtype=torch.long, device=env.device)
        samples = torch.empty(env_ids.numel(), joint_ids_tensor.numel(), device=env.device).uniform_(low, high)
        env._smp_encoder_bias[env_ids[:, None], joint_ids_tensor[None, :]] = samples


def init_smp_state(
    env,
    env_ids: torch.Tensor | None,
    ckpt_path: str = "",
    gsi_buffer_size: int = 4096,
    gsi_batch_size: int = 1024,
) -> None:
    """Load the frozen SMP prior, allocate buffers, and pre-sample the GSI pool.

    EN:
      IsaacLab EventManager treats the first two positional arguments as
      ``(env, env_ids)``. Keep ``env_ids`` required, even for startup events
      where it is passed as None, otherwise parameter validation fails.

    中文：
      IsaacLab 的 EventManager 会把事件函数前两个位置参数固定识别为
      ``(env, env_ids)``。即使 startup 事件传入的是 None，``env_ids`` 也
      不能写默认值，否则 manager 的参数校验会失败。
    """
    del env_ids
    if not ckpt_path:
        raise RuntimeError("init_smp_state requires a non-empty `ckpt_path`.")

    model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(ckpt_path, env.device)
    expected_dim = 3 + 6 + NUM_JOINTS + NUM_EE * 3 + 3 + 3
    if feature_dim != expected_dim:
        raise ValueError(f"SMP prior feature_dim={feature_dim}, expected {expected_dim} for G1.")

    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(list(G1_JOINT_NAMES), preserve_order=True)
    if len(joint_ids) != NUM_JOINTS:
        raise RuntimeError(f"Expected SMP joints {G1_JOINT_NAMES}, but got {joint_names}.")
    body_ids, _ = robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)
    if len(body_ids) != NUM_EE:
        raise RuntimeError(f"Expected SMP end-effectors {EE_BODY_NAMES}, but got body ids {body_ids}.")

    env._smp_bundle = (model, scheduler, q_low, q_high, feature_dim, window_size)
    env._smp_joint_indexes = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
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
        # EN: Prime all environments once after startup sampling.
        # 中文：startup 采样完成后，先用 GSI 给所有环境初始化一次。
        gsi_reset(env, None)


def init_smp_double_prior_state(
    env,
    env_ids: torch.Tensor | None,
    moving_ckpt_path: str = "",
    stand_ckpt_path: str = "",
    gsi_buffer_size: int = 4096,
    gsi_batch_size: int = 1024,
) -> None:
    """Load two SMP priors while keeping GSI on the moving prior.

    EN: The moving prior is installed as the default ``_smp_bundle`` and is used
    for GSI. The standing prior is stored in ``_smp_prior_bundles["stand"]`` and
    is intended for reward computation only.
    中文：moving prior 会作为默认 ``_smp_bundle``，继续负责 GSI。standing
    prior 只保存在 ``_smp_prior_bundles["stand"]`` 里，用于 reward 计算。
    """
    if not moving_ckpt_path:
        raise RuntimeError("init_smp_double_prior_state requires a non-empty `moving_ckpt_path`.")
    if not stand_ckpt_path:
        raise RuntimeError("init_smp_double_prior_state requires a non-empty `stand_ckpt_path`.")

    init_smp_state(
        env,
        env_ids,
        ckpt_path=moving_ckpt_path,
        gsi_buffer_size=gsi_buffer_size,
        gsi_batch_size=gsi_batch_size,
    )

    moving_bundle = env._smp_bundle
    moving_normalizer = env._smp_normalizer
    stand_model, stand_scheduler, stand_q_low, stand_q_high, stand_feature_dim, stand_window_size = load_denoiser(
        stand_ckpt_path, env.device
    )
    _, _, _, _, moving_feature_dim, moving_window_size = moving_bundle
    if stand_feature_dim != moving_feature_dim:
        raise ValueError(f"Stand SMP prior feature_dim={stand_feature_dim}, expected {moving_feature_dim}.")
    if stand_window_size != moving_window_size:
        raise ValueError(f"Stand SMP prior window_size={stand_window_size}, expected {moving_window_size}.")

    env._smp_prior_bundles = {
        "moving": moving_bundle,
        "stand": (stand_model, stand_scheduler, stand_q_low, stand_q_high, stand_feature_dim, stand_window_size),
    }
    env._smp_prior_normalizers = {
        "moving": moving_normalizer,
        "stand": DiffNormalizer(stand_scheduler.num_timesteps, torch.device(env.device)),
    }


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
    robot.write_joint_state_to_sim(
        joint_pos[:, -1],
        joint_vel[:, -1],
        joint_ids=env._smp_joint_indexes,
        env_ids=env_ids,
    )
    env._smp_buffer.reset(env_ids, pelvis_pos_w, pelvis_quat_w, lin_vel_w, ang_vel_w, ee_pos_w, joint_pos, joint_vel)


@torch.no_grad()
def gsi_reset(env, env_ids: torch.Tensor | None) -> None:
    """Generative State Initialization reset event.

    EN: ``env_ids=None`` means reset all environments.
    中文：``env_ids=None`` 表示重置全部环境。
    """
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
    env_ids: torch.Tensor | None,
    num_samples: int = 1024,
) -> None:
    """Periodically refresh part of the GSI sample pool.

    EN:
      IsaacLab does not automatically call arbitrary ``mode="step"`` events.
      This function is wired as a global ``mode="interval"`` event instead,
      so each trigger refreshes a chunk of the shared GSI pool.

    中文：
      IsaacLab 不会自动调用任意 ``mode="step"`` 事件。这里改用全局
      ``mode="interval"`` 定时事件；每次触发时刷新共享 GSI pool 的一部分。
    """
    del env_ids
    if not hasattr(env, "_smp_gsi_pool"):
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


@torch.no_grad()
def reset_stand_counter(env, env_ids: torch.Tensor | None) -> None:
    """Reset the getup success hold counter for selected environments."""
    if not hasattr(env, "_getup_stand_count"):
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env._getup_stand_count[env_ids] = 0
