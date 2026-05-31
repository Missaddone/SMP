from __future__ import annotations

import torch

from SMP_catchball.smp.feature_to_state import EE_BODY_NAMES, NUM_EE, NUM_JOINTS
from SMP_catchball.smp.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser


def init_smp_state(
    env,
    env_ids: torch.Tensor | None = None,
    ckpt_path: str = "",
) -> None:
    """Load the frozen SMP prior and allocate online feature-state buffers."""
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
