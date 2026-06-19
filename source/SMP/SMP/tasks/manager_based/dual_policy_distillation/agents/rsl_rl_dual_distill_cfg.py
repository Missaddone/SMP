# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BodyVelocityDualDistillRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Standard RSL-RL actor config used by simplified body-velocity dual-teacher BC distillation."""

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    logger = "tensorboard"
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "smp_body_velocity_dual_distill"
    teacher_0_checkpoint_path: str = ""
    teacher_1_checkpoint_path: str = ""
    selector_obs_group: str = "selector"
    selector_mode: str = "body_velocity_speed"
    selector_speed_threshold: float = 0.2
    max_grad_norm: float | None = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
