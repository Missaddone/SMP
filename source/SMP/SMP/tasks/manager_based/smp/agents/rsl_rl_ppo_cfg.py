# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    logger = "tensorboard"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "smp_forward_g1"
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
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class GetupPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_getup_g1"
    run_name = "smp_getup_g1"


@configclass
class SteeringPPORunnerCfg(PPORunnerCfg):
    experiment_name = "jushen_speed_06_30"


@configclass
class SteeringModifiedPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_steering_modified_g1"


@configclass
class SteeringWithStandPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_steering_with_stand_g1"
    run_name = "smp_steering_with_stand_g1"


@configclass
class BodyVelocityPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_body_velocity_g1"
    run_name = "smp_body_velocity_g1"


@configclass
class BodyVelocityLafanWalkPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_body_velocity_lafan_walk_g1"
    run_name = "smp_body_velocity_lafan_walk_g1"


@configclass
class BodyVelocityLafanWalkMatchedPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_body_velocity_lafan_walk_matched_g1"
    run_name = "smp_body_velocity_lafan_walk_matched_g1"


@configclass
class BodyVelocityLafanWalkPosturePPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_body_velocity_lafan_walk_posture_g1"
    run_name = "smp_body_velocity_lafan_walk_posture_g1"


@configclass
class BodyVelocityLafanWalkRunPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_body_velocity_lafan_walk_run_g1"
    run_name = "smp_body_velocity_lafan_walk_run_g1"


@configclass
class BodyVelocityAmpAllPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_body_velocity_amp_all_g1"
    run_name = "smp_body_velocity_amp_all_g1"


@configclass
class BodyVelocityHandCollisionPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_body_velocity_hand_collision_g1"
    run_name = "smp_body_velocity_hand_collision_g1"


@configclass
class BodyVelocityFootRegularizedPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_body_velocity_foot_regularized_g1"
    run_name = "smp_body_velocity_foot_regularized_g1"


@configclass
class ZeroVelocityPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_zero_velocity_g1"
    run_name = "smp_zero_velocity_g1"


@configclass
class SteeringDoublePriorPPORunnerCfg(PPORunnerCfg):
    experiment_name = "smp_steering_doubleprior_g1"
