# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .smp_env_cfg import PRETRAIN_CKPT_DIR, RewardsCfg
from .steering_env_cfg import SteeringEventCfg, SteeringModifiedCommandsCfg, SmpG1SteeringEnvCfg


################################################################################
# Steering double-prior MDP components
################################################################################


########################################
# Events
########################################


@configclass
class SteeringDoublePriorEventCfg(SteeringEventCfg):
    """Load moving and standing SMP priors for Smp-G1-Steering-doubleprior-v0."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_double_prior_state,
        mode="startup",
        params={
            "moving_ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_lafan_run.pt"),
            # EN: Replace this with a dedicated standing prior when available.
            # 中文：如果后面有专门的 stand prior，把这里替换成对应 checkpoint。
            "stand_ckpt_path": str(PRETRAIN_CKPT_DIR / "checkpoint_01999.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )


########################################
# Rewards
########################################


@configclass
class SteeringDoublePriorRewardsCfg(RewardsCfg):
    """Reward config using moving prior for motion and standing prior in deadzone."""

    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.steering_doubleprior_task_reward,
        weight=1.0,
        params={
            "command_name": "steering",
            "vel_err_scale": 1.0,
            "velocity_weight": 1.5,
            "face_weight": 0.5,
            "deadzone_face_weight": 0.0,
            "deadzone_stand_weight": 1.0,
            "deadzone_lin_vel_penalty_weight": 2.0,
            "deadzone_joint_vel_penalty_weight": 0.1,
            "deadzone_action_penalty_weight": 0.1,
            "fixed_timesteps": (8, 15, 22),
            "moving_ws": 6.0,
            "stand_ws": 6.0,
            "moving_prior_name": "moving",
            "stand_prior_name": "stand",
        },
    )


################################################################################
# Steering double-prior task environment config
################################################################################


@configclass
class SmpG1SteeringDoublePriorEnvCfg(SmpG1SteeringEnvCfg):
    """Random steering task with separate moving and standing SMP priors."""

    commands: SteeringModifiedCommandsCfg = SteeringModifiedCommandsCfg()
    events: SteeringDoublePriorEventCfg = SteeringDoublePriorEventCfg()
    rewards: SteeringDoublePriorRewardsCfg = SteeringDoublePriorRewardsCfg()
