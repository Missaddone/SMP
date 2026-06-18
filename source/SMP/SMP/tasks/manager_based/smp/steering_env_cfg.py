# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .forward_env_cfg import SmpG1ForwardEnvCfg
from .smp_env_cfg import CommandsCfg, EventCfg, PRETRAIN_CKPT_DIR, RewardsCfg


################################################################################
# Steering MDP components
################################################################################


########################################
# Commands
########################################


@configclass
class SteeringCommandsCfg(CommandsCfg):
    """Command config used by Smp-G1-Steering-v0."""

    steering: CmdTerm = mdp.SteeringCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rand_tar_dir=True,
        rand_face_dir=False,
        tar_speed_min=0.6,
        tar_speed_max=5.0,
        speed_deadzone=0.5,
    )


@configclass
class SteeringModifiedCommandsCfg(CommandsCfg):
    """Command config used by Smp-G1-Steering-modified-v0."""

    steering: CmdTerm = mdp.SteeringCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rand_tar_dir=True,
        rand_face_dir=False,
        tar_speed_min=0.0,
        tar_speed_max=5.0,
        stand_sample_prob=0.3,
        stand_speed=0.0,
        run_speed_min=0.8,
        run_speed_max=5.0,
    )


########################################
# Events
########################################


@configclass
class SteeringEventCfg(EventCfg):
    """Event config used by steering tasks."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            # "ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_jushen.pt"),
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_lafan_run.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )


########################################
# Rewards
########################################


@configclass
class SteeringRewardsCfg(RewardsCfg):
    """Reward config used by Smp-G1-Steering-v0."""

    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.steering_task_smp_product,
        weight=1.0,
        params={
            "command_name": "steering",
            "vel_err_scale": 1.0,
            "velocity_weight": 0.5,
            "face_weight": 0.5,
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


@configclass
class SteeringModifiedRewardsCfg(RewardsCfg):
    """Reward config used by Smp-G1-Steering-modified-v0."""

    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.steering_modified_stand_branch_reward,
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
            "ws": 6.0,
        },
    )


################################################################################
# Steering task environment configs
################################################################################


@configclass
class SmpG1SteeringEnvCfg(SmpG1ForwardEnvCfg):
    """G1 random steering task with SMP guidance."""

    commands: SteeringCommandsCfg = SteeringCommandsCfg()
    events: SteeringEventCfg = SteeringEventCfg()
    rewards: SteeringRewardsCfg = SteeringRewardsCfg()


@configclass
class SmpG1SteeringModifiedEnvCfg(SmpG1SteeringEnvCfg):
    """Random steering task with a standing branch in the command deadzone."""

    commands: SteeringModifiedCommandsCfg = SteeringModifiedCommandsCfg()
    rewards: SteeringModifiedRewardsCfg = SteeringModifiedRewardsCfg()
