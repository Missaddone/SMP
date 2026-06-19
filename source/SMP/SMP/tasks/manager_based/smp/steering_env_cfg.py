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
        rand_face_dir=True,
        tar_speed_min=0.5,
        tar_speed_max=2.0,
        debug_vis=True,
    )


@configclass
class SteeringModifiedCommandsCfg(CommandsCfg):
    """Command config used by Smp-G1-Steering-modified-v0."""

    steering: CmdTerm = mdp.SteeringCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rand_tar_dir=True,
        rand_face_dir=False,
        tar_speed_min=-0.2,
        tar_speed_max=5.0,
        stand_sample_prob=0.1,
        stand_speed=0.0,
        run_speed_min=0.4,
        run_speed_max=5.0,
    )


@configclass
class SteeringWithStandCommandsCfg(CommandsCfg):
    """Command config used by Smp-G1-Steering-with-stand-v0."""

    steering: CmdTerm = mdp.BodyVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        speed_min=0.3,
        speed_max=3.0,
        yaw_rate_min=-1.0,
        yaw_rate_max=1.0,
        stand_sample_prob=0.2,
        stand_speed_max=0.15,
        stand_yaw_rate_max=0.2,
    )


@configclass
class BodyVelocityCommandsCfg(CommandsCfg):
    """Command config used by Smp-G1-BodyVelocity-v0."""

    steering: CmdTerm = mdp.BodyVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        speed_min=0.5,
        speed_max=3.0,
        yaw_rate_min=-1.0,
        yaw_rate_max=1.0,
        stand_sample_prob=0.0,
    )


@configclass
class ZeroVelocityCommandsCfg(CommandsCfg):
    """Command config used by Smp-G1-ZeroVelocity-v0."""

    steering: CmdTerm = mdp.BodyVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        speed_min=0.0,
        speed_max=0.0,
        yaw_rate_min=0.0,
        yaw_rate_max=0.0,
        stand_sample_prob=0.0,
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


@configclass
class ZeroVelocityEventCfg(EventCfg):
    """Load the standing style prior for Smp-G1-ZeroVelocity-v0."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_jushen_stand.pt"),
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
            "deadzone_stand_weight": 0.7,
            "deadzone_lin_vel_penalty_weight": 2.0,
            "deadzone_joint_vel_penalty_weight": 0.07,
            "deadzone_action_penalty_weight": 0.07,
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


@configclass
class SteeringWithStandRewardsCfg(RewardsCfg):
    """Reward config used by Smp-G1-Steering-with-stand-v0."""

    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.body_velocity_task_smp_product,
        weight=1.0,
        params={
            "command_name": "steering",
            "lin_vel_err_scale": 2.0,
            "yaw_rate_err_scale": 1.0,
            "lin_vel_weight": 0.75,
            "yaw_rate_weight": 0.25,
            "stand_speed_threshold": 0.2,
            "stand_yaw_rate_threshold": 0.2,
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


@configclass
class BodyVelocityRewardsCfg(RewardsCfg):
    """Reward config used by Smp-G1-BodyVelocity-v0."""

    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.body_velocity_task_smp_product,
        weight=1.0,
        params={
            "command_name": "steering",
            "lin_vel_err_scale": 2.0,
            "yaw_rate_err_scale": 1.0,
            "lin_vel_weight": 0.75,
            "yaw_rate_weight": 0.25,
            "use_stand_branch": False,
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


@configclass
class ZeroVelocityRewardsCfg(RewardsCfg):
    """Zero-velocity task reward gated by the standing SMP prior."""

    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.body_velocity_task_smp_product,
        weight=1.0,
        params={
            "command_name": "steering",
            "lin_vel_err_scale": 2.0,
            "yaw_rate_err_scale": 1.0,
            "lin_vel_weight": 0.75,
            "yaw_rate_weight": 0.25,
            "use_stand_branch": False,
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


@configclass
class SmpG1SteeringWithStandEnvCfg(SmpG1SteeringEnvCfg):
    """Deployable body-velocity steering task.

    EN: The command is body-frame ``[v_x, v_y, yaw_rate]`` so it does not depend
    on a world-fixed target direction or face direction. Low-speed commands use
    an explicit standing reward branch.
    中文：command 是机体系 ``[v_x, v_y, yaw_rate]``，不再依赖世界坐标系下固定
    的速度/朝向目标。低速命令使用显式站立奖励分支。
    """

    commands: SteeringWithStandCommandsCfg = SteeringWithStandCommandsCfg()
    rewards: SteeringWithStandRewardsCfg = SteeringWithStandRewardsCfg()


@configclass
class SmpG1BodyVelocityEnvCfg(SmpG1SteeringEnvCfg):
    """Deployable body-velocity steering task without a standing branch."""

    commands: BodyVelocityCommandsCfg = BodyVelocityCommandsCfg()
    rewards: BodyVelocityRewardsCfg = BodyVelocityRewardsCfg()


@configclass
class SmpG1ZeroVelocityEnvCfg(SmpG1SteeringEnvCfg):
    """Zero-velocity task using a dedicated standing SMP style prior."""

    commands: ZeroVelocityCommandsCfg = ZeroVelocityCommandsCfg()
    events: ZeroVelocityEventCfg = ZeroVelocityEventCfg()
    rewards: ZeroVelocityRewardsCfg = ZeroVelocityRewardsCfg()
