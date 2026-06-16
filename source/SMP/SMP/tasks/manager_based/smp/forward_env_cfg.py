# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .smp_env_cfg import CommandsCfg, ObservationsCfg, RewardsCfg, SmpEnvCfg, TerminationsCfg


################################################################################
# Forward MDP components
################################################################################


########################################
# Commands
########################################


@configclass
class ForwardCommandsCfg(CommandsCfg):
    """Command config used by Smp-G1-Forward-v0."""

    steering: CmdTerm = mdp.SteeringCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rand_tar_dir=False,
        rand_face_dir=False,
        tar_speed_min=0.0,
        tar_speed_max=5.0,
        speed_deadzone=0.5,
    )


########################################
# Observations
########################################


@configclass
class ForwardObservationsCfg(ObservationsCfg):
    """Observation config used by forward and steering tasks."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "steering"})

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "steering"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


########################################
# Rewards
########################################


@configclass
class ForwardRewardsCfg(RewardsCfg):
    """Reward config used by Smp-G1-Forward-v0."""

    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.forward_task_smp_product,
        weight=1.0,
        params={
            "command_name": "steering",
            "vel_err_scale": 0.5,
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


########################################
# Terminations
########################################


@configclass
class ForwardTerminationsCfg(TerminationsCfg):
    """Termination config used by forward and steering tasks."""

    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )


################################################################################
# Forward task environment config
################################################################################


@configclass
class SmpG1ForwardEnvCfg(SmpEnvCfg):
    """G1 forward locomotion task with SMP guidance."""

    commands: ForwardCommandsCfg = ForwardCommandsCfg()
    observations: ForwardObservationsCfg = ForwardObservationsCfg()
    rewards: ForwardRewardsCfg = ForwardRewardsCfg()
    terminations: ForwardTerminationsCfg = ForwardTerminationsCfg()
