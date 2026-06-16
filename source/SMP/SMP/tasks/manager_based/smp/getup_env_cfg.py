# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .smp_env_cfg import EventCfg, PRETRAIN_CKPT_DIR, RewardsCfg, SmpEnvCfg, TerminationsCfg


################################################################################
# Getup MDP components
################################################################################


########################################
# Events
########################################


@configclass
class GetupEventCfg(EventCfg):
    """Event config used by Smp-G1-Getup-v0."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_getup_f2s2.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )
    reset_stand_counter = EventTerm(func=mdp.reset_stand_counter, mode="reset")


########################################
# Rewards
########################################


@configclass
class GetupRewardsCfg(RewardsCfg):
    """Reward config used by Smp-G1-Getup-v0."""

    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.getup_task_smp_product,
        weight=1.0,
        params={
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


########################################
# Terminations
########################################


@configclass
class GetupTerminationsCfg(TerminationsCfg):
    """Termination config used by Smp-G1-Getup-v0."""

    base_contact = None
    smp_too_low = DoneTerm(
        func=mdp.smp_too_low,
        params={"threshold": 0.02, "ws": 6.0, "grace_steps": 5},
    )
    stood_up = DoneTerm(
        func=mdp.stood_up,
        time_out=True,
        params={"head_height": 1.2, "max_speed": 0.5, "hold_steps": 25},
    )


################################################################################
# Getup task environment config
################################################################################


@configclass
class SmpG1GetupEnvCfg(SmpEnvCfg):
    """G1 getup task with SMP guidance."""

    events: GetupEventCfg = GetupEventCfg()
    rewards: GetupRewardsCfg = GetupRewardsCfg()
    terminations: GetupTerminationsCfg = GetupTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 5.0
