# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from SMP.tasks.manager_based.smp import mdp
from SMP.tasks.manager_based.smp.forward_env_cfg import ForwardObservationsCfg, ForwardTerminationsCfg
from SMP.tasks.manager_based.smp.smp_env_cfg import CommandsCfg, EventCfg, PRETRAIN_CKPT_DIR, RewardsCfg
from SMP.tasks.manager_based.smp.steering_env_cfg import (
    BodyVelocityObservationsCfg,
    SmpG1SteeringEnvCfg,
)


@configclass
class BodyVelocityDistillationCommandsCfg(CommandsCfg):
    """Body-velocity commands coupled to the dual-GSI reset type."""

    steering: CmdTerm = mdp.BodyVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        speed_min=0.3,
        speed_max=3.0,
        yaw_rate_min=-1.0,
        yaw_rate_max=1.0,
        stand_sample_prob=0.4,
        stand_speed_max=0.15,
        stand_yaw_rate_max=1.0,
        reset_stand_mask_attr="_distill_gsi_stand_mask",
    )


@configclass
class BodyVelocityDistillationObservationsCfg:
    """Observation groups used by dual-policy body-velocity distillation."""

    # Student policy must use deployable observations; teacher groups may keep
    # privileged base linear velocity to match existing teacher checkpoints.
    policy: BodyVelocityObservationsCfg.PolicyCfg = BodyVelocityObservationsCfg.PolicyCfg()
    teacher_0: ForwardObservationsCfg.PolicyCfg = ForwardObservationsCfg.PolicyCfg()
    teacher_1: ForwardObservationsCfg.PolicyCfg = ForwardObservationsCfg.PolicyCfg()

    @configclass
    class SelectorCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "steering"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    selector: SelectorCfg = SelectorCfg()


@configclass
class BodyVelocityDistillationEventCfg(EventCfg):
    """Dual-prior GSI events used only to diversify distillation rollouts."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_dual_gsi_state,
        mode="startup",
        params={
            "moving_ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_loco.pt"),
            "stand_ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_jushen_stand.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
            "stand_probability": 0.4,
            "reset_mask_attr": "_distill_gsi_stand_mask",
        },
    )
    gsi_reset = EventTerm(
        func=mdp.dual_gsi_reset,
        mode="reset",
        params={
            "stand_probability": 0.4,
            "reset_mask_attr": "_distill_gsi_stand_mask",
        },
    )
    gsi_refresh = EventTerm(
        func=mdp.dual_gsi_refresh,
        mode="interval",
        interval_range_s=(48.0, 48.0),
        is_global_time=True,
        params={"num_samples_per_prior": 512},
    )


@configclass
class BodyVelocityDistillationRewardsCfg(RewardsCfg):
    """No reward terms are used by pure policy distillation."""

    alive = None
    terminating = None
    smp = None


@configclass
class G1BodyVelocityDualDistillationEnvCfg(SmpG1SteeringEnvCfg):
    """G1 body-velocity stand/move environment used for dual-policy distillation."""

    commands: BodyVelocityDistillationCommandsCfg = BodyVelocityDistillationCommandsCfg()
    observations: BodyVelocityDistillationObservationsCfg = BodyVelocityDistillationObservationsCfg()
    events: BodyVelocityDistillationEventCfg = BodyVelocityDistillationEventCfg()
    rewards: BodyVelocityDistillationRewardsCfg = BodyVelocityDistillationRewardsCfg()
    terminations: ForwardTerminationsCfg = ForwardTerminationsCfg()
