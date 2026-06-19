# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from SMP.tasks.manager_based.smp import mdp
from SMP.tasks.manager_based.smp.forward_env_cfg import ForwardObservationsCfg, ForwardTerminationsCfg
from SMP.tasks.manager_based.smp.smp_env_cfg import EventCfg, RewardsCfg
from SMP.tasks.manager_based.smp.steering_env_cfg import SteeringWithStandCommandsCfg, SmpG1SteeringEnvCfg


@configclass
class BodyVelocityDistillationObservationsCfg:
    """Observation groups used by dual-policy body-velocity distillation."""

    policy: ForwardObservationsCfg.PolicyCfg = ForwardObservationsCfg.PolicyCfg()
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
    """Events for body-velocity distillation without SMP prior resets."""

    init_smp_state = None
    gsi_reset = None
    gsi_refresh = None
    push_robot = None


@configclass
class BodyVelocityDistillationRewardsCfg(RewardsCfg):
    """No reward terms are used by pure policy distillation."""

    alive = None
    terminating = None
    smp = None


@configclass
class G1BodyVelocityDualDistillationEnvCfg(SmpG1SteeringEnvCfg):
    """G1 body-velocity stand/move environment used for dual-policy distillation."""

    commands: SteeringWithStandCommandsCfg = SteeringWithStandCommandsCfg()
    observations: BodyVelocityDistillationObservationsCfg = BodyVelocityDistillationObservationsCfg()
    events: BodyVelocityDistillationEventCfg = BodyVelocityDistillationEventCfg()
    rewards: BodyVelocityDistillationRewardsCfg = BodyVelocityDistillationRewardsCfg()
    terminations: ForwardTerminationsCfg = ForwardTerminationsCfg()

