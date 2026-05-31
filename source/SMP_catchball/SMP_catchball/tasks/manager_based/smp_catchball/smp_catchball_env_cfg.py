# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from SMP_catchball.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG

from . import mdp


##
# Scene definition
##


@configclass
class SmpCatchballSceneCfg(InteractiveSceneCfg):
    """Base scene configuration for SMP G1 tasks."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            "ckpt_path": "D:/OpenSource_Project/smp-master/datasets/pretrain_ckpt/pretrained_loco.pt",
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )
    gsi_reset = EventTerm(func=mdp.gsi_reset, mode="reset")
    gsi_refresh = EventTerm(
        func=mdp.gsi_refresh,
        mode="step",
        params={"num_samples": 1024, "step_interval": 2400},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Constant running reward. Task-specific environments should add their own
    # rewards by subclassing this base config.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    smp = RewTerm(
        func=mdp.smp_guidance_reward,
        weight=1.0,
        params={
            "fixed_timesteps": (8, 15, 22),
            "ws": 4.0,
            "normalize": True,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class SmpEnvCfg(ManagerBasedRLEnvCfg):
    """Base SMP environment config.

    Downstream tasks should inherit from this class and specialize commands,
    rewards, events, and terminations.
    """

    # Scene settings
    scene: SmpCatchballSceneCfg = SmpCatchballSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (3.0, 3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.8)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation


@configclass
class ForwardCommandsCfg(CommandsCfg):
    """Forward locomotion command configuration."""

    steering: CmdTerm = mdp.SteeringCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rand_tar_dir=False,
        rand_face_dir=False,
        tar_speed_min=0.5,
        tar_speed_max=5.0,
    )


@configclass
class ForwardObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "steering"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class ForwardRewardsCfg(RewardsCfg):
    smp = None
    task_smp_product = RewTerm(
        func=mdp.task_smp_product,
        weight=1.0,
        params={
            "task_terms": (
                (
                    mdp.steering_target_velocity,
                    1.0,
                    {"command_name": "steering", "vel_err_scale": 0.5},
                ),
            ),
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


@configclass
class ForwardTerminationsCfg(TerminationsCfg):
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class SmpG1ForwardEnvCfg(SmpEnvCfg):
    """G1 forward locomotion task with SMP guidance."""

    commands: ForwardCommandsCfg = ForwardCommandsCfg()
    observations: ForwardObservationsCfg = ForwardObservationsCfg()
    rewards: ForwardRewardsCfg = ForwardRewardsCfg()
    terminations: ForwardTerminationsCfg = ForwardTerminationsCfg()
