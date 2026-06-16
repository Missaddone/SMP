# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from SMP.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from SMP.smp.feature_to_state import G1_JOINT_NAMES

from . import mdp


################################################################################
# Project paths and shared constants
################################################################################

# EN: Resolve checkpoints relative to this migrated project instead of the
# original Windows smp-master path. This keeps launches independent of cwd.
# 中文：预训练模型路径从当前迁移工程解析，不再使用原工程的 Windows 绝对路径；
# 这样从任意工作目录启动脚本都能找到 datasets/pretrain_ckpt。
PROJECT_ROOT = Path(__file__).resolve().parents[6]
PRETRAIN_CKPT_DIR = PROJECT_ROOT / "datasets" / "pretrain_ckpt"
G1_JOINT_NAMES_LIST = list(G1_JOINT_NAMES)


################################################################################
# Scene definition
################################################################################


@configclass
class SmpSceneCfg(InteractiveSceneCfg):
    """Base scene configuration for SMP G1 tasks."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


################################################################################
# Base MDP components
################################################################################


########################################
# Actions
########################################


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_JOINT_NAMES_LIST,
        preserve_order=True,
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
    )


########################################
# Commands
########################################


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pass


########################################
# Observations
########################################


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=G1_JOINT_NAMES_LIST, preserve_order=True),
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=G1_JOINT_NAMES_LIST, preserve_order=True),
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Uncorrupted critic observations with short history, matching the original SMP setup."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=G1_JOINT_NAMES_LIST, preserve_order=True),
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=G1_JOINT_NAMES_LIST, preserve_order=True),
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


########################################
# Events
########################################


@configclass
class EventCfg:
    """Configuration for events."""

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.03, 0.03)},
        },
    )
    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_JOINT_NAMES_LIST, preserve_order=True),
            "bias_range": (-0.015, 0.015),
        },
    )
    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            # EN: Forward task uses the locomotion prior by default.
            # 中文：Forward 任务默认使用 locomotion 预训练 prior。
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_loco.pt"),
            # "ckpt_path": str(PRETRAIN_CKPT_DIR / "checkpoint_01999.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )
    gsi_reset = EventTerm(func=mdp.gsi_reset, mode="reset")
    gsi_refresh = EventTerm(
        func=mdp.gsi_refresh,
        mode="interval",
        interval_range_s=(48.0, 48.0),
        is_global_time=True,
        params={"num_samples": 1024},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )


########################################
# Rewards
########################################


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


########################################
# Terminations
########################################


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # EN: This is trunk/base contact against anything, not pure Robot-vs-Robot self-collision.
    # 中文：这里检测的是 pelvis/torso 与任意物体的接触，不是严格的机器人自碰撞。
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(pelvis|torso_link)"), "threshold": 1.0},
    )


################################################################################
# Base environment config
################################################################################


@configclass
class SmpEnvCfg(ManagerBasedRLEnvCfg):
    """Base SMP environment config.

    Downstream tasks should inherit from this class and specialize commands,
    rewards, events, and terminations.
    """

    # Scene settings
    scene: SmpSceneCfg = SmpSceneCfg(num_envs=4096, env_spacing=4.0)
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
