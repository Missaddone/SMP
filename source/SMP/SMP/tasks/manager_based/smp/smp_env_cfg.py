# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from SMP.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from SMP.smp.feature_to_state import G1_JOINT_NAMES

from . import mdp


# EN: Resolve checkpoints relative to this migrated project instead of the
# original Windows smp-master path. This keeps launches independent of cwd.
# 中文：预训练模型路径从当前迁移工程解析，不再使用原工程的 Windows 绝对路径；
# 这样从任意工作目录启动脚本都能找到 datasets/pretrain_ckpt。
PROJECT_ROOT = Path(__file__).resolve().parents[6]
PRETRAIN_CKPT_DIR = PROJECT_ROOT / "datasets" / "pretrain_ckpt"
G1_JOINT_NAMES_LIST = list(G1_JOINT_NAMES)


##
# Scene definition
##


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


##
# MDP settings
##


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
    # EN: This is trunk/base contact against anything, not pure Robot-vs-Robot self-collision.
    # 中文：这里检测的是 pelvis/torso 与任意物体的接触，不是严格的机器人自碰撞。
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(pelvis|torso_link)"), "threshold": 1.0},
    )


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


@configclass
class ForwardCommandsCfg(CommandsCfg):
    """Forward locomotion command configuration."""

    steering: CmdTerm = mdp.SteeringCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rand_tar_dir=False,
        rand_face_dir=False,
        tar_speed_min=0.0,
        tar_speed_max=5.0,
        speed_deadzone=0.5,
    )


@configclass
class SteeringCommandsCfg(CommandsCfg):
    """Random directional running command configuration."""

    steering: CmdTerm = mdp.SteeringCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rand_tar_dir=True,
        rand_face_dir=False,
        tar_speed_min=0.6,
        tar_speed_max=3.0,
        speed_deadzone=0.5,
    )


@configclass
class SteeringModifiedCommandsCfg(CommandsCfg):
    """Random steering commands with low-speed samples reserved for standing."""

    steering: CmdTerm = mdp.SteeringCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rand_tar_dir=True,
        rand_face_dir=False,
        tar_speed_min=0.0,
        tar_speed_max=3.0,
        speed_deadzone=0.5,
    )


@configclass
class ForwardObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "steering"})

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "steering"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ForwardRewardsCfg(RewardsCfg):
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


@configclass
class SteeringRewardsCfg(RewardsCfg):
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
    alive = None
    terminating = None
    smp = None
    task_smp_product = RewTerm(
        func=mdp.steering_modified_task_smp_product,
        weight=1.0,
        params={
            "command_name": "steering",
            "vel_err_scale": 1.0,
            "velocity_weight": 1.0,
            "face_weight": 0.5,
            "deadzone_stand_weight": 0.5,
            "deadzone_lin_vel_penalty_weight": 2.0,
            "deadzone_joint_vel_penalty_weight": 0.05,
            "deadzone_action_penalty_weight": 0.05,
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
class GetupEventCfg(EventCfg):
    """Events for the SMP getup task."""

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


@configclass
class GetupRewardsCfg(RewardsCfg):
    """Reward terms for the SMP getup task."""

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


@configclass
class GetupTerminationsCfg(TerminationsCfg):
    """Termination terms for the SMP getup task."""

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


@configclass
class SmpG1ForwardEnvCfg(SmpEnvCfg):
    """G1 forward locomotion task with SMP guidance."""

    commands: ForwardCommandsCfg = ForwardCommandsCfg()
    observations: ForwardObservationsCfg = ForwardObservationsCfg()
    rewards: ForwardRewardsCfg = ForwardRewardsCfg()
    terminations: ForwardTerminationsCfg = ForwardTerminationsCfg()


@configclass
class SteeringEventCfg(EventCfg):
    """Events for the random steering task."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_jushen.pt"),
            # "ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_lafan_run.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )


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
class SmpG1GetupEnvCfg(SmpEnvCfg):
    """G1 getup task with SMP guidance."""

    events: GetupEventCfg = GetupEventCfg()
    rewards: GetupRewardsCfg = GetupRewardsCfg()
    terminations: GetupTerminationsCfg = GetupTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 5.0
