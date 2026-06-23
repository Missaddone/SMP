"""BodyVelocity task with foot-contact regularization and the original motion prior."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .smp_env_cfg import SmpSceneCfg
from .steering_env_cfg import BodyVelocityRewardsCfg, SmpG1BodyVelocityEnvCfg


FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


@configclass
class BodyVelocityFootRegularizedSceneCfg(SmpSceneCfg):
    """Track foot contact and air durations in addition to contact forces."""

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )


@configclass
class BodyVelocityFootRegularizedRewardsCfg(BodyVelocityRewardsCfg):
    """Add foot-flatness, anti-slip, alternation, and action-smoothing terms."""

    feet_air_time = RewTerm(
        func=mdp.biped_feet_air_time_reward,
        weight=0.25,
        params={
            "command_name": "steering",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "threshold": 0.4,
            "command_speed_threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "contact_threshold": 1.0,
        },
    )
    support_foot_tilt = RewTerm(
        func=mdp.support_foot_tilt_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "contact_threshold": 1.0,
        },
    )
    persistent_single_support = RewTerm(
        func=mdp.persistent_single_support_penalty,
        weight=-0.2,
        params={
            "command_name": "steering",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "max_single_support_time": 0.6,
            "max_excess_time": 1.0,
            "command_speed_threshold": 0.15,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class SmpG1BodyVelocityFootRegularizedEnvCfg(SmpG1BodyVelocityEnvCfg):
    """Original BodyVelocity task augmented with targeted foot regularization."""

    scene: BodyVelocityFootRegularizedSceneCfg = BodyVelocityFootRegularizedSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
    )
    rewards: BodyVelocityFootRegularizedRewardsCfg = BodyVelocityFootRegularizedRewardsCfg()
