"""LAFAN walk body-velocity task with light posture regularization."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .body_velocity_lafan_walk_env_cfg import BodyVelocityLafanWalkRewardsCfg, SmpG1BodyVelocityLafanWalkEnvCfg


@configclass
class BodyVelocityLafanWalkPostureRewardsCfg(BodyVelocityLafanWalkRewardsCfg):
    """Add targeted anti-hunching terms without constraining the legs."""

    base_upright = RewTerm(
        func=mdp.base_upright_penalty,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    root_height = RewTerm(
        func=mdp.root_height_below_target_penalty,
        weight=-0.2,
        params={
            "target_height": 0.74,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    waist_pitch_roll = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.1,
        params={
            "target": 0.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_roll_joint", "waist_pitch_joint"],
                preserve_order=True,
            ),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class SmpG1BodyVelocityLafanWalkPostureEnvCfg(SmpG1BodyVelocityLafanWalkEnvCfg):
    """LAFAN walk task variant for testing explicit anti-hunching rewards."""

    rewards: BodyVelocityLafanWalkPostureRewardsCfg = BodyVelocityLafanWalkPostureRewardsCfg()
