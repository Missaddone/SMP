"""Body-velocity task with filtered hand-to-body self-collision penalties."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .smp_env_cfg import SmpSceneCfg
from .steering_env_cfg import BodyVelocityRewardsCfg, SmpG1BodyVelocityEnvCfg


CORE_AND_LEG_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
]


def _robot_paths(body_names: list[str]) -> list[str]:
    return [f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in body_names]


@configclass
class BodyVelocityHandCollisionSceneCfg(SmpSceneCfg):
    """Add one-to-many filtered contact sensors for both palms."""

    left_hand_body_contact = ContactSensorCfg(
        # The fixed rubber-hand link is merged into this rigid body by the URDF importer.
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
        filter_prim_paths_expr=_robot_paths(
            CORE_AND_LEG_BODY_NAMES
            + [
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
                "right_wrist_roll_link",
                "right_wrist_pitch_link",
                "right_wrist_yaw_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
            ]
        ),
        history_length=3,
    )
    right_hand_body_contact = ContactSensorCfg(
        # Its offset collision cylinder represents the right palm/hand contact geometry.
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
        filter_prim_paths_expr=_robot_paths(
            CORE_AND_LEG_BODY_NAMES
            + [
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_wrist_pitch_link",
                "left_wrist_yaw_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
            ]
        ),
        history_length=3,
    )


@configclass
class BodyVelocityHandCollisionRewardsCfg(BodyVelocityRewardsCfg):
    """Penalize either palm contacting the trunk, legs, or non-adjacent arm links."""

    left_hand_body_collision = RewTerm(
        func=mdp.filtered_contact_force_penalty,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("left_hand_body_contact"),
            "threshold": 2.0,
            "saturation_force": 20.0,
        },
    )
    right_hand_body_collision = RewTerm(
        func=mdp.filtered_contact_force_penalty,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("right_hand_body_contact"),
            "threshold": 2.0,
            "saturation_force": 20.0,
        },
    )


@configclass
class SmpG1BodyVelocityHandCollisionEnvCfg(SmpG1BodyVelocityEnvCfg):
    """Body-velocity task that teaches both hands to avoid the rest of the body."""

    scene: BodyVelocityHandCollisionSceneCfg = BodyVelocityHandCollisionSceneCfg(num_envs=4096, env_spacing=4.0)
    rewards: BodyVelocityHandCollisionRewardsCfg = BodyVelocityHandCollisionRewardsCfg()
