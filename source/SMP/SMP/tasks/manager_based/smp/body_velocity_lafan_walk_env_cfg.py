"""Body-velocity experiment using the LAFAN walking motion prior."""

from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .smp_env_cfg import CommandsCfg, EventCfg, PRETRAIN_CKPT_DIR, RewardsCfg
from .steering_env_cfg import SmpG1BodyVelocityEnvCfg


@configclass
class BodyVelocityLafanWalkEventCfg(EventCfg):
    """Load the LAFAN walking prior while preserving the BodyVelocity setup."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "pretrained_lafan_walk.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )


@configclass
class SmpG1BodyVelocityLafanWalkEnvCfg(SmpG1BodyVelocityEnvCfg):
    """BodyVelocity task differing only in its LAFAN walking prior."""

    events: BodyVelocityLafanWalkEventCfg = BodyVelocityLafanWalkEventCfg()


@configclass
class BodyVelocityLafanWalkMatchedCommandsCfg(CommandsCfg):
    """Commands restricted to the speed range represented by the walk prior."""

    steering: CmdTerm = mdp.BodyVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        speed_min=0.15,
        speed_max=1.2,
        yaw_rate_min=-0.5,
        yaw_rate_max=0.5,
        stand_sample_prob=0.0,
    )


@configclass
class BodyVelocityLafanWalkMatchedRewardsCfg(RewardsCfg):
    """Preserve task feedback while softly applying the LAFAN walk style."""

    alive = None
    terminating = None
    smp = None
    task_smp_blend = RewTerm(
        func=mdp.body_velocity_task_smp_product,
        weight=1.0,
        params={
            "command_name": "steering",
            "lin_vel_err_scale": 2.0,
            "yaw_rate_err_scale": 1.0,
            "lin_vel_weight": 0.75,
            "yaw_rate_weight": 0.25,
            "use_stand_branch": False,
            "style_floor": 0.5,
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


@configclass
class SmpG1BodyVelocityLafanWalkMatchedEnvCfg(SmpG1BodyVelocityLafanWalkEnvCfg):
    """Walk-prior validation task with matched commands and non-vanishing task reward."""

    commands: BodyVelocityLafanWalkMatchedCommandsCfg = BodyVelocityLafanWalkMatchedCommandsCfg()
    rewards: BodyVelocityLafanWalkMatchedRewardsCfg = BodyVelocityLafanWalkMatchedRewardsCfg()
