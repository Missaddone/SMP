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
    """Load the local-normalized LAFAN walking prior."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "lafan_walk_500.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )


@configclass
class BodyVelocityLafanWalkCommandsCfg(CommandsCfg):
    """Commands restricted to the speed range represented by the walk prior."""

    steering: CmdTerm = mdp.BodyVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        speed_min=0.15,
        speed_max=1.6,
        yaw_rate_min=-0.8,
        yaw_rate_max=0.8,
        stand_sample_prob=0.0,
    )


@configclass
class BodyVelocityLafanWalkRewardsCfg(RewardsCfg):
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
            "style_floor": 0.0,
            "fixed_timesteps": (8, 15, 22),
            "ws": 6.0,
        },
    )


@configclass
class SmpG1BodyVelocityLafanWalkEnvCfg(SmpG1BodyVelocityEnvCfg):
    """BodyVelocity task matched to the local-normalized LAFAN walking prior."""

    commands: BodyVelocityLafanWalkCommandsCfg = BodyVelocityLafanWalkCommandsCfg()
    events: BodyVelocityLafanWalkEventCfg = BodyVelocityLafanWalkEventCfg()
    rewards: BodyVelocityLafanWalkRewardsCfg = BodyVelocityLafanWalkRewardsCfg()


# Backward-compatible alias for the previously registered matched task id.
BodyVelocityLafanWalkMatchedCommandsCfg = BodyVelocityLafanWalkCommandsCfg
BodyVelocityLafanWalkMatchedRewardsCfg = BodyVelocityLafanWalkRewardsCfg


@configclass
class SmpG1BodyVelocityLafanWalkMatchedEnvCfg(SmpG1BodyVelocityLafanWalkEnvCfg):
    """Alias of the matched local-normalized LAFAN walking task."""

    pass
