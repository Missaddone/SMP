"""Body-velocity experiment using the local-normalized LAFAN walk/run prior."""

from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .smp_env_cfg import CommandsCfg, EventCfg, PRETRAIN_CKPT_DIR, RewardsCfg
from .steering_env_cfg import SmpG1BodyVelocityEnvCfg


@configclass
class BodyVelocityLafanWalkRunEventCfg(EventCfg):
    """Load the local-normalized LAFAN walk/run prior."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "lafan_walk_run_local_nrom.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )


@configclass
class BodyVelocityLafanWalkRunCommandsCfg(CommandsCfg):
    """Commands matched to the walk/run prior without standing samples."""

    steering: CmdTerm = mdp.BodyVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        speed_min=0.25,
        speed_max=2.6,
        yaw_rate_min=-0.8,
        yaw_rate_max=0.8,
        stand_sample_prob=0.0,
    )


@configclass
class BodyVelocityLafanWalkRunRewardsCfg(RewardsCfg):
    """Keep task feedback alive while applying the walk/run style prior."""

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
class SmpG1BodyVelocityLafanWalkRunEnvCfg(SmpG1BodyVelocityEnvCfg):
    """BodyVelocity task for validating a single walk/run motion prior."""

    commands: BodyVelocityLafanWalkRunCommandsCfg = BodyVelocityLafanWalkRunCommandsCfg()
    events: BodyVelocityLafanWalkRunEventCfg = BodyVelocityLafanWalkRunEventCfg()
    rewards: BodyVelocityLafanWalkRunRewardsCfg = BodyVelocityLafanWalkRunRewardsCfg()
