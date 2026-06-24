"""Body-velocity experiment using the AMP run prior."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from . import mdp
from .smp_env_cfg import EventCfg, PRETRAIN_CKPT_DIR
from .steering_env_cfg import SmpG1BodyVelocityEnvCfg


@configclass
class BodyVelocityAmpRunEventCfg(EventCfg):
    """Load the AMP run prior for a plain BodyVelocity task."""

    init_smp_state = EventTerm(
        func=mdp.init_smp_state,
        mode="startup",
        params={
            "ckpt_path": str(PRETRAIN_CKPT_DIR / "amp_run.pt"),
            "gsi_buffer_size": 4096,
            "gsi_batch_size": 1024,
        },
    )


@configclass
class SmpG1BodyVelocityAmpRunEnvCfg(SmpG1BodyVelocityEnvCfg):
    """Plain BodyVelocity task for checking whether the AMP run prior works."""

    events: BodyVelocityAmpRunEventCfg = BodyVelocityAmpRunEventCfg()
