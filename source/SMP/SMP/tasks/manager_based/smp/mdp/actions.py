from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class BiasedJointPositionAction(JointPositionAction):
    """Joint position action that compensates the randomized encoder bias.

    EN: mjlab subtracts ``encoder_bias`` from the PD position target because the
    controller acts on the real joint while the policy observes a biased encoder.
    中文：mjlab 会从 PD 位置目标里减去 ``encoder_bias``，因为策略看到的是带
    编码器偏置的关节角，但控制器实际作用在真实关节上。
    """

    def apply_actions(self):
        target = self.processed_actions
        encoder_bias = getattr(self._env, "_smp_encoder_bias", None)
        if encoder_bias is not None:
            target = target - encoder_bias[:, self._joint_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`BiasedJointPositionAction`."""

    class_type: type = BiasedJointPositionAction
