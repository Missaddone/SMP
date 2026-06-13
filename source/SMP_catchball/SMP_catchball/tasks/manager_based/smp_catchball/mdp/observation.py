from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def joint_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to defaults, with optional encoder bias applied.

    EN: This mirrors mjlab's encoder model: the policy observes joint position
    after encoder bias, while the simulator state remains unbiased.
    中文：这里复刻 mjlab 的编码器模型：策略看到的是叠加 encoder bias 后的关节
    位置，但仿真内部真实关节状态不被修改。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    encoder_bias = getattr(env, "_smp_encoder_bias", None)
    if encoder_bias is not None:
        joint_pos = joint_pos + encoder_bias[:, asset_cfg.joint_ids]
    return joint_pos - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
