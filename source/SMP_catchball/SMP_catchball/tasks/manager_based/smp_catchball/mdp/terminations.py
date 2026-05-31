from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def root_height_below_minimum(env, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the robot root height falls below a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_link_pos_w if hasattr(asset.data, "root_link_pos_w") else asset.data.root_pos_w
    return root_pos[:, 2] < minimum_height
