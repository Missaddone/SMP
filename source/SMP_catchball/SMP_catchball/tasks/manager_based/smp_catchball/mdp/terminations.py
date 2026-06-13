from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _print_termination(name: str, values: torch.Tensor, debug_print: bool) -> None:
    if not debug_print:
        return
    env_ids = values.nonzero(as_tuple=False).flatten()
    if env_ids.numel() > 0:
        print(f"[SMP GETUP TERMINATION] {name}: env_ids={env_ids.detach().cpu().tolist()}")


def time_out_debug(env, debug_print: bool = False) -> torch.Tensor:
    """Terminate on episode timeout, optionally printing the triggering env ids."""
    values = env.episode_length_buf >= env.max_episode_length
    _print_termination("time_out", values, debug_print)
    return values


def root_height_below_minimum(env, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the robot root height falls below a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_link_pos_w if hasattr(asset.data, "root_link_pos_w") else asset.data.root_pos_w
    return root_pos[:, 2] < minimum_height


def _data_attr(data, *names: str) -> torch.Tensor:
    for name in names:
        if hasattr(data, name):
            return getattr(data, name)
    raise AttributeError(f"None of these articulation data fields exist: {names}")


def _virtual_head_pos_w(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    head_pos_in_torso: tuple[float, float, float] = (0.0, 0.0, 0.43),
) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply

    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_getup_head_body_indexes"):
        body_ids, body_names = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected exactly one getup head parent body, got {body_names}.")
        env._getup_head_body_indexes = torch.tensor(body_ids, dtype=torch.long, device=env.device)

    body_id = env._getup_head_body_indexes
    body_pos = _data_attr(asset.data, "body_link_pos_w", "body_pos_w")[:, body_id].squeeze(1)
    body_quat = _data_attr(asset.data, "body_link_quat_w", "body_quat_w")[:, body_id].squeeze(1)
    offset_b = torch.tensor(head_pos_in_torso, dtype=body_pos.dtype, device=body_pos.device).expand_as(body_pos)
    return body_pos + quat_apply(body_quat, offset_b)


def stood_up(
    env,
    head_height: float = 1.2,
    max_speed: float = 0.5,
    hold_steps: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    debug_print: bool = False,
) -> torch.Tensor:
    """Truncate after the virtual head stays high and root speed stays low."""
    asset: Articulation = env.scene[asset_cfg.name]
    head_pos = _virtual_head_pos_w(env)
    root_lin_vel = _data_attr(asset.data, "root_link_lin_vel_w", "root_lin_vel_w")
    speed = torch.linalg.norm(root_lin_vel, dim=-1)
    is_standing = (head_pos[:, 2] >= head_height) & (speed < max_speed)

    cnt = getattr(env, "_getup_stand_count", None)
    if cnt is None:
        cnt = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    cnt = torch.where(is_standing, cnt + 1, torch.zeros_like(cnt))
    env._getup_stand_count = cnt
    values = cnt >= hold_steps
    _print_termination("stood_up", values, debug_print)
    return values


def smp_too_low(
    env,
    threshold: float = 0.02,
    ws: float = 6.0,
    grace_steps: int = 15,
    debug_print: bool = False,
) -> torch.Tensor:
    """Terminate getup if the raw SMP score collapses after a short grace period."""
    raw_err = getattr(env, "_smp_raw_err", None)
    if raw_err is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    raw_smp = torch.exp(-ws * raw_err)
    past_grace = env.episode_length_buf >= grace_steps
    values = (raw_smp < threshold) & past_grace
    _print_termination("smp_too_low", values, debug_print)
    return values
