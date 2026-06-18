from __future__ import annotations

import math
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass


def _heading_w(robot_data) -> torch.Tensor:
    if hasattr(robot_data, "heading_w"):
        return robot_data.heading_w
    quat = robot_data.root_link_quat_w if hasattr(robot_data, "root_link_quat_w") else robot_data.root_quat_w
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _dir_world_to_local(direction_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
    cos_h = torch.cos(heading_w)
    sin_h = torch.sin(heading_w)
    x_w, y_w = direction_w[..., 0], direction_w[..., 1]
    return torch.stack([cos_h * x_w + sin_h * y_w, -sin_h * x_w + cos_h * y_w], dim=-1)


def _deadzone_bounds(cfg: "SteeringCommandCfg") -> tuple[float, float]:
    lower = cfg.speed_deadzone_min
    upper = cfg.speed_deadzone_max
    if lower is None and upper is None:
        lower = cfg.tar_speed_min
        upper = cfg.speed_deadzone
    elif lower is None:
        lower = cfg.tar_speed_min
    elif upper is None:
        upper = cfg.speed_deadzone
    return lower, upper


def _has_two_bin_speed_sampling(cfg: "SteeringCommandCfg") -> bool:
    return cfg.stand_sample_prob > 0.0 and cfg.run_speed_min is not None and cfg.run_speed_max is not None


class SteeringCommand(CommandTerm):
    """Periodic target xy velocity and facing direction command."""

    cfg: "SteeringCommandCfg"

    def __init__(self, cfg: "SteeringCommandCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.tar_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.face_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.tar_speed = torch.zeros(self.num_envs, device=self.device)
        self.command_b = torch.zeros(self.num_envs, 5, device=self.device)
        self.tar_dir_w[:, 0] = 1.0
        self.face_dir_w[:, 0] = 1.0
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_face"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    def _update_metrics(self) -> None:
        root_vel = self.robot.data.root_link_lin_vel_w if hasattr(self.robot.data, "root_link_lin_vel_w") else self.robot.data.root_lin_vel_w
        tar_vel_w = self.tar_speed.unsqueeze(-1) * self.tar_dir_w
        self.metrics["error_vel_xy"] = torch.linalg.norm(tar_vel_w - root_vel[:, :2], dim=-1)
        heading_w = _heading_w(self.robot.data)
        char_face_w = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
        self.metrics["error_face"] = 1.0 - (self.face_dir_w * char_face_w).sum(dim=-1).clamp(-1.0, 1.0)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        num_envs = int(env_ids.numel())
        if num_envs == 0:
            return
        if self.cfg.rand_tar_dir:
            theta = torch.empty(num_envs, device=self.device).uniform_(-math.pi, math.pi)
        else:
            theta = torch.zeros(num_envs, device=self.device)
        self.tar_dir_w[env_ids, 0] = torch.cos(theta)
        self.tar_dir_w[env_ids, 1] = torch.sin(theta)

        if _has_two_bin_speed_sampling(self.cfg):
            run_speed_min = max(float(self.cfg.run_speed_min), self.cfg.tar_speed_min)
            run_speed_max = min(float(self.cfg.run_speed_max), self.cfg.tar_speed_max)
            if run_speed_max <= run_speed_min:
                raise ValueError(f"Invalid run speed interval [{run_speed_min}, {run_speed_max}].")
            stand_mask = torch.rand(num_envs, device=self.device) < self.cfg.stand_sample_prob
            speeds = torch.empty(num_envs, device=self.device)
            if stand_mask.any():
                speeds[stand_mask] = self.cfg.stand_speed
            if (~stand_mask).any():
                speeds[~stand_mask] = torch.empty(int((~stand_mask).sum()), device=self.device).uniform_(
                    run_speed_min,
                    run_speed_max,
                )
            self.tar_speed[env_ids] = speeds

        else:
            deadzone_min, deadzone_max = _deadzone_bounds(self.cfg)
            deadzone_min = max(deadzone_min, self.cfg.tar_speed_min)
            deadzone_max = min(deadzone_max, self.cfg.tar_speed_max)
            has_deadzone = deadzone_max > deadzone_min
            if self.cfg.deadzone_sample_prob > 0.0 and has_deadzone:
                deadzone_mask = torch.rand(num_envs, device=self.device) < self.cfg.deadzone_sample_prob
                speeds = torch.empty(num_envs, device=self.device)
                if deadzone_mask.any():
                    speeds[deadzone_mask] = torch.empty(int(deadzone_mask.sum()), device=self.device).uniform_(
                        deadzone_min,
                        deadzone_max,
                    )
                if (~deadzone_mask).any():
                    moving_count = int((~deadzone_mask).sum())
                    moving_speeds = torch.empty(moving_count, device=self.device)
                    lower_len = max(deadzone_min - self.cfg.tar_speed_min, 0.0)
                    upper_len = max(self.cfg.tar_speed_max - deadzone_max, 0.0)
                    if lower_len == 0.0 and upper_len == 0.0:
                        moving_speeds.uniform_(deadzone_min, deadzone_max)
                    elif lower_len > 0.0 and upper_len > 0.0:
                        lower_mask = torch.rand(moving_count, device=self.device) < lower_len / (
                            lower_len + upper_len
                        )
                        if lower_mask.any():
                            moving_speeds[lower_mask] = torch.empty(
                                int(lower_mask.sum()), device=self.device
                            ).uniform_(
                                self.cfg.tar_speed_min,
                                deadzone_min,
                            )
                        if (~lower_mask).any():
                            moving_speeds[~lower_mask] = torch.empty(
                                int((~lower_mask).sum()), device=self.device
                            ).uniform_(
                                deadzone_max,
                                self.cfg.tar_speed_max,
                            )
                    elif lower_len > 0.0:
                        moving_speeds.uniform_(self.cfg.tar_speed_min, deadzone_min)
                    else:
                        moving_speeds.uniform_(deadzone_max, self.cfg.tar_speed_max)
                    speeds[~deadzone_mask] = moving_speeds
                self.tar_speed[env_ids] = speeds
            else:
                self.tar_speed[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
                    self.cfg.tar_speed_min,
                    self.cfg.tar_speed_max,
                )
        if self.cfg.rand_face_dir:
            face_theta = torch.empty(num_envs, device=self.device).uniform_(-math.pi, math.pi)
        else:
            face_theta = theta
        self.face_dir_w[env_ids, 0] = torch.cos(face_theta)
        self.face_dir_w[env_ids, 1] = torch.sin(face_theta)

    def _update_command(self) -> None:
        heading_w = _heading_w(self.robot.data)
        self.command_b[:, 0:2] = _dir_world_to_local(self.tar_dir_w, heading_w)
        self.command_b[:, 2] = self.tar_speed
        self.command_b[:, 3:5] = _dir_world_to_local(self.face_dir_w, heading_w)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        root_pos = self.robot.data.root_link_pos_w if hasattr(self.robot.data, "root_link_pos_w") else self.robot.data.root_pos_w
        root_vel = (
            self.robot.data.root_link_lin_vel_w
            if hasattr(self.robot.data, "root_link_lin_vel_w")
            else self.robot.data.root_lin_vel_w
        )
        marker_pos = root_pos.clone()
        marker_pos[:, 2] += self.cfg.viz_z_offset

        target_vel_w = torch.zeros(self.num_envs, 2, device=self.device)
        target_vel_w[:, :2] = self.tar_speed.unsqueeze(-1) * self.tar_dir_w
        goal_scale, goal_quat = self._resolve_xy_velocity_to_arrow(target_vel_w)
        current_scale, current_quat = self._resolve_xy_velocity_to_arrow(root_vel[:, :2])

        self.goal_vel_visualizer.visualize(marker_pos, goal_quat, goal_scale)
        self.current_vel_visualizer.visualize(marker_pos, current_quat, current_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity_w.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity_w, dim=-1) * self.cfg.viz_scale

        heading_angle = torch.atan2(xy_velocity_w[:, 1], xy_velocity_w[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, arrow_quat


@configclass
class SteeringCommandCfg(CommandTermCfg):
    class_type: type = SteeringCommand
    asset_name: str = MISSING
    rand_tar_dir: bool = True
    rand_face_dir: bool = True
    tar_speed_min: float = 0.5
    tar_speed_max: float = 3.0
    # Legacy one-sided deadzone: [tar_speed_min, speed_deadzone).
    speed_deadzone: float = 0.0
    # Optional explicit interval deadzone: [speed_deadzone_min, speed_deadzone_max].
    speed_deadzone_min: float | None = None
    speed_deadzone_max: float | None = None
    deadzone_sample_prob: float = 0.0
    # Optional two-bin command sampling: stand at a fixed speed, otherwise run in [run_speed_min, run_speed_max].
    stand_sample_prob: float = 0.0
    stand_speed: float = 0.0
    stand_speed_tolerance: float = 1e-4
    run_speed_min: float | None = None
    run_speed_max: float | None = None
    viz_z_offset: float = 0.7
    viz_scale: float = 1.5
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
