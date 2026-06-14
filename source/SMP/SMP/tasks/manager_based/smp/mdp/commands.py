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
    speed_deadzone: float = 0.0
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
