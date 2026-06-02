"""Feature layout helpers for the G1 SMP prior."""

from __future__ import annotations

import torch

from SMP_catchball.smp.utils import matrix_from_quat, quat_apply, quat_conjugate, quat_mul, yaw_quat

NUM_JOINTS = 29
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
"""Canonical MJCF/SMP joint order for the 29-D G1 joint feature vector."""

EE_BODY_NAMES: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
NUM_EE = len(EE_BODY_NAMES)


def slice_features(frame: torch.Tensor) -> dict[str, torch.Tensor]:
    """Slice G1 SMP features into named components."""
    expected = 3 + 6 + NUM_JOINTS + NUM_EE * 3 + 3 + 3
    if frame.shape[-1] != expected:
        raise ValueError(f"expected feature_dim={expected}; got {frame.shape[-1]}")

    joint_pos_end = 9 + NUM_JOINTS
    ee_pos_end = joint_pos_end + NUM_EE * 3
    lin_vel_end = ee_pos_end + 3
    ang_vel_end = lin_vel_end + 3
    return {
        "root_pos": frame[..., 0:3],
        "root_rot": frame[..., 3:9],
        "joint_pos": frame[..., 9:joint_pos_end],
        "ee_pos": frame[..., joint_pos_end:ee_pos_end],
        "root_lin_vel": frame[..., ee_pos_end:lin_vel_end],
        "root_ang_vel": frame[..., lin_vel_end:ang_vel_end],
    }


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert SMP 6D tan-norm ``[col0, col2]`` to a rotation matrix."""
    col0 = torch.nn.functional.normalize(d6[..., :3], dim=-1)
    col2 = d6[..., 3:6]
    col2 = col2 - (col0 * col2).sum(dim=-1, keepdim=True) * col0
    col2 = torch.nn.functional.normalize(col2, dim=-1)
    col1 = torch.cross(col2, col0, dim=-1)
    return torch.stack([col0, col1, col2], dim=-1)


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to wxyz quaternions."""
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))
    qx = torch.copysign(qx, matrix[..., 2, 1] - matrix[..., 1, 2])
    qy = torch.copysign(qy, matrix[..., 0, 2] - matrix[..., 2, 0])
    qz = torch.copysign(qz, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.nn.functional.normalize(torch.stack([qw, qx, qy, qz], dim=-1), dim=-1)


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
    """Convert SMP 6D tan-norm to wxyz quaternion."""
    return quat_from_matrix(rot6d_to_matrix(d6))


def tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to SMP 6D tan-norm ``[col0, col2]``."""
    mat = matrix_from_quat(quat)
    return torch.cat([mat[..., :, 0], mat[..., :, 2]], dim=-1)


def heading_inv_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only inverse of a world quaternion."""
    return quat_conjugate(yaw_quat(quat))


def window_to_pelvis_trajectory(
    window: torch.Tensor,
    anchor_pelvis_pos_w: torch.Tensor,
    anchor_pelvis_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct world-frame pelvis pose and joint position from a denormalized window."""
    parts = slice_features(window)
    root_pos_local = parts["root_pos"]
    root_rot_6d = parts["root_rot"]
    window_size = window.shape[0]

    anchor_pelvis_pos_w = anchor_pelvis_pos_w.to(window)
    anchor_pelvis_quat_w = anchor_pelvis_quat_w.to(window)
    yaw_t = yaw_quat(anchor_pelvis_quat_w[None]).squeeze(0)

    local_xy = root_pos_local.clone()
    local_xy[..., 2] = 0.0
    world_offset_xy = quat_apply(yaw_t[None].expand(window_size, 4), local_xy)
    pelvis_pos_w = world_offset_xy + anchor_pelvis_pos_w[None, :3]
    pelvis_pos_w = pelvis_pos_w.clone()
    pelvis_pos_w[..., 2] = root_pos_local[..., 2]

    root_rot_local_quat = rot6d_to_quat(root_rot_6d)
    pelvis_quat_w = quat_mul(yaw_t[None].expand(window_size, 4), root_rot_local_quat)
    return pelvis_pos_w, pelvis_quat_w, parts["joint_pos"]


def window_to_ee_trajectories(
    window: torch.Tensor,
    pelvis_pos_w: torch.Tensor,
    pelvis_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Lift SMP end-effector offsets in a window back to world-frame positions."""
    parts = slice_features(window)
    window_size = window.shape[0]
    ee_pos_local = parts["ee_pos"].reshape(window_size, NUM_EE, 3)
    yaw_t = yaw_quat(pelvis_quat_w[-1:])
    yaw_t_ee = yaw_t.expand(window_size, 4)[:, None, :].expand(window_size, NUM_EE, 4).reshape(-1, 4)
    ee_offset_w = quat_apply(yaw_t_ee, ee_pos_local.reshape(-1, 3)).reshape(window_size, NUM_EE, 3)
    return ee_offset_w + pelvis_pos_w[:, None, :]
