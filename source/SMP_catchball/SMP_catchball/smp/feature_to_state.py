"""Feature layout helpers for the G1 SMP prior."""

from __future__ import annotations

import torch

NUM_JOINTS = 29
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
