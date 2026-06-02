"""Dump original mjlab/MuJoCo G1 joint/body order from the upstream SMP environment."""

from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec


def main() -> None:
    model = get_spec().compile()

    nonfree_joint_names: list[str] = []
    qpos_adrs: list[int] = []
    dof_adrs: list[int] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        nonfree_joint_names.append(name)
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        dof_adrs.append(int(model.jnt_dofadr[joint_id]))

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(model.nbody)
    ]

    print("[ORIG] nonfree_joint_names=", nonfree_joint_names)
    print("[ORIG] qpos_adrs=", qpos_adrs)
    print("[ORIG] dof_adrs=", dof_adrs)
    print("[ORIG] nonfree_joint_count=", len(nonfree_joint_names))
    print("[ORIG] body_names=", body_names)


if __name__ == "__main__":
    main()
