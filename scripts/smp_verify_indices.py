"""Verify runtime joint/body index order for the migrated SMP task.

EN:
  This script launches the IsaacLab environment and prints the actual runtime
  joint/body order used by the articulation, action term, SMP GSI writer, and
  SMP reward buffer. It is intended to catch hidden index-order mismatches
  between the original mjlab/MuJoCo G1 and the migrated IsaacLab/PhysX G1.

中文：
  这个脚本会启动 IsaacLab 环境，并打印 articulation、action term、SMP GSI
  写入和 SMP reward buffer 实际使用的 joint/body 顺序。它用于检查从
  mjlab/MuJoCo 迁移到 IsaacLab/PhysX 后是否存在隐藏的 index 顺序错位。
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Verify SMP runtime joint/body index order.")
parser.add_argument("--task", type=str, default="Smp-G1-Forward-v0", help="IsaacLab task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to create.")
parser.add_argument(
    "--original-smp-root",
    type=Path,
    default=Path.home() / "smp",
    help="Path to the original SMP repository used to dump mjlab/MuJoCo runtime order.",
)
parser.add_argument(
    "--skip-original",
    action="store_true",
    help="Skip dumping the original mjlab/MuJoCo order and only check IsaacLab against the canonical order.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from SMP_catchball.smp.feature_to_state import EE_BODY_NAMES, G1_JOINT_NAMES
import SMP_catchball.tasks  # noqa: F401


def _dump_original_order(original_root: Path) -> tuple[list[str] | None, list[str] | None]:
    """Run the original SMP repository and return its MuJoCo-compiled order."""
    dump_script = Path(__file__).resolve().parent / "smp_dump_original_order.py"
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    cmd = ["uv", "run", "python", str(dump_script)]
    try:
        result = subprocess.run(
            cmd,
            cwd=original_root,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        print(f"[VERIFY] original_dump_failed={exc}")
        return None, None

    joint_names = None
    body_names = None
    print("\n[VERIFY] original SMP dump output:")
    for line in result.stdout.splitlines():
        print(line)
        if line.startswith("[ORIG] nonfree_joint_names="):
            joint_names = ast.literal_eval(line.split("=", 1)[1].strip())
        elif line.startswith("[ORIG] body_names="):
            body_names = ast.literal_eval(line.split("=", 1)[1].strip())
    return joint_names, body_names


def _names_from_ids(names: list[str], ids) -> list[str]:
    if isinstance(ids, slice):
        return names[ids]
    return [names[int(i)] for i in ids]


def _print_mapping(title: str, canonical: tuple[str, ...], runtime_names: list[str], runtime_ids) -> bool:
    resolved = _names_from_ids(runtime_names, runtime_ids)
    ok = list(canonical) == resolved
    print(f"\n[VERIFY] {title}")
    print(f"[VERIFY]   ids={list(map(int, runtime_ids)) if not isinstance(runtime_ids, slice) else runtime_ids}")
    print(f"[VERIFY]   names={resolved}")
    print(f"[VERIFY]   matches_canonical={ok}")
    if not ok:
        print(f"[VERIFY]   canonical={list(canonical)}")
    return ok


def main() -> None:
    orig_joint_names = None
    orig_body_names = None
    if not args_cli.skip_original:
        orig_joint_names, orig_body_names = _dump_original_order(args_cli.original_smp_root)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    print("[VERIFY] env_created=True")
    print(f"[VERIFY] task={args_cli.task}")
    print(f"[VERIFY] device={base_env.device}")
    print(f"[VERIFY] action_space={env.action_space}")
    print(f"[VERIFY] observation_space={env.observation_space}")

    runtime_joint_names = list(robot.data.joint_names)
    runtime_body_names = list(robot.data.body_names)
    print(f"\n[VERIFY] runtime_joint_names={runtime_joint_names}")
    print(f"[VERIFY] canonical_smp_joint_names={list(G1_JOINT_NAMES)}")
    print(f"[VERIFY] runtime_joint_order_matches_canonical={runtime_joint_names == list(G1_JOINT_NAMES)}")
    if orig_joint_names is not None:
        print(f"[VERIFY] original_joint_order_matches_canonical={orig_joint_names == list(G1_JOINT_NAMES)}")
        print(f"[VERIFY] runtime_joint_order_matches_original={runtime_joint_names == orig_joint_names}")
    if orig_body_names is not None:
        print(f"[VERIFY] original_body_names={orig_body_names}")

    joint_ids, joint_names = robot.find_joints(list(G1_JOINT_NAMES), preserve_order=True)
    body_ids, body_names = robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)

    joints_ok = _print_mapping("SMP joint mapping", G1_JOINT_NAMES, runtime_joint_names, joint_ids)
    bodies_ok = _print_mapping("SMP end-effector body mapping", EE_BODY_NAMES, runtime_body_names, body_ids)

    action_ok = True
    action_term = getattr(base_env.action_manager, "_terms", {}).get("joint_pos", None)
    if action_term is not None:
        action_joint_names = list(getattr(action_term, "_joint_names", []))
        action_joint_ids = getattr(action_term, "_joint_ids", None)
        print("\n[VERIFY] action term joint_pos")
        print(f"[VERIFY]   _joint_ids={action_joint_ids}")
        print(f"[VERIFY]   _joint_names={action_joint_names}")
        action_ok = action_joint_names == list(G1_JOINT_NAMES)
        print(f"[VERIFY]   matches_canonical={action_ok}")
    else:
        print("\n[VERIFY] action term joint_pos not found")
        action_ok = False

    smp_joint_indexes = getattr(base_env, "_smp_joint_indexes", None)
    if smp_joint_indexes is not None:
        smp_joint_indexes_cpu = smp_joint_indexes.detach().cpu().tolist()
        print("\n[VERIFY] env._smp_joint_indexes")
        print(f"[VERIFY]   ids={smp_joint_indexes_cpu}")
        print(f"[VERIFY]   names={_names_from_ids(runtime_joint_names, smp_joint_indexes_cpu)}")
    else:
        print("\n[VERIFY] env._smp_joint_indexes missing")
        joints_ok = False

    # Reset once to ensure GSI can write using these ids without shape/order errors.
    env.reset()
    with torch.inference_mode():
        zero_action = torch.zeros(env.action_space.shape, device=base_env.device)
        _, reward, terminated, truncated, _ = env.step(zero_action)
    print("\n[VERIFY] one_step_ok=True")
    print(f"[VERIFY] reward_mean={float(reward.mean().detach().cpu()):.6f}")
    print(f"[VERIFY] terminated={int(terminated.sum().detach().cpu())} truncated={int(truncated.sum().detach().cpu())}")

    all_ok = joints_ok and bodies_ok and action_ok
    print(f"\n[VERIFY] overall_index_check_passed={all_ok}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
