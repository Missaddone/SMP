"""Debug SMP runtime wiring for a manager-based IsaacLab environment.

EN: This script does not train. It only creates the env, resets it, runs a few
random steps, and prints SMP prior / GSI / guidance-reward diagnostics.

中文：这个脚本不会训练。它只创建环境、reset、随机 step 几步，并打印 SMP
预训练模型、GSI 和 guidance/style reward 的诊断信息。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug SMP prior, GSI, and guidance reward runtime state.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Smp-G1-Forward-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=8, help="Number of random steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from SMP_catchball.smp.feature_to_state import G1_JOINT_NAMES
import SMP_catchball.tasks  # noqa: F401


def _shape(value) -> tuple[int, ...] | str:
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def _root_z(base_env) -> torch.Tensor:
    robot = base_env.scene["robot"]
    data = robot.data
    root_pos = data.root_link_pos_w if hasattr(data, "root_link_pos_w") else data.root_pos_w
    return root_pos[:, 2]


def _termination_counts(base_env) -> str:
    terms = getattr(base_env.termination_manager, "_term_names", [])
    chunks = []
    for name in terms:
        try:
            value = base_env.termination_manager.get_term(name)
            chunks.append(f"{name}={int(value.sum().detach().cpu())}")
        except Exception:
            pass
    return " ".join(chunks)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    print("[SMP-DEBUG] env created")
    print(f"[SMP-DEBUG] observation_space={env.observation_space}")
    print(f"[SMP-DEBUG] action_space={env.action_space}")
    robot = base_env.scene["robot"]
    runtime_joint_names = list(robot.data.joint_names)
    print(f"[SMP-DEBUG] runtime_joint_names={runtime_joint_names}")
    print(f"[SMP-DEBUG] smp_joint_names={list(G1_JOINT_NAMES)}")
    print(f"[SMP-DEBUG] runtime_matches_smp_order={runtime_joint_names == list(G1_JOINT_NAMES)}")

    # EN: These attributes are created by init_smp_state during startup.
    # 中文：这些属性由 startup 阶段的 init_smp_state 创建。
    model, scheduler, q_low, q_high, feature_dim, window_size = base_env._smp_bundle
    print(
        "[SMP-DEBUG] prior_loaded=True "
        f"feature_dim={feature_dim} window_size={window_size} timesteps={scheduler.num_timesteps} "
        f"q_low={_shape(q_low)} q_high={_shape(q_high)} device={next(model.parameters()).device}"
    )
    print(
        "[SMP-DEBUG] gsi_pool="
        f"{_shape(base_env._smp_gsi_pool)} ee_indexes={base_env._smp_ee_indexes.detach().cpu().tolist()}"
    )
    print(f"[SMP-DEBUG] buffer={_shape(base_env._smp_buffer.root_pos_w)}")

    env.reset()
    print("[SMP-DEBUG] reset_ok=True")
    root_z = _root_z(base_env)
    print(
        "[SMP-DEBUG] reset_root_z "
        f"min={float(root_z.min().detach().cpu()):.4f} "
        f"mean={float(root_z.mean().detach().cpu()):.4f} "
        f"max={float(root_z.max().detach().cpu()):.4f}"
    )

    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=base_env.device) - 1
            _, reward, terminated, truncated, _ = env.step(actions)
        raw_err = getattr(base_env, "_smp_raw_err", None)
        raw_err_mean = float(raw_err.mean().detach().cpu()) if raw_err is not None else float("nan")
        root_z = _root_z(base_env)
        print(
            f"[SMP-DEBUG] step={step + 1} "
            f"reward_mean={float(reward.mean().detach().cpu()):.6f} "
            f"smp_raw_err_mean={raw_err_mean:.6f} "
            f"root_z_min={float(root_z.min().detach().cpu()):.4f} "
            f"root_z_mean={float(root_z.mean().detach().cpu()):.4f} "
            f"terminated={int(terminated.sum().detach().cpu())} truncated={int(truncated.sum().detach().cpu())}"
            f" terms=[{_termination_counts(base_env)}]"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
