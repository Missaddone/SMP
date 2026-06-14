"""Finite-step smoke test for SMP IsaacLab tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a short SMP task smoke test.")
parser.add_argument("--task", default="Smp-G1-Steering-modified-v0", help="Gym task id to test.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of vectorized environments.")
parser.add_argument("--steps", type=int, default=8, help="Number of random-action steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "SMP"))
import SMP.tasks  # noqa: F401, E402


def main() -> None:
    """Create an env, reset it, step it a few times, then close cleanly."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[SMP SMOKE] task={args_cli.task}")
    print(f"[SMP SMOKE] observation_space={env.observation_space}")
    print(f"[SMP SMOKE] action_space={env.action_space}")

    env.reset()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            _, _, terminated, truncated, _ = env.step(actions)
            done_count = int((terminated | truncated).sum().item())
            print(f"[SMP SMOKE] step={step + 1}/{args_cli.steps}, done_count={done_count}")

    env.close()
    simulation_app.close()
    print("[SMP SMOKE] ok")


if __name__ == "__main__":
    main()
