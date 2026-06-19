# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a student policy by simple dual-teacher behavior cloning."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train a dual-teacher distilled policy without forked RSL-RL.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of distillation iterations.")
parser.add_argument("--teacher_0_checkpoint_path", type=str, default=None, help="Checkpoint for low-speed teacher.")
parser.add_argument("--teacher_1_checkpoint_path", type=str, default=None, help="Checkpoint for high-speed teacher.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "SMP"))
import SMP.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _runner_cfg_for_obs(agent_cfg: RslRlBaseRunnerCfg, obs_group: str) -> dict:
    cfg = copy.deepcopy(agent_cfg.to_dict())
    cfg["class_name"] = "OnPolicyRunner"
    cfg["obs_groups"] = {"actor": [obs_group], "critic": [obs_group]}
    return cfg


def _get_policy_module(runner: OnPolicyRunner) -> torch.nn.Module:
    alg = getattr(runner, "alg", None)
    for attr_name in ("actor_critic", "policy", "actor"):
        module = getattr(alg, attr_name, None)
        if module is not None:
            return module
    raise RuntimeError("Could not find policy module on RSL-RL runner.")


def _call_policy(policy: torch.nn.Module, obs) -> torch.Tensor:
    if hasattr(policy, "act_inference"):
        return policy.act_inference(obs)
    if hasattr(policy, "act"):
        return policy.act(obs)
    return policy(obs)


def _load_actor_only(policy: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    checkpoint_path = os.path.expanduser(checkpoint_path)
    loaded = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = loaded.get("model_state_dict", loaded)

    actor_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("actor.") or key.startswith("actor_obs_normalizer."):
            actor_state_dict[key] = value
        elif key in ("std", "log_std", "std_param", "log_std_param"):
            actor_state_dict[key] = value

    if not actor_state_dict:
        raise ValueError(f"Checkpoint does not contain actor parameters: {checkpoint_path}")

    missing_keys, unexpected_keys = policy.load_state_dict(actor_state_dict, strict=False)
    actor_missing = [
        key
        for key in missing_keys
        if key.startswith("actor.") or key.startswith("actor_obs_normalizer.") or key in actor_state_dict
    ]
    actor_unexpected = [
        key
        for key in unexpected_keys
        if key.startswith("actor.") or key.startswith("actor_obs_normalizer.") or key in actor_state_dict
    ]
    if actor_missing or actor_unexpected:
        print(
            "[WARN] Teacher actor checkpoint loaded with non-strict differences: "
            f"missing={actor_missing}, unexpected={actor_unexpected}"
        )


def _teacher_1_mask_from_selector(selector_obs: torch.Tensor, agent_cfg: RslRlBaseRunnerCfg) -> torch.Tensor:
    selector_mode = getattr(agent_cfg, "selector_mode", "body_velocity_speed")
    if selector_mode == "body_velocity_speed":
        speed = torch.linalg.norm(selector_obs[:, 0:2], dim=-1)
        return speed > agent_cfg.selector_speed_threshold
    if selector_mode == "index_threshold":
        return selector_obs[:, agent_cfg.selector_obs_index] > agent_cfg.selector_threshold
    raise ValueError(f"Unsupported selector_mode: {selector_mode}")


def _timeout_tensor_from_extras(extras: dict, dones: torch.Tensor) -> torch.Tensor:
    """Best-effort timeout extraction across Isaac Lab/RSL-RL wrapper versions."""
    for key in ("time_outs", "timeouts", "time_out", "timeout"):
        value = extras.get(key)
        if value is not None:
            return value.to(device=dones.device, dtype=torch.bool).view_as(dones)
    return torch.zeros_like(dones, dtype=torch.bool)


def _save_student_checkpoint(
    path: str,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> None:
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": iteration,
        },
        path,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "3.0.1")

    if args_cli.teacher_0_checkpoint_path is not None:
        agent_cfg.teacher_0_checkpoint_path = args_cli.teacher_0_checkpoint_path
    if args_cli.teacher_1_checkpoint_path is not None:
        agent_cfg.teacher_1_checkpoint_path = args_cli.teacher_1_checkpoint_path
    if not agent_cfg.teacher_0_checkpoint_path or not agent_cfg.teacher_1_checkpoint_path:
        raise ValueError("Both teacher_0_checkpoint_path and teacher_1_checkpoint_path are required.")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "dual_distill", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] Logging dual distillation in directory: {log_dir}")

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    student_runner = OnPolicyRunner(env, _runner_cfg_for_obs(agent_cfg, "policy"), log_dir=None, device=agent_cfg.device)
    teacher_0_runner = OnPolicyRunner(
        env, _runner_cfg_for_obs(agent_cfg, "teacher_0"), log_dir=None, device=agent_cfg.device
    )
    teacher_1_runner = OnPolicyRunner(
        env, _runner_cfg_for_obs(agent_cfg, "teacher_1"), log_dir=None, device=agent_cfg.device
    )

    student = _get_policy_module(student_runner)
    teacher_0 = _get_policy_module(teacher_0_runner)
    teacher_1 = _get_policy_module(teacher_1_runner)
    _load_actor_only(teacher_0, agent_cfg.teacher_0_checkpoint_path, agent_cfg.device)
    _load_actor_only(teacher_1, agent_cfg.teacher_1_checkpoint_path, agent_cfg.device)
    teacher_0.eval()
    teacher_1.eval()
    for parameter in teacher_0.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher_1.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=agent_cfg.algorithm.learning_rate,
    )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    obs = env.get_observations()
    episode_lengths = torch.zeros(env.num_envs, device=agent_cfg.device)
    start_time = time.time()
    last_loss = 0.0
    last_teacher_1_frac = 0.0

    try:
        for iteration in range(agent_cfg.max_iterations):
            mean_loss = 0.0
            mean_teacher_1_frac = 0.0
            mean_done_frac = 0.0
            mean_timeout_frac = 0.0
            mean_command_speed = 0.0
            finished_lengths = []
            for _ in range(agent_cfg.num_steps_per_env):
                student.train()
                update_normalization = getattr(student, "update_normalization", None)
                if update_normalization is not None:
                    update_normalization(obs)

                student_actions = _call_policy(student, obs)
                with torch.no_grad():
                    teacher_0_actions = _call_policy(teacher_0, obs)
                    teacher_1_actions = _call_policy(teacher_1, obs)

                selector_obs = obs[agent_cfg.selector_obs_group]
                command_speed = torch.linalg.norm(selector_obs[:, 0:2], dim=-1)
                teacher_1_mask = _teacher_1_mask_from_selector(selector_obs, agent_cfg)
                target_actions = torch.where(teacher_1_mask.unsqueeze(-1), teacher_1_actions, teacher_0_actions)
                loss = torch.nn.functional.mse_loss(student_actions, target_actions)

                optimizer.zero_grad()
                loss.backward()
                if agent_cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), agent_cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    obs, _, dones, extras = env.step(student_actions.detach())

                dones = dones.to(device=agent_cfg.device, dtype=torch.bool)
                timeouts = _timeout_tensor_from_extras(extras, dones)
                episode_lengths += 1
                if dones.any():
                    finished_lengths.append(episode_lengths[dones].detach())
                    episode_lengths[dones] = 0

                mean_loss += loss.item()
                mean_teacher_1_frac += teacher_1_mask.float().mean().item()
                mean_done_frac += dones.float().mean().item()
                mean_timeout_frac += timeouts.float().mean().item()
                mean_command_speed += command_speed.mean().item()

            last_loss = mean_loss / agent_cfg.num_steps_per_env
            last_teacher_1_frac = mean_teacher_1_frac / agent_cfg.num_steps_per_env
            done_frac = mean_done_frac / agent_cfg.num_steps_per_env
            timeout_frac = mean_timeout_frac / agent_cfg.num_steps_per_env
            command_speed = mean_command_speed / agent_cfg.num_steps_per_env
            if finished_lengths:
                mean_episode_length = torch.cat(finished_lengths).float().mean().item()
            else:
                mean_episode_length = episode_lengths.mean().item()

            writer.add_scalar("distill/loss", last_loss, iteration)
            writer.add_scalar("distill/teacher_1_frac", last_teacher_1_frac, iteration)
            writer.add_scalar("env/done_frac", done_frac, iteration)
            writer.add_scalar("env/timeout_frac", timeout_frac, iteration)
            writer.add_scalar("env/mean_episode_length", mean_episode_length, iteration)
            writer.add_scalar("command/mean_speed", command_speed, iteration)
            writer.add_scalar("train/learning_rate", agent_cfg.algorithm.learning_rate, iteration)

            if iteration % 10 == 0:
                print(
                    f"[INFO] Iteration {iteration:05d} | loss={last_loss:.6f} "
                    f"| teacher_1_frac={last_teacher_1_frac:.3f} "
                    f"| done_frac={done_frac:.3f} | ep_len={mean_episode_length:.1f}"
                )
            if iteration % agent_cfg.save_interval == 0 or iteration == agent_cfg.max_iterations - 1:
                checkpoint_path = os.path.join(log_dir, "checkpoints", f"model_{iteration:05d}.pt")
                _save_student_checkpoint(checkpoint_path, student, optimizer, iteration)

        final_path = os.path.join(log_dir, "model_final.pt")
        _save_student_checkpoint(final_path, student, optimizer, agent_cfg.max_iterations)
        print(
            f"[INFO] Finished dual distillation in {round(time.time() - start_time, 2)}s "
            f"| final_loss={last_loss:.6f} | teacher_1_frac={last_teacher_1_frac:.3f}"
        )
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
