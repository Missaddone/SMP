"""Convert retargeted G1 CSV motions to SMP windowed NPZ files.

Input CSV rows are expected to be:
  root_pos(3), root_quat(4), joint_pos(29)

The output matches the SMP G1 prior feature layout:
  [root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3), root_ang_vel(3)]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert G1 retargeting CSV files to SMP NPZ windows.")
    parser.add_argument("--input-dir", default="datasets/g1", help="Directory containing input CSV motion files.")
    parser.add_argument("--output-dir", default="datasets/g1_npz", help="Directory to write NPZ window files.")
    parser.add_argument("--window-size", type=int, default=10, help="Frames per SMP window.")
    parser.add_argument("--stride", type=int, default=1, help="Stride between consecutive windows.")
    parser.add_argument("--input-fps", type=float, default=30.0, help="Frame rate of the input CSV files.")
    parser.add_argument("--output-fps", type=float, default=50.0, help="Frame rate of the output feature sequence.")
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion order in the CSV root quaternion columns.",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for parallel conversion.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for parallel conversion.")
    parser.add_argument(
        "--graceful-close",
        action="store_true",
        help="Try SimulationApp.close() at exit. Useful for debugging Kit shutdown, but it may hang.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "SMP_catchball"))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from SMP_catchball.robots.g1 import G1_CYLINDER_CFG  # noqa: E402
from SMP_catchball.smp.feature_to_state import EE_BODY_NAMES, G1_JOINT_NAMES, NUM_EE, NUM_JOINTS  # noqa: E402
from SMP_catchball.smp.utils import quat_conjugate, quat_mul  # noqa: E402
from SMP_catchball.smp.utils import MotionFeatureBuffer  # noqa: E402


JOINT_NAMES: tuple[str, ...] = G1_JOINT_NAMES


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


def _load_csv(path: Path, quat_order: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except ValueError:
        # EN: Some retargeting exports include a CSV header such as
        # ``root_PosX,root_PosY,...``. Fall back to genfromtxt with names so the
        # numeric rows are parsed while the header is skipped.
        # 中文：部分动捕/重定向导出的 CSV 带表头，例如 ``root_PosX``。这里回退到
        # genfromtxt(names=True)，跳过表头并读取数值数据。
        named = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
        if named.dtype.names is None:
            raise
        arr = np.column_stack([named[name] for name in named.dtype.names]).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[1] != 7 + NUM_JOINTS:
        raise ValueError(f"{path.name}: expected {7 + NUM_JOINTS} columns, got {arr.shape[1]}")

    root_pos = torch.from_numpy(arr[:, 0:3])
    root_quat = torch.from_numpy(arr[:, 3:7])
    if quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    root_quat = torch.nn.functional.normalize(root_quat, dim=-1)
    joint_pos = torch.from_numpy(arr[:, 7:])
    return root_pos, root_quat, joint_pos


def _slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    linear = dot.abs() > 0.9995
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * alpha[:, None]
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    out = torch.where(linear, (1.0 - alpha[:, None]) * q0 + alpha[:, None] * q1, s0 * q0 + s1 * q1)
    return torch.nn.functional.normalize(out, dim=-1)


def _resample_motion(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    input_fps: float,
    output_fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if abs(input_fps - output_fps) < 1e-6:
        return root_pos, root_quat, joint_pos

    num_frames = root_pos.shape[0]
    duration = (num_frames - 1) / input_fps
    # EN: Match mjlab MotionLoader exactly: do not include the final timestamp.
    # 中文：和原工程 mjlab MotionLoader 对齐：重采样不包含最后一个时间端点。
    times = torch.arange(0.0, duration, 1.0 / output_fps, dtype=torch.float32)
    src = (times * input_fps).clamp(max=num_frames - 1)
    idx0 = torch.floor(src).long()
    idx1 = torch.clamp(idx0 + 1, max=num_frames - 1)
    alpha = src - idx0.float()

    root_pos_out = (1.0 - alpha[:, None]) * root_pos[idx0] + alpha[:, None] * root_pos[idx1]
    joint_pos_out = (1.0 - alpha[:, None]) * joint_pos[idx0] + alpha[:, None] * joint_pos[idx1]
    root_quat_out = _slerp(root_quat[idx0], root_quat[idx1], alpha)
    return root_pos_out, root_quat_out, joint_pos_out


def _quat_diff(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    return quat_mul(q1, quat_conjugate(q0))


def _quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., :1].clamp(min=1e-8))
    axis = xyz / sin_half.clamp(min=1e-8)
    return axis * angle


def _finite_difference_velocities(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if root_pos.shape[0] <= 1:
        root_lin_vel = torch.zeros_like(root_pos)
        root_ang_vel = torch.zeros_like(root_pos)
        joint_vel = torch.zeros_like(joint_pos)
        return root_lin_vel, root_ang_vel, joint_vel

    # EN: Match mjlab MotionLoader: torch.gradient for linear/dof velocities.
    # 中文：和原工程 MotionLoader 一致：线速度和关节速度使用 torch.gradient。
    root_lin_vel = torch.gradient(root_pos, spacing=dt, dim=0)[0]
    joint_vel = torch.gradient(joint_pos, spacing=dt, dim=0)[0]

    if root_pos.shape[0] <= 2:
        dq = _quat_diff(root_quat[:-1], root_quat[1:])
        root_ang_vel = torch.zeros_like(root_pos)
        root_ang_vel[:-1] = _quat_to_rotvec(dq) / dt
        root_ang_vel[-1] = root_ang_vel[-2]
        return root_lin_vel, root_ang_vel, joint_vel

    # EN: Match mjlab MotionLoader SO(3) derivative: q[t+1] * conj(q[t-1]) / (2dt),
    # then repeat the first/last interior samples at the boundaries.
    # 中文：和原工程 SO(3) 角速度计算一致：用前后两帧四元数做中心差分，
    # 边界处重复首尾内部样本。
    q_rel = quat_mul(root_quat[2:], quat_conjugate(root_quat[:-2]))
    omega_mid = _quat_to_rotvec(q_rel) / (2.0 * dt)
    root_ang_vel = torch.cat([omega_mid[:1], omega_mid, omega_mid[-1:]], dim=0)
    return root_lin_vel, root_ang_vel, joint_vel


def _data_attr(data, *names: str) -> torch.Tensor:
    for name in names:
        if hasattr(data, name):
            return getattr(data, name)
    raise AttributeError(f"None of these articulation data fields exist: {names}")


def _setup_scene(device: str, dt: float) -> tuple[sim_utils.SimulationContext, InteractiveScene]:
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = G1SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(dt)
    return sim, scene


def _shutdown_scene(sim: sim_utils.SimulationContext, scene: InteractiveScene, graceful_close: bool) -> None:
    """Release IsaacLab objects before closing Kit.

    EN:
    Conversion scripts do not have an env wrapper, so we mirror the important
    parts of IsaacLab env cleanup. On some Isaac Sim/Kit versions even
    ``SimulationApp.close(skip_cleanup=True)`` can hang, so the default path
    exits the one-shot conversion process directly after all files are flushed.
    Pass ``--graceful-close`` only when debugging Kit shutdown itself.

    中文：
    这个转换脚本没有 IsaacLab env wrapper 帮我们释放资源，所以这里手动执行
    env.close() 中最关键的清理步骤。某些 Isaac Sim/Kit 版本里，即使用
    ``SimulationApp.close(skip_cleanup=True)`` 也可能卡住；因此默认路径会在
    文件全部写盘后直接以成功状态退出一次性转换进程。只有排查 Kit 关闭问题时
    才传入 ``--graceful-close``。
    """
    print("Conversion complete. Releasing IsaacLab scene...", flush=True)
    try:
        # EN: Drop scene references before clearing the SimulationContext.
        # 中文：先释放 scene 引用，再清理 SimulationContext。
        del scene
    except Exception:
        pass
    try:
        # EN: Stop timeline/physics before callbacks and singleton cleanup.
        # 中文：先停止 timeline/physics，再清理回调和单例。
        sim.stop()
    except Exception:
        pass
    try:
        sim.clear_all_callbacks()
    except Exception:
        pass
    try:
        sim.clear_instance()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if graceful_close:
        # EN: No replicator work is used here; skip full Kit cleanup when requested.
        # 中文：本脚本不使用 replicator；显式请求时尝试快速关闭 Kit。
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        return

    print("IsaacLab scene released. Exiting without Kit graceful shutdown.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@torch.no_grad()
def _replay_motion_kinematics(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_ids: torch.Tensor,
    ee_ids: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = scene["robot"]
    device = robot.device
    root_pos = root_pos.to(device)
    root_quat = root_quat.to(device)
    root_lin_vel = root_lin_vel.to(device)
    root_ang_vel = root_ang_vel.to(device)
    joint_pos = joint_pos.to(device)
    joint_vel = joint_vel.to(device)

    root_pos_out = []
    root_quat_out = []
    root_lin_vel_out = []
    root_ang_vel_out = []
    ee_pos_out = []
    joint_pos_out = []
    joint_vel_out = []
    origin = scene.env_origins[0].detach().cpu()
    for frame in range(root_pos.shape[0]):
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = root_pos[frame]
        root_state[:, 3:7] = root_quat[frame]
        root_state[:, 7:10] = root_lin_vel[frame]
        root_state[:, 10:13] = root_ang_vel[frame]
        robot.write_root_state_to_sim(root_state)

        joint_pos_full = robot.data.default_joint_pos.clone()
        joint_vel_full = robot.data.default_joint_vel.clone()
        joint_pos_full[:, joint_ids] = joint_pos[frame]
        joint_vel_full[:, joint_ids] = joint_vel[frame]
        robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

        sim.forward()
        scene.update(dt)
        data = robot.data
        root_pos_out.append(_data_attr(data, "root_link_pos_w", "root_pos_w")[0].detach().cpu() - origin)
        root_quat_out.append(_data_attr(data, "root_link_quat_w", "root_quat_w")[0].detach().cpu())
        root_lin_vel_out.append(_data_attr(data, "root_link_lin_vel_w", "root_lin_vel_w")[0].detach().cpu())
        root_ang_vel_out.append(_data_attr(data, "root_link_ang_vel_w", "root_ang_vel_w")[0].detach().cpu())
        body_pos = _data_attr(robot.data, "body_link_pos_w", "body_pos_w")
        ee_pos_out.append(body_pos[0, ee_ids].detach().cpu() - origin)
        joint_pos_runtime = data.joint_pos[:, joint_ids]
        joint_vel_runtime = data.joint_vel[:, joint_ids]
        joint_pos_out.append(joint_pos_runtime[0].detach().cpu())
        joint_vel_out.append(joint_vel_runtime[0].detach().cpu())

    return (
        torch.stack(root_pos_out, dim=0),
        torch.stack(root_quat_out, dim=0),
        torch.stack(root_lin_vel_out, dim=0),
        torch.stack(root_ang_vel_out, dim=0),
        torch.stack(ee_pos_out, dim=0),
        torch.stack(joint_pos_out, dim=0),
        torch.stack(joint_vel_out, dim=0),
    )

def _compute_windows(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    ee_pos: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    window_size: int,
    stride: int,
) -> torch.Tensor | None:
    num_frames = root_pos.shape[0]
    if num_frames < window_size:
        return None

    starts = torch.arange(0, num_frames - window_size + 1, stride, dtype=torch.long)
    offsets = torch.arange(window_size, dtype=torch.long)
    win_idx = starts[:, None] + offsets[None, :]
    num_windows = win_idx.shape[0]
    flat_idx = win_idx.reshape(-1)

    win_root_pos = root_pos.index_select(0, flat_idx).reshape(num_windows, window_size, 3)
    win_root_quat = root_quat.index_select(0, flat_idx).reshape(num_windows, window_size, 4)
    win_root_lin_vel = root_lin_vel.index_select(0, flat_idx).reshape(num_windows, window_size, 3)
    win_root_ang_vel = root_ang_vel.index_select(0, flat_idx).reshape(num_windows, window_size, 3)
    win_ee_pos = ee_pos.index_select(0, flat_idx).reshape(num_windows, window_size, NUM_EE, 3)
    win_joint_pos = joint_pos.index_select(0, flat_idx).reshape(num_windows, window_size, NUM_JOINTS)
    win_joint_vel = joint_vel.index_select(0, flat_idx).reshape(num_windows, window_size, NUM_JOINTS)

    # EN: Use the same feature builder as online SMP reward, so offline
    # pretraining data and runtime reward features share the exact layout.
    # 中文：离线预训练数据直接调用在线 SMP reward 同一个特征构造器，
    # 保证 feature 布局和坐标变换完全一致。
    buffer = MotionFeatureBuffer(
        num_envs=num_windows,
        window_size=window_size,
        num_joints=NUM_JOINTS,
        num_ee=NUM_EE,
        device=win_root_pos.device,
    )
    buffer.reset(
        torch.arange(num_windows, dtype=torch.long, device=win_root_pos.device),
        win_root_pos,
        win_root_quat,
        win_root_lin_vel,
        win_root_ang_vel,
        win_ee_pos,
        win_joint_pos,
        win_joint_vel,
    )
    return buffer.compute_features()


def main() -> None:
    input_dir = Path(args_cli.input_dir)
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    if args_cli.num_shards > 1:
        csv_files = csv_files[args_cli.shard_index :: args_cli.num_shards]

    dt = 1.0 / args_cli.output_fps
    sim, scene = _setup_scene(args_cli.device, dt)
    robot = scene["robot"]
    joint_ids = torch.tensor(robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0], dtype=torch.long, device=robot.device)
    ee_ids = torch.tensor(robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0], dtype=torch.long, device=robot.device)

    print(f"Input: {input_dir} ({len(csv_files)} files)")
    print(f"Output: {output_dir}")
    print(f"FPS: {args_cli.input_fps} -> {args_cli.output_fps}; window={args_cli.window_size}; stride={args_cli.stride}")
    print(f"Joints: {len(joint_ids)} | End-effectors: {tuple(EE_BODY_NAMES)}")
    print(f"Joint order: CSV/NPZ columns are interpreted as {JOINT_NAMES}")
    print(f"Joint mapping: IsaacLab runtime ids for that order are {joint_ids.detach().cpu().tolist()}")

    feature_dims = np.array([3, 6, NUM_JOINTS, NUM_EE * 3, 3, 3], dtype=np.int32)
    for index, csv_path in enumerate(csv_files):
        print(f"[{index + 1}/{len(csv_files)}] {csv_path.name}")
        root_pos, root_quat, joint_pos = _load_csv(csv_path, args_cli.quat_order)
        root_pos, root_quat, joint_pos = _resample_motion(
            root_pos, root_quat, joint_pos, args_cli.input_fps, args_cli.output_fps
        )
        root_lin_vel, root_ang_vel, joint_vel = _finite_difference_velocities(root_pos, root_quat, joint_pos, dt)
        (
            root_pos_sim,
            root_quat_sim,
            root_lin_vel_sim,
            root_ang_vel_sim,
            ee_pos,
            joint_pos_sim,
            joint_vel_sim,
        ) = _replay_motion_kinematics(
            sim,
            scene,
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
            joint_ids,
            ee_ids,
            dt,
        )
        windows = _compute_windows(
            root_pos_sim,
            root_quat_sim,
            root_lin_vel_sim,
            root_ang_vel_sim,
            ee_pos,
            joint_pos_sim,
            joint_vel_sim,
            args_cli.window_size,
            args_cli.stride,
        )
        if windows is None:
            print(f"  [skip] too short for window_size={args_cli.window_size}")
            continue

        out_path = output_dir / f"{csv_path.stem}.npz"
        np.savez_compressed(
            out_path,
            windows=windows.numpy().astype(np.float32),
            fps=np.array([args_cli.output_fps], dtype=np.float32),
            window_size=np.array([args_cli.window_size], dtype=np.int32),
            stride=np.array([args_cli.stride], dtype=np.int32),
            ee_body_names=np.array(EE_BODY_NAMES),
            joint_names=np.array(JOINT_NAMES),
            feature_dims=feature_dims,
        )
        print(f"  saved {out_path.name}: windows={tuple(windows.shape)}")

    _shutdown_scene(sim, scene, args_cli.graceful_close)


if __name__ == "__main__":
    main()
