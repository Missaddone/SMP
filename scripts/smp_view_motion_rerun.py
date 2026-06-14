"""Play G1 CSV motions in rerun using the robot URDF meshes.

This script does not launch Isaac Sim. For CSV files it parses the G1 URDF,
computes forward kinematics from root pose + 29 joint angles, and logs the
robot link meshes to rerun over time. NPZ support is kept only as a lightweight
feature-window inspector because SMP windows are not full global trajectories.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "source" / "SMP" / "SMP" / "assets"
DEFAULT_URDF = ASSET_DIR / "unitree_description" / "urdf" / "g1" / "main.urdf"

G1_JOINT_NAMES = (
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


@dataclass
class Visual:
    mesh_path: Path
    origin: np.ndarray
    color: tuple[int, int, int, int] | None = None


@dataclass
class Link:
    name: str
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass
class RobotModel:
    links: dict[str, Link]
    joints: list[Joint]
    children: dict[str, list[Joint]]
    root_link: str


def _import_rerun():
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise SystemExit("rerun-sdk is not installed. Install it with: pip install rerun-sdk") from exc
    return rr


def _load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        names = []
    except ValueError:
        named = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
        if named.dtype.names is None:
            raise
        names = list(named.dtype.names)
        arr = np.column_stack([named[name] for name in names]).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    return arr, names


def _parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    return _rot_z(float(rpy[2])) @ _rot_y(float(rpy[1])) @ _rot_x(float(rpy[0]))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _transform(xyz: np.ndarray, rpy: np.ndarray | None = None, rot: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot if rot is not None else _rpy_to_rot(np.zeros(3) if rpy is None else rpy)
    out[:3, 3] = xyz
    return out


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q.astype(np.float64)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    return _quat_xyzw_to_rot(q[[1, 2, 3, 0]])


def _resolve_package_path(path: str, urdf_path: Path) -> Path:
    if path.startswith("package://unitree_description/"):
        rel = path.removeprefix("package://unitree_description/")
        return ASSET_DIR / "unitree_description" / rel
    if path.startswith("file://"):
        return Path(path.removeprefix("file://"))
    p = Path(path)
    if p.is_absolute():
        return p
    return urdf_path.parent / p


def _origin_from_xml(elem: ET.Element | None) -> np.ndarray:
    if elem is None:
        return np.eye(4, dtype=np.float64)
    xyz = _parse_vec(elem.attrib.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_vec(elem.attrib.get("rpy"), (0.0, 0.0, 0.0))
    return _transform(xyz, rpy)


def _parse_color(link_elem: ET.Element, visual_elem: ET.Element) -> tuple[int, int, int, int] | None:
    material = visual_elem.find("material")
    if material is not None:
        color = material.find("color")
        if color is not None and "rgba" in color.attrib:
            rgba = [float(v) for v in color.attrib["rgba"].split()]
            return tuple(int(np.clip(v, 0.0, 1.0) * 255) for v in rgba)  # type: ignore[return-value]
        material_name = material.attrib.get("name")
        if material_name:
            for mat in link_elem.iterfind("../material"):
                if mat.attrib.get("name") == material_name:
                    color = mat.find("color")
                    if color is not None and "rgba" in color.attrib:
                        rgba = [float(v) for v in color.attrib["rgba"].split()]
                        return tuple(int(np.clip(v, 0.0, 1.0) * 255) for v in rgba)  # type: ignore[return-value]
    return None


def load_urdf(path: Path) -> RobotModel:
    tree = ET.parse(path)
    root = tree.getroot()
    links: dict[str, Link] = {}
    child_links: set[str] = set()

    for link_elem in root.findall("link"):
        name = link_elem.attrib["name"]
        link = Link(name=name)
        for idx, visual_elem in enumerate(link_elem.findall("visual")):
            mesh_elem = visual_elem.find("geometry/mesh")
            if mesh_elem is None or "filename" not in mesh_elem.attrib:
                continue
            mesh_path = _resolve_package_path(mesh_elem.attrib["filename"], path)
            if not mesh_path.exists():
                continue
            link.visuals.append(
                Visual(
                    mesh_path=mesh_path,
                    origin=_origin_from_xml(visual_elem.find("origin")),
                    color=_parse_color(link_elem, visual_elem),
                )
            )
        links[name] = link

    joints: list[Joint] = []
    children: dict[str, list[Joint]] = {}
    for joint_elem in root.findall("joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent = parent_elem.attrib["link"]
        child = child_elem.attrib["link"]
        child_links.add(child)
        axis = _parse_vec(joint_elem.find("axis").attrib.get("xyz") if joint_elem.find("axis") is not None else None, (0.0, 0.0, 1.0))
        joint = Joint(
            name=joint_elem.attrib["name"],
            type=joint_elem.attrib.get("type", "fixed"),
            parent=parent,
            child=child,
            origin=_origin_from_xml(joint_elem.find("origin")),
            axis=axis,
        )
        joints.append(joint)
        children.setdefault(parent, []).append(joint)

    roots = [name for name in links if name not in child_links]
    root_link = roots[0] if roots else "pelvis"
    return RobotModel(links=links, joints=joints, children=children, root_link=root_link)


def compute_fk(model: RobotModel, root_tf: np.ndarray, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {model.root_link: root_tf}
    stack = [model.root_link]
    while stack:
        parent = stack.pop()
        parent_tf = out[parent]
        for joint in model.children.get(parent, []):
            joint_tf = joint.origin.copy()
            if joint.type in {"revolute", "continuous"}:
                angle = float(joint_values.get(joint.name, 0.0))
                joint_tf = joint_tf @ _transform(np.zeros(3), rot=_axis_angle(joint.axis, angle))
            out[joint.child] = parent_tf @ joint_tf
            stack.append(joint.child)
    return out


def _entity_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _rr_transform(rr, tf: np.ndarray):
    return rr.Transform3D(translation=tf[:3, 3].tolist(), mat3x3=tf[:3, :3].tolist())


def log_robot_static(rr, model: RobotModel, log_meshes: bool) -> None:
    if not log_meshes:
        return
    for link in model.links.values():
        for idx, visual in enumerate(link.visuals):
            entity = f"world/robot/{_entity_name(link.name)}/visual_{idx}"
            rr.log(entity, _rr_transform(rr, visual.origin), static=True)
            kwargs = {}
            if visual.color is not None:
                kwargs["albedo_factor"] = visual.color
            try:
                rr.log(entity, rr.Asset3D(path=visual.mesh_path, **kwargs), static=True)
            except Exception as exc:
                print(f"[WARN] failed to log mesh {visual.mesh_path}: {exc}")


def log_robot_frame(rr, model: RobotModel, link_tfs: dict[str, np.ndarray], joint_values: dict[str, float], frame_idx: int, fps: float) -> None:
    rr.set_time("time", duration=frame_idx / fps)
    for link_name, tf in link_tfs.items():
        rr.log(f"world/robot/{_entity_name(link_name)}", _rr_transform(rr, tf))

    strips = []
    points = []
    labels = []
    for joint in model.joints:
        if joint.parent not in link_tfs or joint.child not in link_tfs:
            continue
        p = link_tfs[joint.parent][:3, 3]
        c = link_tfs[joint.child][:3, 3]
        strips.append(np.stack([p, c], axis=0))
        points.append(c)
        labels.append(joint.child)
    if strips:
        rr.log("world/robot_skeleton", rr.LineStrips3D(strips, radii=0.008, colors=[[80, 180, 255, 180]]))
    if points:
        rr.log("world/robot_joints", rr.Points3D(np.asarray(points), radii=0.018, colors=[[255, 180, 80, 220]], labels=labels))

    root_pos = link_tfs[model.root_link][:3, 3]
    rr.log("world/root", rr.Points3D(root_pos[None], radii=0.04, colors=[[255, 60, 60]]))


def view_csv_robot(
    path: Path,
    urdf_path: Path,
    fps: float,
    max_frames: int,
    quat_order: str,
    log_meshes: bool,
) -> None:
    rr = _import_rerun()
    arr, _ = _load_csv(path)
    expected_cols = 7 + len(G1_JOINT_NAMES)
    if arr.shape[1] != expected_cols:
        raise ValueError(f"{path} should have {expected_cols} columns: root(7)+joints(29), got {arr.shape[1]}")

    model = load_urdf(urdf_path)
    root_pos = arr[:, 0:3].astype(np.float64)
    root_quat = arr[:, 3:7].astype(np.float64)
    joint_pos = arr[:, 7:36].astype(np.float64)
    count = min(root_pos.shape[0], max_frames)

    rr.init(f"smp_robot_{path.stem}", spawn=True)
    rr.log("world/root_trajectory", rr.LineStrips3D([root_pos[:count]], colors=[[60, 180, 255]]), static=True)
    log_robot_static(rr, model, log_meshes=log_meshes)

    for i in range(count):
        root_tf = np.eye(4, dtype=np.float64)
        root_tf[:3, 3] = root_pos[i]
        root_tf[:3, :3] = _quat_xyzw_to_rot(root_quat[i]) if quat_order == "xyzw" else _quat_wxyz_to_rot(root_quat[i])
        joint_values = {name: float(joint_pos[i, idx]) for idx, name in enumerate(G1_JOINT_NAMES)}
        link_tfs = compute_fk(model, root_tf, joint_values)
        log_robot_frame(rr, model, link_tfs, joint_values, i, fps)

    print(f"Logged {count} frames to rerun from {path}.")


def _scalar(rr, value: float):
    if hasattr(rr, "Scalar"):
        return rr.Scalar(float(value))
    return rr.Scalars(float(value))


def view_npz_features(path: Path, window_index: int, fps: float | None, max_frames: int) -> None:
    rr = _import_rerun()
    data = np.load(path, allow_pickle=True)
    if "windows" not in data:
        raise ValueError(f"{path} does not contain `windows`.")
    windows = data["windows"]
    if windows.ndim != 3 or windows.shape[-1] < 59:
        raise ValueError(f"Expected windows with shape (N, W, 59), got {windows.shape}.")
    if not 0 <= window_index < windows.shape[0]:
        raise ValueError(f"--window-index must be in [0, {windows.shape[0] - 1}], got {window_index}.")

    fps = float(fps if fps is not None else data["fps"][0] if "fps" in data else 50.0)
    window = windows[window_index]
    count = min(window.shape[0], max_frames)
    root_pos = window[:, 0:3]
    ee_pos = window[:, 38:53].reshape(window.shape[0], 5, 3)
    root_lin_vel = window[:, 53:56]
    ee_names = list(data["ee_body_names"]) if "ee_body_names" in data else [f"ee_{i}" for i in range(5)]

    rr.init(f"smp_npz_features_{path.stem}", spawn=True)
    rr.log("world/window_root_trajectory", rr.LineStrips3D([root_pos[:count]], colors=[[60, 180, 255]]), static=True)
    for i in range(count):
        rr.set_time("time", duration=i / fps)
        rr.log("world/root", rr.Points3D(root_pos[i : i + 1], radii=0.035, colors=[[255, 80, 80]]))
        rr.log("world/root_velocity", rr.Arrows3D(origins=root_pos[i : i + 1], vectors=root_lin_vel[i : i + 1], colors=[[80, 160, 255]]))
        rr.log("world/end_effectors", rr.Points3D(ee_pos[i], radii=0.03, colors=[[80, 255, 120]] * 5))
        for ee_idx, ee_name in enumerate(ee_names):
            rr.log(f"world/ee/{ee_name}", rr.Points3D(ee_pos[i, ee_idx : ee_idx + 1], radii=0.025))
        rr.log("root/height", _scalar(rr, float(root_pos[i, 2])))
        rr.log("root/speed_xy", _scalar(rr, float(np.linalg.norm(root_lin_vel[i, :2]))))

    print("NPZ windows are SMP features, not full robot trajectories. Logged root/EE feature points only.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play G1 CSV robot motion with rerun without Isaac Sim.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=Path, help="CSV motion file to play on the G1 URDF model.")
    group.add_argument("--npz", type=Path, help="SMP NPZ file to inspect as root/EE feature points.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="G1 URDF path used for CSV robot playback.")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS. Defaults to 50.")
    parser.add_argument("--max-frames", type=int, default=2000, help="Maximum frames to log.")
    parser.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw", help="CSV root quaternion order.")
    parser.add_argument("--no-meshes", action="store_true", help="Only show FK skeleton, without STL meshes.")
    parser.add_argument("--window-index", type=int, default=0, help="NPZ window index to inspect.")
    args = parser.parse_args()

    if args.csv is not None:
        view_csv_robot(
            args.csv,
            urdf_path=args.urdf,
            fps=float(args.fps or 50.0),
            max_frames=args.max_frames,
            quat_order=args.quat_order,
            log_meshes=not args.no_meshes,
        )
    else:
        view_npz_features(args.npz, window_index=args.window_index, fps=args.fps, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
