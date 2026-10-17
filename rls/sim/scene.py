"""MuJoCo scene: Franka Panda (MuJoCo Menagerie) on a table with objects, a bin,
a front RGB-D camera, a wrist camera and a wrist force-torque sensor.

The robot model is vendored unchanged from MuJoCo Menagerie (Apache-2.0); the
wrist camera, force-torque site and grip site are spliced in at load time so the
vendored file stays byte-identical to upstream.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets" / "franka_panda"

# Distinct, nameable colours: the language instruction refers to them.
COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.15, 0.12),
    "green": (0.15, 0.65, 0.2),
    "blue": (0.15, 0.3, 0.85),
    "yellow": (0.95, 0.8, 0.1),
    "purple": (0.55, 0.2, 0.7),
    "orange": (0.95, 0.5, 0.1),
}

HOME_Q = np.array([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.785398])
N_ARM = 7

# Inserted into the hand body: force-torque site at the flange, a site between the
# fingertips used as the tool centre point, and a wrist camera looking past the fingers.
_HAND_ADDITIONS = """
                      <site name="ft_site" pos="0 0 0" size="0.005" rgba="1 0 1 0" group="4"/>
                      <site name="grip_site" pos="0 0 0.1034" size="0.005" rgba="0 1 1 0" group="4"/>
                      <camera name="wrist" pos="0.085 0 0.03" xyaxes="0 1 0 0.866 0 0.5" fovy="80"/>"""


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    color: str
    half_size: float
    mass: float = 0.08
    friction: float = 1.0


@dataclass(frozen=True)
class SceneSpec:
    """Everything that changes the compiled model. Poses are set at reset instead."""

    objects: tuple[ObjectSpec, ...]
    bin_pos: tuple[float, float] = (0.45, -0.28)
    table_rgb: tuple[float, float, float] = (0.62, 0.55, 0.45)
    light_dir: tuple[float, float, float] = (0.0, 0.3, -1.0)
    light_diffuse: float = 0.7
    front_cam_pos: tuple[float, float, float] = (1.05, 0.0, 0.78)
    front_cam_target: tuple[float, float, float] = (0.45, -0.06, 0.0)
    image_hw: tuple[int, int] = (120, 160)
    wrist_hw: tuple[int, int] = (96, 128)
    timestep: float = 1.0 / 480.0
    extra: dict[str, float] = field(default_factory=dict, compare=False, hash=False)


BIN_INNER_HALF = 0.075
BIN_WALL_H = 0.05
BIN_WALL_T = 0.006


def look_at_xyaxes(pos: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """MuJoCo camera xyaxes for a camera at ``pos`` looking at ``target``."""
    up = np.array([0.0, 0.0, 1.0]) if up is None else up
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    return np.concatenate([right, cam_up])


@functools.lru_cache(maxsize=1)
def _asset_bytes() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for p in (ASSETS / "assets").iterdir():
        files[f"assets/{p.name}"] = p.read_bytes()
    panda = (ASSETS / "panda.xml").read_text()
    anchor = '<geom mesh="hand_c" class="collision"/>'
    if anchor not in panda:
        raise RuntimeError("vendored panda.xml changed: hand anchor not found")
    panda = panda.replace(anchor, anchor + _HAND_ADDITIONS, 1)
    # Render with the (much lighter) collision meshes: the visual meshes have ~400k
    # vertices and cost ~30 ms per frame on a CPU rasteriser, the collision meshes ~3 ms.
    coll = '<geom type="mesh" group="3"/>'
    if coll not in panda:
        raise RuntimeError("vendored panda.xml changed: collision default not found")
    panda = panda.replace(coll, '<geom type="mesh" group="3" rgba="0.9 0.9 0.92 1"/>', 1)
    files["panda.xml"] = panda.encode()
    return files


def _rgba(rgb: tuple[float, float, float], a: float = 1.0) -> str:
    return " ".join(f"{c:.3f}" for c in (*rgb, a))


def scene_xml(spec: SceneSpec) -> str:
    h, w = spec.image_hw
    cam_pos = np.array(spec.front_cam_pos)
    xy = look_at_xyaxes(cam_pos, np.array(spec.front_cam_target))
    bx, by = spec.bin_pos
    wall = BIN_INNER_HALF + BIN_WALL_T
    objects = []
    for o in spec.objects:
        objects.append(
            f'''    <body name="{o.name}" pos="0.5 0 {o.half_size:.4f}">
      <freejoint name="{o.name}_joint"/>
      <geom name="{o.name}" type="box" size="{o.half_size:.4f} {o.half_size:.4f} {o.half_size:.4f}"
            rgba="{_rgba(COLORS[o.color])}" mass="{o.mass:.4f}" friction="{o.friction:.3f} 0.03 0.003"
            condim="4" solref="0.01 1" priority="1"/>
    </body>'''
        )
    ld = np.array(spec.light_dir) / np.linalg.norm(spec.light_dir)
    return f"""<mujoco model="rls_scene">
  <include file="panda.xml"/>
  <option timestep="{spec.timestep:.8f}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <quality shadowsize="0"/>
    <global offwidth="{max(w, 320)}" offheight="{max(h, 240)}"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <asset>
    <material name="table" rgba="{_rgba(spec.table_rgb)}" reflectance="0"/>
    <material name="bin" rgba="0.3 0.3 0.32 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.4 -0.3 1.6" dir="{ld[0]:.3f} {ld[1]:.3f} {ld[2]:.3f}" diffuse="{spec.light_diffuse:.3f} {spec.light_diffuse:.3f} {spec.light_diffuse:.3f}" directional="true"/>
    <geom name="table" type="plane" size="1.2 1.2 0.05" material="table" priority="1" friction="0.8 0.02 0.002"/>
    <camera name="front" pos="{cam_pos[0]:.4f} {cam_pos[1]:.4f} {cam_pos[2]:.4f}" xyaxes="{" ".join(f"{v:.5f}" for v in xy)}" fovy="50"/>
    <body name="bin" pos="{bx:.4f} {by:.4f} 0">
      <geom name="bin_floor" type="box" pos="0 0 0.003" size="{wall:.4f} {wall:.4f} 0.003" material="bin"/>
      <geom name="bin_wall_px" type="box" pos="{wall:.4f} 0 {BIN_WALL_H / 2:.4f}" size="{BIN_WALL_T:.4f} {wall:.4f} {BIN_WALL_H / 2:.4f}" material="bin"/>
      <geom name="bin_wall_nx" type="box" pos="{-wall:.4f} 0 {BIN_WALL_H / 2:.4f}" size="{BIN_WALL_T:.4f} {wall:.4f} {BIN_WALL_H / 2:.4f}" material="bin"/>
      <geom name="bin_wall_py" type="box" pos="0 {wall:.4f} {BIN_WALL_H / 2:.4f}" size="{wall:.4f} {BIN_WALL_T:.4f} {BIN_WALL_H / 2:.4f}" material="bin"/>
      <geom name="bin_wall_ny" type="box" pos="0 {-wall:.4f} {BIN_WALL_H / 2:.4f}" size="{wall:.4f} {BIN_WALL_T:.4f} {BIN_WALL_H / 2:.4f}" material="bin"/>
    </body>
{chr(10).join(objects)}
  </worldbody>
  <sensor>
    <force name="ft_force" site="ft_site"/>
    <torque name="ft_torque" site="ft_site"/>
  </sensor>
</mujoco>
"""


def build_model(spec: SceneSpec) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(scene_xml(spec), assets=_asset_bytes())


def render_option() -> mujoco.MjvOption:
    """Show collision geometry (group 3) instead of the heavy visual meshes (group 2)."""
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 0
    opt.geomgroup[3] = 1
    return opt
