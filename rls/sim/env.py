"""Panda tabletop environment: pick-and-place into a bin and stacking.

Episodes are fully described by an :class:`EpisodeSpec` (scene + initial poses +
task), which is stored with every recording, so any episode can be re-simulated
exactly. The same spec drives the synthetic generator, the ROS 2 driver and the
policy benchmarks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import mujoco
import numpy as np

from rls.sim.scene import (
    BIN_INNER_HALF,
    BIN_WALL_H,
    COLORS,
    HOME_Q,
    N_ARM,
    ObjectSpec,
    SceneSpec,
    build_model,
    render_option,
)

CONTROL_HZ = 15
TASKS = ("pick_place", "stack")

# Joint limits of the Panda arm (rad), used to clip commanded targets.
Q_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


@dataclass(frozen=True)
class Variant:
    """Ranges an episode is sampled from. Benchmark splits are different Variants."""

    name: str = "train"
    x_range: tuple[float, float] = (0.40, 0.60)
    y_range: tuple[float, float] = (-0.12, 0.18)
    yaw_range: tuple[float, float] = (-0.6, 0.6)
    colors: tuple[str, ...] = ("red", "green", "blue", "yellow")
    n_distractors: tuple[int, int] = (1, 1)
    half_size: tuple[float, float] = (0.020, 0.024)
    mass: tuple[float, float] = (0.05, 0.12)
    friction: tuple[float, float] = (0.8, 1.2)
    bin_xy_jitter: float = 0.02
    light_jitter: float = 0.25
    table_jitter: float = 0.06
    cam_pos_jitter: float = 0.0
    tasks: tuple[str, ...] = ("pick_place",)
    target_colors: tuple[str, ...] | None = None  # restrict the target's colour (e.g. unseen colours)


@dataclass
class EpisodeSpec:
    seed: int
    task: str
    instruction: str
    target: str
    support: str | None
    scene: SceneSpec
    init_poses: dict[str, tuple[float, float, float, float]]  # x, y, z, yaw
    variant: str = "train"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> EpisodeSpec:
        d = json.loads(s)
        sc = d.pop("scene")
        objs = tuple(ObjectSpec(**o) for o in sc.pop("objects"))
        for k in ("bin_pos", "table_rgb", "light_dir", "front_cam_pos", "front_cam_target", "image_hw", "wrist_hw"):
            if k in sc:
                sc[k] = tuple(sc[k])
        d["init_poses"] = {k: tuple(v) for k, v in d["init_poses"].items()}
        return EpisodeSpec(scene=SceneSpec(objects=objs, **sc), **d)


def instruction_for(task: str, target_color: str, support_color: str | None) -> str:
    if task == "pick_place":
        return f"put the {target_color} cube in the bin"
    if task == "stack":
        return f"stack the {target_color} cube on the {support_color} cube"
    raise ValueError(task)


def sample_episode(seed: int, variant: Variant, task: str | None = None) -> EpisodeSpec:
    """Sample a collision-free scene for ``task`` from ``variant``."""
    rng = np.random.default_rng(seed)
    task = task or str(rng.choice(variant.tasks))
    n_distractors = int(rng.integers(variant.n_distractors[0], variant.n_distractors[1] + 1))
    n_obj = 1 + (1 if task == "stack" else 0) + n_distractors
    if variant.target_colors:
        first = str(rng.choice(variant.target_colors))
        rest = [c for c in variant.colors if c != first]
        colors = [first, *rng.choice(rest, size=n_obj - 1, replace=False)]
    else:
        colors = list(rng.choice(variant.colors, size=n_obj, replace=False))
    objects = []
    for i, c in enumerate(colors):
        objects.append(
            ObjectSpec(
                name=f"cube_{i}",
                color=str(c),
                half_size=float(rng.uniform(*variant.half_size)),
                mass=float(rng.uniform(*variant.mass)),
                friction=float(rng.uniform(*variant.friction)),
            )
        )
    poses: dict[str, tuple[float, float, float, float]] = {}
    placed: list[np.ndarray] = []
    bin_pos = np.array([0.45, -0.28]) + rng.uniform(-variant.bin_xy_jitter, variant.bin_xy_jitter, 2)
    for o in objects:
        for _ in range(200):
            p = np.array([rng.uniform(*variant.x_range), rng.uniform(*variant.y_range)])
            far_from_bin = bool(np.any(np.abs(p - bin_pos) > BIN_INNER_HALF + 0.05))
            if far_from_bin and all(np.linalg.norm(p - q) > 0.09 for q in placed):
                break
        else:
            raise RuntimeError(f"could not place objects for seed {seed}")
        placed.append(p)
        poses[o.name] = (float(p[0]), float(p[1]), o.half_size, float(rng.uniform(*variant.yaw_range)))
    target = objects[0]
    support = objects[1] if task == "stack" else None
    lj = variant.light_jitter
    light_dir = (float(rng.uniform(-lj, lj)), float(0.3 + rng.uniform(-lj, lj)), -1.0)
    base_table = np.array([0.62, 0.55, 0.45])
    table = tuple(float(v) for v in np.clip(base_table + rng.uniform(-variant.table_jitter, variant.table_jitter, 3), 0, 1))
    cam = np.array([1.05, 0.0, 0.78]) + rng.uniform(-variant.cam_pos_jitter, variant.cam_pos_jitter, 3)
    scene = SceneSpec(
        objects=tuple(objects),
        bin_pos=(float(bin_pos[0]), float(bin_pos[1])),
        table_rgb=table,  # type: ignore[arg-type]
        light_dir=light_dir,
        light_diffuse=float(0.7 + rng.uniform(-lj, lj) * 0.6),
        front_cam_pos=(float(cam[0]), float(cam[1]), float(cam[2])),
    )
    return EpisodeSpec(
        seed=seed,
        task=task,
        instruction=instruction_for(task, target.color, support.color if support else None),
        target=target.name,
        support=support.name if support else None,
        scene=scene,
        init_poses=poses,
        variant=variant.name,
    )


@dataclass
class Outcome:
    success: bool
    failure_mode: str  # "success" | no_grasp | wrong_object | dropped | misplaced | timeout
    target_final: tuple[float, float, float]
    max_target_height: float
    distractor_moved: float
    max_contact_force: float
    steps: int


class PandaEnv:
    """Joint-position-controlled Panda with front RGB-D, wrist RGB and wrist F/T."""

    def __init__(self, render: bool = True, depth: bool = True) -> None:
        self.render_enabled = render
        self.depth_enabled = depth
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.spec: EpisodeSpec | None = None
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._render_opt = render_option()
        self._scene_key: str | None = None

    # ------------------------------------------------------------------ setup
    def reset(self, spec: EpisodeSpec) -> dict[str, np.ndarray]:
        key = json.dumps(asdict(spec.scene), sort_keys=True)
        if key != self._scene_key:
            self.close_renderers()
            self.model = build_model(spec.scene)
            self.data = mujoco.MjData(self.model)
            self._scene_key = key
            self.substeps = int(round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))
            m = self.model
            self.grip_site = m.site("grip_site").id
            self.finger_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("left_finger", "right_finger")}
            self.obj_qadr = {o.name: m.jnt_qposadr[m.joint(f"{o.name}_joint").id] for o in spec.scene.objects}
            self.obj_geom = {o.name: m.geom(o.name).id for o in spec.scene.objects}
            self.table_geom = m.geom("table").id
            robot_bodies = {m.body(n).id for n in ("link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")}
            self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies}
        assert self.model is not None and self.data is not None
        m, d = self.model, self.data
        self.spec = spec
        mujoco.mj_resetData(m, d)
        d.qpos[:N_ARM] = HOME_Q
        d.qpos[N_ARM : N_ARM + 2] = 0.04
        for name, (x, y, z, yaw) in spec.init_poses.items():
            a = self.obj_qadr[name]
            d.qpos[a : a + 3] = (x, y, z + 0.0005)
            d.qpos[a + 3 : a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        d.ctrl[:N_ARM] = HOME_Q
        d.ctrl[N_ARM] = 255
        mujoco.mj_forward(m, d)
        for _ in range(20):  # let objects settle
            mujoco.mj_step(m, d)
        self._init_obj = {n: self.object_pos(n).copy() for n in spec.init_poses}
        self._max_h = 0.0
        self._max_force = 0.0
        self._lifted_other = False
        self.steps = 0
        self.time = 0.0
        return self.observe()

    # ------------------------------------------------------------- stepping
    def step(self, action: np.ndarray, substep_callback=None) -> dict[str, np.ndarray]:
        assert self.model is not None and self.data is not None
        a = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:N_ARM] = np.clip(a[:N_ARM], Q_LOW, Q_HIGH)
        self.data.ctrl[N_ARM] = float(np.clip(a[N_ARM], 0.0, 1.0)) * 255.0
        for i in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if substep_callback is not None:
                substep_callback(i)
        self.steps += 1
        self.time = float(self.data.time)
        self._track()
        return self.observe()

    def _track(self) -> None:
        assert self.spec is not None and self.data is not None
        z = self.object_pos(self.spec.target)[2] - self.spec.init_poses[self.spec.target][2]
        self._max_h = max(self._max_h, float(z))
        for n in self.spec.init_poses:
            if n not in (self.spec.target, self.spec.support) and self.object_pos(n)[2] - self.spec.init_poses[n][2] > 0.03:
                self._lifted_other = True
        f = np.zeros(6)
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if self.table_geom in pair and pair & self.robot_geoms:
                mujoco.mj_contactForce(self.model, d, i, f)
                self._max_force = max(self._max_force, float(abs(f[0])))

    # ---------------------------------------------------------- observations
    def object_pos(self, name: str) -> np.ndarray:
        assert self.data is not None
        a = self.obj_qadr[name]
        return self.data.qpos[a : a + 3]

    def object_yaw(self, name: str) -> float:
        assert self.data is not None
        a = self.obj_qadr[name]
        w, x, y, z = self.data.qpos[a + 3 : a + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def ee_pos(self) -> np.ndarray:
        assert self.data is not None
        return self.data.site_xpos[self.grip_site].copy()

    def gripper_width(self) -> float:
        assert self.data is not None
        return float(self.data.qpos[N_ARM] + self.data.qpos[N_ARM + 1])

    def proprio(self) -> dict[str, np.ndarray]:
        assert self.data is not None and self.model is not None
        d = self.data
        return {
            "qpos": d.qpos[:N_ARM].copy(),
            "qvel": d.qvel[:N_ARM].copy(),
            "gripper": np.array([self.gripper_width() / 0.08]),
            "ft": np.concatenate([d.sensor("ft_force").data, d.sensor("ft_torque").data]).copy(),
        }

    def render(self, camera: str, depth: bool = False) -> np.ndarray:
        assert self.model is not None and self.data is not None and self.spec is not None
        hw = self.spec.scene.image_hw if camera == "front" else self.spec.scene.wrist_hw
        key = f"{camera}:{'d' if depth else 'c'}"
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, hw[0], hw[1])
            if depth:
                r.enable_depth_rendering()
            self._renderers[key] = r
        r.update_scene(self.data, camera, scene_option=self._render_opt)
        return r.render().copy()

    def observe(self) -> dict[str, np.ndarray]:
        obs = self.proprio()
        if self.render_enabled:
            obs["front_rgb"] = self.render("front")
            obs["wrist_rgb"] = self.render("wrist")
            if self.depth_enabled:
                obs["front_depth"] = self.render("front", depth=True).astype(np.float32)
        return obs

    def camera_params(self, camera: str) -> dict[str, Any]:
        """Intrinsics (pinhole) and world pose of a camera, as a ROS CameraInfo would carry."""
        assert self.model is not None and self.data is not None and self.spec is not None
        m, d = self.model, self.data
        cid = m.camera(camera).id
        h, w = self.spec.scene.image_hw if camera == "front" else self.spec.scene.wrist_hw
        fovy = np.deg2rad(m.cam_fovy[cid])
        f = 0.5 * h / np.tan(fovy / 2)
        return {
            "width": w,
            "height": h,
            "K": [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0],
            "cam_pos": d.cam_xpos[cid].tolist(),
            "cam_mat": d.cam_xmat[cid].reshape(3, 3).tolist(),
        }

    def privileged(self) -> dict[str, Any]:
        assert self.spec is not None
        return {
            "ee": self.ee_pos().tolist(),
            "objects": {n: [*self.object_pos(n).tolist(), self.object_yaw(n)] for n in self.spec.init_poses},
            "bin": list(self.spec.scene.bin_pos),
        }

    # -------------------------------------------------------------- outcome
    def in_bin(self, name: str) -> bool:
        assert self.spec is not None
        p = self.object_pos(name)
        bx, by = self.spec.scene.bin_pos
        return bool(abs(p[0] - bx) < BIN_INNER_HALF and abs(p[1] - by) < BIN_INNER_HALF and p[2] < BIN_WALL_H + 0.01)

    def goal_xy(self) -> np.ndarray:
        assert self.spec is not None
        if self.spec.task == "pick_place":
            return np.array(self.spec.scene.bin_pos)
        assert self.spec.support is not None
        return self.object_pos(self.spec.support)[:2].copy()

    def holding(self) -> bool:
        assert self.data is not None and self.spec is not None
        tg = self.obj_geom[self.spec.target]
        touching = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == tg and c.geom2 in self.finger_geoms:
                touching.add(self.model.geom_bodyid[c.geom2])  # type: ignore[union-attr]
            elif c.geom2 == tg and c.geom1 in self.finger_geoms:
                touching.add(self.model.geom_bodyid[c.geom1])  # type: ignore[union-attr]
        return len(touching) == 2

    def task_success(self) -> bool:
        assert self.spec is not None
        if self.spec.task == "pick_place":
            return self.in_bin(self.spec.target) and not self.holding()
        assert self.spec.support is not None
        top, bot = self.object_pos(self.spec.target), self.object_pos(self.spec.support)
        hs_top = next(o.half_size for o in self.spec.scene.objects if o.name == self.spec.target)
        hs_bot = next(o.half_size for o in self.spec.scene.objects if o.name == self.spec.support)
        aligned = np.linalg.norm(top[:2] - bot[:2]) < 0.6 * hs_bot
        on_top = abs(top[2] - (bot[2] + hs_bot + hs_top)) < 0.008
        return bool(aligned and on_top and not self.holding())

    def outcome(self) -> Outcome:
        assert self.spec is not None
        tgt = self.object_pos(self.spec.target)
        moved = max(
            (
                float(np.linalg.norm(self.object_pos(n)[:2] - self._init_obj[n][:2]))
                for n in self.spec.init_poses
                if n not in (self.spec.target, self.spec.support)
            ),
            default=0.0,
        )
        ok = self.task_success()
        if ok:
            mode = "success"
        elif self._lifted_other or any(self.in_bin(n) for n in self.spec.init_poses if n != self.spec.target):
            mode = "wrong_object"
        elif self._max_h < 0.03:
            mode = "no_grasp"
        elif self.holding():
            mode = "timeout"
        elif np.linalg.norm(tgt[:2] - self.goal_xy()) < 0.07:
            mode = "misplaced"
        else:
            mode = "dropped"
        return Outcome(
            success=ok,
            failure_mode=mode,
            target_final=tuple(float(v) for v in tgt),  # type: ignore[arg-type]
            max_target_height=self._max_h,
            distractor_moved=moved,
            max_contact_force=self._max_force,
            steps=self.steps,
        )

    def close_renderers(self) -> None:
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers = {}


def color_of(spec: EpisodeSpec, obj: str) -> str:
    return next(o.color for o in spec.scene.objects if o.name == obj)


assert set(COLORS) >= {"red", "green", "blue", "yellow"}
