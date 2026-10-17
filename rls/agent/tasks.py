"""Multi-step task episodes for the skill executor, and their ground-truth scoring."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from rls.perception.cubes import in_bin, stacked
from rls.sim.env import EpisodeSpec, color_of, sample_episode
from rls.sim.variants import LAB, with_tasks

AGENT_SEED = 950_000


def multistep_specs(n: int, seed0: int = AGENT_SEED) -> list[EpisodeSpec]:
    """Three cubes; 'put A in the bin, then stack B on C'."""
    variant = replace(with_tasks(LAB, ("pick_place",)), name="multistep", n_distractors=(2, 2))
    out = []
    for i in range(n):
        spec = sample_episode(seed0 + i, variant, "pick_place")
        a, b, c = (o.color for o in spec.scene.objects)
        instr = f"put the {a} cube in the bin, then stack the {b} cube on the {c} cube"
        out.append(replace(spec, instruction=instr, extra={"steps": [["pick_place", a], ["stack", b, c]]}))
    return out


def write_specs(specs: list[EpisodeSpec], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s.to_json() for s in specs) + "\n")
    return path


def score_final_state(spec: dict[str, Any], final_state: dict[str, Any]) -> list[bool]:
    """Ground-truth success of each step from the simulator's final object poses."""
    objs = {color_of(EpisodeSpec.from_json(__import__("json").dumps(spec)), name): np.array(p[:3]) for name, p in final_state["objects"].items()}
    results = []
    for st in spec["extra"]["steps"]:
        if st[0] == "pick_place":
            results.append(in_bin(objs[st[1]], tuple(final_state["bin"]), half=0.075, max_z=0.06))
        else:
            results.append(stacked(objs[st[1]], objs[st[2]], xy_tol=0.025, min_dz=0.035))
    return results
