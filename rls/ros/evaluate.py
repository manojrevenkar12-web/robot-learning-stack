"""Evaluate policies on the robot over ROS 2 (sim driver in lockstep), not just in-process.

``deploy_eval`` runs the same benchmark episodes as :mod:`rls.learn.benchmark`
but through the full stack: driver with sensor timing/faults -> compressed
images over DDS -> policy runtime (freshness/sync checks, ONNX) -> joint
commands. Comparing the two isolates what the deployment path costs.

``agent_eval`` runs multi-step instructions through the skill executor.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rls.agent.tasks import multistep_specs, score_final_state, write_specs
from rls.learn.benchmark import bench_seed, wilson
from rls.ros.session import record_session
from rls.sim.env import sample_episode
from rls.sim.variants import SPLITS, with_tasks


def deploy_eval(
    bundle: Path,
    onnx: Path | None,
    out: Path,
    episodes: int,
    split: str = "in_distribution",
    profile: Path = Path("configs/faults/realistic.yaml"),
    clock_offsets: str = '{"cam_wrist": 0.12}',
    tasks: tuple[str, ...] = ("pick_place", "stack"),
) -> dict[str, Any]:
    v = with_tasks(SPLITS[split], tasks)
    specs = [sample_episode(bench_seed(split, i), v, tasks[i % len(tasks)]) for i in range(episodes)]
    out.mkdir(parents=True, exist_ok=True)
    spec_file = write_specs(specs, out / "specs.jsonl")
    pp: dict[str, object] = {"bundle": str(bundle), "clock_offsets": clock_offsets}
    if onnx:
        pp["onnx"] = str(onnx)
    record_session(
        out / "session",
        episodes,
        profile,
        episode_specs=spec_file,
        operator="policy",
        record=False,
        policy_params=pp,
        driver_params={"max_steps": 300, "success_stop_steps": 8, "command_timeout_s": 2.0},
    )
    gt = json.loads((out / "session" / "ground_truth.json").read_text())
    rt = json.loads((out / "session" / "policy_runtime.json").read_text())
    succ = [e["outcome"]["success"] for e in gt["episodes"]]
    res = {
        "split": split,
        "episodes": len(succ),
        "success_rate": sum(succ) / max(len(succ), 1),
        "ci95": wilson(sum(succ), len(succ)),
        "failures": dict(Counter(e["outcome"]["failure_mode"] for e in gt["episodes"] if not e["outcome"]["success"])),
        "runtime": rt,
        "fault_profile": str(profile),
    }
    (out / "deploy_eval.json").write_text(json.dumps(res, indent=1))
    return res


def agent_eval(
    bundle: Path, onnx: Path | None, out: Path, episodes: int, profile: Path = Path("configs/faults/realistic.yaml"), clock_offsets: str = '{"cam_wrist": 0.12}'
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    specs = multistep_specs(episodes)
    spec_file = write_specs(specs, out / "specs.jsonl")
    pp: dict[str, object] = {"bundle": str(bundle), "clock_offsets": clock_offsets}
    if onnx:
        pp["onnx"] = str(onnx)
    record_session(
        out / "session",
        episodes,
        profile,
        episode_specs=spec_file,
        operator="agent",
        record=False,
        policy_params=pp,
        driver_params={"max_steps": 1200, "settle_steps": 3, "command_timeout_s": 2.0},
    )
    gt = json.loads((out / "session" / "ground_truth.json").read_text())
    reports = json.loads((out / "session" / "agent_report.json").read_text()) if (out / "session" / "agent_report.json").exists() else []
    rows = []
    for e in gt["episodes"]:
        spec = json.loads(specs[e["episode_index"]].to_json())
        truth = score_final_state(spec, e["final_state"])
        rep: dict = next((r for r in reports if r["episode_index"] == e["episode_index"]), {})
        rows.append(
            {
                "episode_index": e["episode_index"],
                "instruction": e["instruction"],
                "steps_true": truth,
                "executor": rep.get("steps", []),
                "plan": rep.get("plan"),
            }
        )
    step1 = [r["steps_true"][0] for r in rows]
    both = [all(r["steps_true"]) for r in rows]
    agree = [
        (s.get("result") == "success") == t
        for r in rows
        for s, t in zip([x for x in r["executor"] if x.get("result") != "retry"], r["steps_true"], strict=False)
    ]
    res = {
        "episodes": len(rows),
        "step1_success": sum(step1) / max(len(rows), 1),
        "task_success": sum(both) / max(len(rows), 1),
        "task_ci95": wilson(sum(both), len(rows)),
        "retries": sum(1 for r in rows for s in r["executor"] if s.get("result") == "retry"),
        "postcondition_agreement": sum(agree) / max(len(agree), 1),
        "rows": rows,
    }
    (out / "agent_eval.json").write_text(json.dumps(res, indent=1, default=str))
    return res
