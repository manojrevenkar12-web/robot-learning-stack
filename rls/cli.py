"""`rls` command line: one entry point for the whole workflow.

rls record   --out data/raw/s0 --episodes 12 --profile configs/faults/realistic.yaml
rls ingest   data/raw/s0/bag --out data/staging/teleop/s0
rls synth    --out data/staging/synthetic --episodes 100
rls curate   data/staging --config configs/curation/default.yaml --out data/curated
rls release  data/curated/manifest.parquet --out data/release/rls_panda
rls train    configs/train/act_main.yaml
rls bench    runs/act_main/best --splits in_distribution,position_ood --episodes 30 --out reports/bench
rls export   runs/act_main/best --out runs/act_main/policy.onnx
rls deploy   runs/act_main/best --onnx runs/act_main/policy.onnx --episodes 10 --out reports/deploy
rls agent    runs/act_main/best --onnx runs/act_main/policy.onnx --episodes 10 --out reports/agent
rls inspect  data/raw/s0/bag
rls viewer   --data data --runs runs --reports reports
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=1, default=str))


def cmd_record(a: argparse.Namespace) -> None:
    from rls.ros.session import record_session

    print(record_session(a.out, a.episodes, a.profile, a.variant, a.tasks, a.seed_start, a.fault_seed))


def cmd_ingest(a: argparse.Namespace) -> None:
    from rls.data.ingest import ingest_session

    rep = ingest_session(a.bag, a.out)
    _print({"episodes": [(e["episode_id"], e["status"]) for e in rep["episodes"]], "clocks": rep["clocks"]})


def cmd_inspect(a: argparse.Namespace) -> None:
    from rls.data.bag import read_session

    s = read_session(a.bag)
    _print(
        {
            "episodes": [{k: e[k] for k in ("episode_index", "t_start", "t_end", "complete")} for e in s.episodes()],
            "streams": {k: len(v.stamp) for k, v in s.streams.items()},
            "read_errors": s.read_errors,
        }
    )


def cmd_synth(a: argparse.Namespace) -> None:
    from rls.data.synthetic import generate
    from rls.sim.variants import SYNTH_DR, with_tasks

    paths = generate(a.out, with_tasks(SYNTH_DR, tuple(a.tasks.split(","))), range(a.seed_start, a.seed_start + a.episodes), tasks=tuple(a.tasks.split(",")))
    print(f"{len(paths)} episodes -> {a.out}")


def cmd_curate(a: argparse.Namespace) -> None:
    from rls.data.curate import CurationConfig, curate, scan, write_outputs

    df, rep = curate(scan(a.staging), CurationConfig.load(a.config))
    write_outputs(df, rep, a.out)
    _print(rep)


def cmd_release(a: argparse.Namespace) -> None:
    py = os.environ.get("RLS_LEROBOT_PYTHON", sys.executable)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), HF_HUB_OFFLINE="1")
    subprocess.run(
        [py, "-m", "rls.data.export_lerobot", "--manifest", str(a.manifest), "--out", str(a.out), "--vcodec", a.vcodec, "--crf", str(a.crf)],
        check=True,
        env=env,
    )


def cmd_train(a: argparse.Namespace) -> None:
    from rls.learn.train import TrainConfig, train

    _print(train(TrainConfig.load(a.config, steps=a.steps, run_name=a.run_name), resume=a.resume))


def cmd_bench(a: argparse.Namespace) -> None:
    from rls.learn.benchmark import EnvConditions, benchmark

    res = benchmark(a.bundle, a.splits.split(","), a.episodes, a.out, tuple(a.tasks.split(",")), EnvConditions(obs_latency_steps=a.obs_latency), a.onnx)
    _print({k: {"success": v.success_rate, "ci95": v.ci95, "failures": v.failures} for k, v in res.items()})


def cmd_export(a: argparse.Namespace) -> None:
    from rls.learn.export import export_onnx

    _print(export_onnx(a.bundle, a.out))


def cmd_deploy(a: argparse.Namespace) -> None:
    from rls.ros.evaluate import deploy_eval

    _print(deploy_eval(a.bundle, a.onnx, a.out, a.episodes, a.split, a.profile, a.clock_offsets))


def cmd_agent(a: argparse.Namespace) -> None:
    from rls.ros.evaluate import agent_eval

    _print(agent_eval(a.bundle, a.onnx, a.out, a.episodes, a.profile, a.clock_offsets))


def cmd_viewer(a: argparse.Namespace) -> None:
    import uvicorn

    from rls.tools.viewer import create_app

    uvicorn.run(create_app(a.data, a.runs, a.reports), host=a.host, port=a.port)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rls", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record", help="record a teleop session over ROS 2 (needs ROS 2 Jazzy)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--profile", type=Path, default=Path("configs/faults/realistic.yaml"))
    p.add_argument("--variant", default="lab")
    p.add_argument("--tasks", default="pick_place,stack")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--fault-seed", type=int, default=0)
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("ingest", help="bag -> health-checked, aligned staging episodes (no ROS needed)")
    p.add_argument("bag", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("inspect", help="list episodes and streams in a bag")
    p.add_argument("bag", type=Path)
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("synth", help="generate synthetic episodes in simulation")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=5000)
    p.add_argument("--tasks", default="pick_place,stack")
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("curate", help="filter, deduplicate and split staged episodes")
    p.add_argument("staging", type=Path, nargs="+")
    p.add_argument("--config", type=Path, default=Path("configs/curation/default.yaml"))
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_curate)
    p = sub.add_parser("release", help="export a curated manifest as a LeRobot v3.0 dataset")
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vcodec", default="libsvtav1")
    p.add_argument("--crf", type=int, default=30)
    p.set_defaults(fn=cmd_release)
    p = sub.add_parser("train", help="train or fine-tune a policy")
    p.add_argument("config", type=Path)
    p.add_argument("--steps", type=int)
    p.add_argument("--run-name")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("bench", help="closed-loop benchmark in simulation")
    p.add_argument("bundle", type=Path)
    p.add_argument("--splits", default="in_distribution")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--tasks", default="pick_place,stack")
    p.add_argument("--obs-latency", type=int, default=0)
    p.add_argument("--onnx", type=Path)
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("export", help="export a policy bundle to ONNX (parity-checked)")
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_export)
    for name, fn in (("deploy", cmd_deploy), ("agent", cmd_agent)):
        p = sub.add_parser(
            name, help="run the policy on the robot over ROS 2" if name == "deploy" else "run multi-step instructions through the skill executor"
        )
        p.add_argument("bundle", type=Path)
        p.add_argument("--onnx", type=Path)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--episodes", type=int, default=10)
        p.add_argument("--profile", type=Path, default=Path("configs/faults/realistic.yaml"))
        p.add_argument("--clock-offsets", default='{"cam_wrist": 0.12}')
        if name == "deploy":
            p.add_argument("--split", default="in_distribution")
        p.set_defaults(fn=fn)
    p = sub.add_parser("viewer", help="dataset / training / benchmark viewer (web UI)")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_viewer)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
