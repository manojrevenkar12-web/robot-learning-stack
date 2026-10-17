"""Training, benchmark and ablation studies, run as a small dependency-aware job queue.

    python scripts/run_studies.py --workers 2            # everything (hours on a 2-core CPU)
    python scripts/run_studies.py --only main            # one study
    python scripts/run_studies.py --report-only          # rebuild reports/studies/STUDIES.md

Every job is idempotent: it is skipped when its output marker exists, so the
script can be interrupted and restarted. Logs go to reports/studies/logs/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ALL_SPLITS = "in_distribution,position_ood,two_distractors,lighting_shift,camera_shift,unseen_color"


@dataclass
class Job:
    name: str
    cmd: list[str]
    marker: Path
    deps: list[str] = field(default_factory=list)
    study: str = ""


def train(name: str, cfg: str, study: str, deps: list[str] | None = None, **flags: str | int) -> Job:
    cmd = [PY, "-m", "rls.learn.train", f"configs/train/{cfg}.yaml", "--run-name", name]
    for k, v in flags.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return Job(f"train:{name}", cmd, ROOT / "runs" / name / "summary.json", deps or [], study)


def export(name: str, study: str) -> Job:
    return Job(
        f"export:{name}",
        [PY, "-m", "rls.cli", "export", f"runs/{name}/best", "--out", f"runs/{name}/policy.onnx"],
        ROOT / "runs" / name / "policy.onnx",
        [f"train:{name}"],
        study,
    )


def bench(run: str, rid: str, splits: str, episodes: int, study: str, obs_latency: int = 0, action_delay: int = 0, videos: bool = False) -> Job:
    out = f"reports/bench/{rid}"
    cmd = [
        PY,
        "-m",
        "rls.learn.benchmark",
        f"runs/{run}/best",
        "--onnx",
        f"runs/{run}/policy.onnx",
        "--splits",
        splits,
        "--episodes",
        str(episodes),
        "--out",
        out,
    ]
    cmd += ["--obs-latency", str(obs_latency), "--action-delay", str(action_delay)]
    if not videos:
        cmd += ["--videos-per-mode", "0"]
    return Job(f"bench:{rid}", cmd, ROOT / out / "benchmark.json", [f"export:{run}"], study)


ABLATION = [  # (run, what changed); trained by configs/train/ablation/*.yaml, 1500 steps unless noted
    ("act_joint", "first recipe: joint-position actions, DrQ shift 4 px (3500 steps)"),
    ("pilot_ee", "tool-pose (EE) actions"),
    ("pilot_noaug", "EE, no image shift augmentation"),
    ("pilot_kp", "EE, spatial-softmax keypoint readout, shift 2 px"),
    ("pilot_nostate", "EE, no proprioception input"),
    ("pilot_sd", "EE, proprioception noise 0.1 and dropout 0.5"),
    ("pilot_aux", "EE, auxiliary object localisation"),
    ("pilot_aux_sd", "EE, auxiliary localisation + proprioception noise 0.1 and dropout 0.3"),
    ("pilot_film", "previous row + FiLM: the instruction modulates every CNN stage (final recipe)"),
]


def plan(episodes: int, study_episodes: int) -> list[Job]:
    jobs: list[Job] = []
    # 1. main policy (final recipe, 5000 steps) on every generalisation split
    jobs += [train("act_main", "act_main", "main"), export("act_main", "main"), bench("act_main", "act_main", ALL_SPLITS, episodes, "main", videos=True)]
    # 2. data source and 3. policy class: same recipe, 1500 steps; pilot_film is the matched mixed-data chunked model
    for src in ("teleop", "synthetic"):
        jobs += [train(f"src_{src}", f"src_{src}", "data_source"), export(f"src_{src}", "data_source")]
    for name in ("bc_single_step", "diffusion_chunk"):
        jobs += [train(name, name, "policy_class"), export(name, "policy_class")]
    for run in ("src_teleop", "src_synthetic", "pilot_film"):
        jobs.append(bench(run, f"{run}_sources", "in_distribution,lighting_shift,camera_shift", study_episodes, "data_source"))
    for run in ("bc_single_step", "diffusion_chunk", "pilot_film"):
        jobs.append(bench(run, f"{run}_class", "in_distribution,position_ood", study_episodes, "policy_class"))
    # 4. ablation: what it took to make closed-loop behaviour cloning work on this budget
    for run, _ in [*ABLATION, ("act_main", "")]:
        if run != "act_main":
            jobs.append(bench(run, f"abl_{run}", "in_distribution", study_episodes, "ablation"))
        deps = ["train:act_main"] if run == "act_main" else []
        probe_cmd = [PY, "scripts/probe_policy.py", f"runs/{run}/best", "--episodes", "40", "--out", f"reports/probes/{run}.json"]
        jobs.append(Job(f"probe:{run}", probe_cmd, ROOT / "reports" / "probes" / f"{run}.json", deps, "ablation"))
    # 5. transfer under deployment conditions the policy never saw
    for lat in (2, 4):
        jobs.append(bench("act_main", f"act_main_obs_latency_{lat}", "in_distribution", episodes, "latency", obs_latency=lat))
    jobs.append(bench("act_main", "act_main_action_delay_2", "in_distribution", episodes, "latency", action_delay=2))
    # 6. fine-tuning to unseen colours from few demos
    data_job = Job(
        "data:unseen_color",
        [PY, "scripts/build_finetune_data.py", "--root", "data", "--episodes", "48"],
        ROOT / "data" / "release" / "rls_unseen_color" / "meta" / "rls_release.json",
        [],
        "finetune",
    )
    jobs.append(data_job)
    for n, mode in FINETUNE:
        name = f"ft_{mode}_{n}"
        extra: dict[str, str | int | float] = {"finetune": mode, "max_train_episodes": n, "steps": 1500 if mode == "scratch" else 600}
        extra["lr"] = 1e-3 if mode == "lora" else 3e-4 if mode == "scratch" else 1e-4
        jobs += [train(name, "ft_unseen_color", "finetune", deps=[data_job.name, "train:act_main"], **extra), export(name, "finetune")]
        jobs.append(bench(name, name, "unseen_color,in_distribution", study_episodes, "finetune"))
    return jobs


FINETUNE = [(10, "lora"), (10, "full"), (25, "lora"), (25, "full"), (25, "scratch")]


def run_queue(jobs: list[Job], workers: int, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    done = {j.name for j in jobs if j.marker.exists()}
    known = {j.name for j in jobs}
    pending = [j for j in jobs if j.name not in done]
    running: dict[str, tuple[subprocess.Popen[bytes], float]] = {}
    failed: set[str] = set()
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1", OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 2) // workers)))
    while pending or running:
        for name, (p, t0) in list(running.items()):
            if p.poll() is not None:
                del running[name]
                ok = p.returncode == 0
                (done if ok else failed).add(name)
                print(f"[studies] {'done' if ok else 'FAILED'} {name} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        for j in list(pending):
            if any(d in failed for d in j.deps):
                pending.remove(j)
                failed.add(j.name)
                print(f"[studies] skip {j.name}: dependency failed", flush=True)
                continue
            if len(running) >= workers or not all(d in done or d not in known for d in j.deps):
                continue
            pending.remove(j)
            log = open(log_dir / f"{j.name.replace(':', '_')}.log", "wb")
            print(f"[studies] start {j.name}", flush=True)
            running[j.name] = (subprocess.Popen(j.cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), time.time())
        time.sleep(2)
    if failed:
        print(f"[studies] failed: {sorted(failed)}", flush=True)


def _bench(rid: str) -> dict | None:
    p = ROOT / "reports" / "bench" / rid / "benchmark.json"
    return json.loads(p.read_text()) if p.exists() else None


def _cell(rid: str, split: str) -> str:
    b = _bench(rid)
    if not b or split not in b["splits"]:
        return "–"
    s = b["splits"][split]
    return f"{100 * s['success_rate']:.0f}% ({100 * s['ci95'][0]:.0f}–{100 * s['ci95'][1]:.0f})"


def _probe(name: str) -> dict:
    p = ROOT / "reports" / "probes" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _run(name: str) -> dict:
    p = ROOT / "runs" / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_report() -> Path:
    L = [
        "# Studies",
        "",
        "Closed-loop success in simulation on held-out benchmark seeds (the same seeds for every policy), with 95% Wilson",
        "intervals. With 20-30 episodes per cell an interval spans roughly 30-40 points, so only differences outside",
        "overlapping intervals are evidence. Everything trains on 2 CPU cores; step budgets are small by robot-learning",
        "standards, which is the point of studies 1 and 4: what works on a small budget.",
        "",
    ]
    splits = ALL_SPLITS.split(",")
    r = _run("act_main")
    L += [
        f"## 1. Main policy ({r.get('steps', '?')} steps, {r.get('train_episodes', '?')} training episodes)",
        "",
        "| split | success (95% CI) | top failure |",
        "|---|---|---|",
    ]
    b = _bench("act_main")
    for sp in splits:
        top = "–"
        if b and sp in b["splits"] and b["splits"][sp]["failures"]:
            top = max(b["splits"][sp]["failures"].items(), key=lambda kv: kv[1])[0]
        L.append(f"| {sp} | {_cell('act_main', sp)} | {top} |")
    L += ["", "Failure analysis and videos: `reports/bench/act_main/REPORT.md`.", ""]
    L += [
        "## 2. Data source (same recipe, 1500 steps)",
        "",
        "| training data | episodes | in_distribution | lighting_shift | camera_shift |",
        "|---|---|---|---|---|",
    ]
    for run, label in (("src_teleop", "ROS 2 teleop only"), ("src_synthetic", "synthetic (domain-randomised) only"), ("pilot_film", "both")):
        rr = _run(run)
        rid = f"{run}_sources"
        L.append(
            f"| {label} | {rr.get('train_episodes', '–')} | {_cell(rid, 'in_distribution')} | {_cell(rid, 'lighting_shift')} | {_cell(rid, 'camera_shift')} |"
        )
    L += ["", "## 3. Policy class (same encoder, recipe and data, 1500 steps)", "", "| policy | in_distribution | position_ood |", "|---|---|---|"]
    for run, label in (
        ("bc_single_step", "single-step behaviour cloning (chunk 1)"),
        ("pilot_film", "action chunking (chunk 20, temporal ensembling)"),
        ("diffusion_chunk", "diffusion head (chunk 20, 10-step DDIM, temporal ensembling)"),
    ):
        L.append(f"| {label} | {_cell(run + '_class', 'in_distribution')} | {_cell(run + '_class', 'position_ood')} |")
    L += [
        "",
        "## 4. Ablation: making closed-loop behaviour cloning work (in_distribution)",
        "",
        "Each row changes the recipe of the row it names; all use all training data.",
        "",
        "| run | change | val loss | first-frame grounding | success (95% CI) | top failure |",
        "|---|---|---|---|---|---|",
    ]
    for run, what in ABLATION:
        bb = _bench(f"abl_{run}")
        top = "–"
        if bb and bb["splits"]["in_distribution"]["failures"]:
            top = max(bb["splits"]["in_distribution"]["failures"].items(), key=lambda kv: kv[1])[0]
        vl = _run(run).get("best_val_loss")
        pr = _probe(run)
        g = pr.get("val", {}).get("first_frame_target_within_30mm") if pr else None
        gtxt = f"{100 * g:.0f}%" if isinstance(g, float) and g == g else "–"
        L.append(
            f"| `{run}` | {what} | {vl:.3f} | {gtxt} | {_cell('abl_' + run, 'in_distribution')} | {top} |" if vl else f"| `{run}` | {what} | – | – | – | – |"
        )
    pm = _probe("act_main").get("val", {}).get("first_frame_target_within_30mm")
    bm = _bench("act_main")
    if bm:
        top = max(bm["splits"]["in_distribution"]["failures"].items(), key=lambda kv: kv[1])[0] if bm["splits"]["in_distribution"]["failures"] else "–"
        vl = _run("act_main").get("best_val_loss", float("nan"))
        L.append(
            f"| `act_main` | final recipe, {_run('act_main').get('steps')} steps (30 episodes) | {vl:.3f} | {f'{100 * pm:.0f}%' if pm is not None else '–'} | {_cell('act_main', 'in_distribution')} | {top} |"
        )
    L += [
        "",
        "Validation loss includes the auxiliary term for runs that have it, so it is only comparable within the first and",
        "second halves of the table.",
        "",
        "First-frame grounding: share of validation episodes where, before the arm has moved, the auxiliary head puts the",
        "instructed cube within 30 mm (`scripts/probe_policy.py`; only for runs with the auxiliary head). Validation loss does",
        "not predict closed-loop success here; grounding from the first frame does. See `docs/training.md`.",
        "",
        "## 5. Transfer under deployment conditions (act_main, in_distribution)",
        "",
        "| condition | success (95% CI) |",
        "|---|---|",
    ]
    for rid, label in (
        ("act_main", "as trained"),
        ("act_main_obs_latency_2", "observations 2 ticks old (133 ms)"),
        ("act_main_obs_latency_4", "observations 4 ticks old (267 ms)"),
        ("act_main_action_delay_2", "actions applied 2 ticks late"),
    ):
        L.append(f"| {label} | {_cell(rid, 'in_distribution')} |")
    L += [
        "",
        "## 6. Fine-tuning to unseen object colours",
        "",
        "Starting point: `act_main`, which never saw purple or orange cubes, scores " + _cell("act_main", "unseen_color") + " on unseen_color.",
        "",
        "| demos | method | trainable params | unseen_color | in_distribution (forgetting check) |",
        "|---|---|---|---|---|",
    ]
    for n, mode in FINETUNE:
        name = f"ft_{mode}_{n}"
        rr = _run(name)
        tp = rr.get("params", {}).get("trainable")
        L.append(f"| {n} | {mode} | {f'{tp / 1e6:.2f} M' if tp else '–'} | {_cell(name, 'unseen_color')} | {_cell(name, 'in_distribution')} |")
    out = ROOT / "reports" / "studies" / "STUDIES.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=30, help="per split, main policy")
    ap.add_argument("--study-episodes", type=int, default=20, help="per split, comparison studies")
    ap.add_argument("--only", help="comma-separated studies: main,data_source,policy_class,ablation,latency,finetune")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        jobs = plan(a.episodes, a.study_episodes)
        if a.only:
            want = set(a.only.split(","))
            jobs = [j for j in jobs if j.study in want]
        run_queue(jobs, a.workers, ROOT / "reports" / "studies" / "logs")
    print(write_report())


if __name__ == "__main__":
    main()
