#!/usr/bin/env python3
"""Resume-safe runner for the 48-job impairment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/home/arsany/paper2/runs/proposed_impairment_matrix_s0_s2_full"
)

TRAINING = {
    0: {
        "train_run": Path(
            "/home/arsany/paper2/runs/"
            "proposed_certificate_train_s0_T6MifT"
        ),
        "checkpoint_prefix": "reward3.53",
    },
    1: {
        "train_run": Path(
            "/home/arsany/paper2/runs/"
            "proposed_certificate_train_s1_qZ8sk6"
        ),
        "checkpoint_prefix": "reward3.45",
    },
    2: {
        "train_run": Path(
            "/home/arsany/paper2/runs/"
            "proposed_certificate_train_s2_USyTWQ"
        ),
        "checkpoint_prefix": "reward3.84",
    },
}

SCENARIOS = {
    "CPM_mixed": 4,
    "intersection_2": 6,
    "on_ramp_1": 8,
    "roundabout_1": 8,
}

CONDITIONS = {
    "certificate_clean": {
        "delay_steps": 0,
        "loss_probability": 0.0,
        "outage_windows": [],
    },
    "delay2": {
        "delay_steps": 2,
        "loss_probability": 0.0,
        "outage_windows": [],
    },
    "loss10": {
        "delay_steps": 0,
        "loss_probability": 0.10,
        "outage_windows": [],
    },
    "outage": {
        "delay_steps": 0,
        "loss_probability": 0.0,
        "outage_windows": [[300, 340], [800, 840]],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_for(seed: int) -> Path:
    item = TRAINING[seed]
    return (
        item["train_run"]
        / "model"
        / f'{item["checkpoint_prefix"]}_policy.pth'
    )


def make_jobs():
    jobs = []
    for seed, training in TRAINING.items():
        checkpoint = checkpoint_for(seed)
        policy_hash = sha256(checkpoint)

        for scenario, agents in SCENARIOS.items():
            for condition, impairment in CONDITIONS.items():
                jobs.append(
                    {
                        "job_id": f"{scenario}_{condition}_seed{seed}",
                        "seed": seed,
                        "scenario": scenario,
                        "agents": agents,
                        "condition": condition,
                        "train_run": str(training["train_run"]),
                        "checkpoint_prefix": training["checkpoint_prefix"],
                        "checkpoint": str(checkpoint),
                        "policy_sha256": policy_hash,
                        **impairment,
                    }
                )
    return jobs


def is_completed(job_dir: Path) -> bool:
    manifest_path = job_dir / "manifest.json"
    stats_path = job_dir / "communication_statistics.json"
    summary_path = job_dir / "summary_untrimmed.json"

    if not (
        manifest_path.is_file()
        and stats_path.is_file()
        and summary_path.is_file()
    ):
        return False

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False

    return manifest.get("status") == "completed"


def archive_incomplete(job_dir: Path) -> None:
    if not job_dir.exists():
        return

    suffix = int(time.time())
    archive = job_dir.with_name(
        f"{job_dir.name}.incomplete_{suffix}"
    )
    shutil.move(str(job_dir), str(archive))
    print(f"ARCHIVED incomplete directory: {archive}", flush=True)


def run_one(job, root: Path, simulations: int, random_seed: int):
    job_dir = root / job["job_id"]

    if is_completed(job_dir):
        print(f"SKIP completed {job['job_id']}", flush=True)
        return

    archive_incomplete(job_dir)
    job_dir.mkdir(parents=True, exist_ok=False)

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    environment.update(
        {
            "PAPER2_PROPOSED_RUN": str(job_dir),
            "PAPER2_TRAIN_RUN": job["train_run"],
            "PAPER2_CHECKPOINT_PREFIX": job["checkpoint_prefix"],
            "PAPER2_EVAL_SCENARIO": job["scenario"],
            "PAPER2_NUM_AGENTS": str(job["agents"]),
            "PAPER2_NUM_SIMULATIONS": str(simulations),
            "PAPER2_COMMUNICATION_MODE": "certificate",
            "PAPER2_COMMUNICATION_DELAY_STEPS": str(
                job["delay_steps"]
            ),
            "PAPER2_COMMUNICATION_LOSS_PROBABILITY": str(
                job["loss_probability"]
            ),
            "PAPER2_COMMUNICATION_OUTAGE_WINDOWS": json.dumps(
                job["outage_windows"]
            ),
            "PAPER2_COMMUNICATION_RANDOM_SEED": str(random_seed),
        }
    )

    log_path = job_dir / "evaluation.log"

    print(
        f"RUNNING {job['job_id']}: "
        f"agents={job['agents']}, "
        f"delay={job['delay_steps']}, "
        f"loss={job['loss_probability']}, "
        f"outages={job['outage_windows']}",
        flush=True,
    )

    with log_path.open("w") as log:
        completed = subprocess.run(
            [sys.executable, "evaluate_proposed_certificate.py"],
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{job['job_id']} failed with exit status "
            f"{completed.returncode}; inspect {log_path}"
        )

    stats_path = job_dir / "communication_statistics.json"
    summary_path = job_dir / "summary_untrimmed.json"
    manifest_path = job_dir / "manifest.json"

    for required in (stats_path, summary_path, manifest_path):
        if not required.is_file():
            raise RuntimeError(
                f"{job['job_id']} is missing {required}"
            )

    job_record = dict(job)
    job_record["status"] = "completed"
    job_record["run_directory"] = str(job_dir)
    (job_dir / "job_config.json").write_text(
        json.dumps(job_record, indent=2, sort_keys=True)
    )

    stats = json.loads(stats_path.read_text())
    print(
        f"PASS {job['job_id']}: "
        f"messages={stats['messages_sent']}, "
        f"received={stats['messages_received']}",
        flush=True,
    )


def write_plan(root: Path, jobs, simulations: int, random_seed: int):
    plan_path = root.parent / "proposed_impairment_matrix_plan_s0_s2.json"

    if plan_path.exists():
        return plan_path

    plan = {
        "status": "planned",
        "repository": str(REPO),
        "expected_jobs": len(jobs),
        "num_simulations": simulations,
        "communication_mode": "certificate",
        "communication_random_seed": random_seed,
        "scenarios": SCENARIOS,
        "conditions": CONDITIONS,
        "jobs": jobs,
    }

    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--communication-random-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    args = parser.parse_args()

    if args.num_simulations <= 0:
        raise ValueError("--num-simulations must be positive")

    if args.communication_random_seed < 0:
        raise ValueError(
            "--communication-random-seed must be non-negative"
        )

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs()

    for job in jobs:
        checkpoint = Path(job["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    plan_path = write_plan(
        root,
        jobs,
        args.num_simulations,
        args.communication_random_seed,
    )

    print(f"PLAN: {plan_path}", flush=True)
    print(f"EXPECTED JOBS: {len(jobs)}", flush=True)

    failures = []

    for job in jobs:
        try:
            run_one(
                job,
                root,
                args.num_simulations,
                args.communication_random_seed,
            )
        except Exception as exc:
            failures.append((job["job_id"], repr(exc)))
            print(f"FAIL {job['job_id']}: {exc}", flush=True)

            if not args.continue_on_error:
                raise

    if failures:
        print("FAILED JOBS:", flush=True)
        for job_id, error in failures:
            print(f"{job_id}: {error}", flush=True)
        raise SystemExit(1)

    aggregator = REPO / "tools" / "aggregate_impairment_matrix.py"
    subprocess.run(
        [
            sys.executable,
            str(aggregator),
            "--root",
            str(root),
            "--output-dir",
            str(root),
        ],
        cwd=REPO,
        check=True,
    )

    print("PASS: all impairment jobs completed", flush=True)
    print(f"RUN ROOT: {root}", flush=True)


if __name__ == "__main__":
    main()
