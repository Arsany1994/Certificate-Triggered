#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

WORKSPACE = Path("/home/arsany/paper2/corrected_normalization")
PLAN_PATH = (
    WORKSPACE / "manifests/corrected_full_evaluation_plan_s0_s2.json"
)
EXPECTED_PLAN_SHA = (
    "f933939f02f8fe3db0a685d8db8237ecc4e6f49d7bc8ac0912d13dba492e9d9e"
)
MIN_FREE_BYTES = 2 * 1024**3

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)

require(sha256(PLAN_PATH) == EXPECTED_PLAN_SHA, "Frozen plan hash mismatch")
plan = read_json(PLAN_PATH)
repo = Path(plan["repo"])
root = Path(plan["output_root"])
root.mkdir(parents=True, exist_ok=True)

def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()

def check_source():
    require(
        git("branch", "--show-current") == plan["git_branch"],
        "Wrong source branch",
    )
    require(
        git("rev-parse", "HEAD") == plan["git_commit"],
        "Source commit differs from the frozen plan",
    )
    require(not git("diff", "HEAD", "--"), "Tracked source changes detected")

def check_checkpoint(job):
    model = Path(job["train_run"]) / "model"
    for suffix, expected in job["checkpoint_sha256"].items():
        path = model / f"{job['checkpoint_prefix']}_{suffix}"
        require(path.is_file(), f"Missing checkpoint: {path}")
        require(sha256(path) == expected, f"Checkpoint hash mismatch: {path}")

metric_names = (
    "collision_union_frames_pct",
    "mean_speed_m_s",
    "mean_centerline_deviation_cm",
)

def validate_run(job, run):
    manifest = read_json(run / "manifest.json")
    parameters = read_json(run / "effective_evaluation_parameters.json")
    stats = read_json(run / "communication_statistics.json")
    summary = read_json(run / "summary_untrimmed.json")

    expected_manifest = {
        "status": "completed",
        "git_branch": plan["git_branch"],
        "git_commit": plan["git_commit"],
        "scenario": job["scenario"],
        "agents": job["agents"],
        "parallel_simulations": job["parallel_simulations"],
        "configured_steps": job["configured_steps"],
        "evaluation_seed": job["evaluation_seed"],
        "communication_mode": job["communication_mode"],
    }
    for key, value in expected_manifest.items():
        require(
            manifest.get(key) == value,
            f"{job['job_id']}: manifest mismatch for {key}",
        )

    require(
        Path(manifest["training_run"]).resolve()
        == Path(job["train_run"]).resolve(),
        "Training-run provenance mismatch",
    )
    require(
        manifest["checkpoint"] == f"{job['checkpoint_prefix']}_policy.pth",
        "Selected checkpoint mismatch",
    )
    for suffix, expected in job["checkpoint_sha256"].items():
        name = f"{job['checkpoint_prefix']}_{suffix}"
        require(
            manifest["source_sha256"].get(name) == expected,
            f"Recorded source hash mismatch: {name}",
        )
    policy = run / "model" / f"{job['checkpoint_prefix']}_policy.pth"
    require(
        sha256(policy) == job["checkpoint_sha256"]["policy.pth"],
        "Copied policy hash mismatch",
    )

    expected_parameters = {
        "scenario_type": job["scenario"],
        "n_agents": job["agents"],
        "num_vmas_envs": job["parallel_simulations"],
        "max_steps": job["configured_steps"],
        "communication_mode": job["communication_mode"],
        "communication_period_steps": job["communication_period_steps"],
        "communication_max_age_steps": job["communication_max_age_steps"],
        "communication_delay_steps": 0,
        "communication_loss_probability": 0.0,
        "communication_outage_windows": [],
        "communication_random_seed": 0,
        "is_continue_train": False,
        "is_testing_mode": True,
    }
    for key, value in expected_parameters.items():
        require(
            parameters.get(key) == value,
            f"Effective parameter mismatch: {key}",
        )

    require(stats["active"] is True, "Communication accounting inactive")
    require(stats["mode"] == job["communication_mode"], "Mode mismatch")
    require(stats["delay_steps"] == 0, "Unexpected delay")
    require(stats["loss_probability"] == 0.0, "Unexpected packet loss")
    require(stats["outage_windows"] == [], "Unexpected outage")
    require(stats["random_seed"] == 0, "Unexpected channel seed")
    require(stats["messages_sent"] > 0, "No messages recorded")
    require(
        stats["messages_sent"] == stats["messages_received"],
        "Clean-channel sent/received mismatch",
    )
    require(stats["payload_dim"] == 11, "Unexpected payload dimension")
    require(stats["payload_element_size_bytes"] == 4, "Unexpected element size")
    require(
        stats["bytes_sent"] == stats["messages_sent"] * 44,
        "Sent-byte accounting mismatch",
    )
    require(stats["bytes_received"] == stats["bytes_sent"], "Received-byte mismatch")
    require(
        stats["cache_active"] == (job["communication_mode"] != "perfect"),
        "Unexpected cache state",
    )
    if job["communication_mode"] == "certificate":
        require(stats["certificate_evaluations"] > 0, "No certificate evaluations")

    require(manifest["actual_steps"] == 1199, "Unexpected saved rollout length")
    require(
        manifest["observation_shape"]
        == [32, 1199, job["agents"], 32],
        "Unexpected observation shape",
    )
    require(
        manifest["action_shape"] == [32, 1199, job["agents"], 2],
        "Unexpected action shape",
    )

    rollouts = list((run / "model").glob("*_out_td_*.pth"))
    require(len(rollouts) == 1, "Expected exactly one rollout")
    require(
        rollouts[0].resolve() == Path(manifest["raw_rollout"]).resolve(),
        "Rollout path mismatch",
    )
    require(
        sha256(rollouts[0]) == manifest["raw_rollout_sha256"],
        "Rollout hash mismatch",
    )

    with (run / "per_simulation_metrics.csv").open(newline="") as stream:
        simulation_rows = list(csv.DictReader(stream))
    require(len(simulation_rows) == 32, "Expected 32 per-simulation rows")

    row = {
        "job_id": job["job_id"],
        "role": job["role"],
        "scenario": job["scenario"],
        "training_seed": job["training_seed"],
        "mode": job["communication_mode"],
        "period_steps": job["communication_period_steps"],
        "max_age_steps": job["communication_max_age_steps"],
        "simulations": 32,
        "run_directory": str(run),
        "policy_sha256": job["checkpoint_sha256"]["policy.pth"],
        "messages_sent": stats["messages_sent"],
        "bytes_sent": stats["bytes_sent"],
        "information_age_mean": stats["mean_valid_information_age"],
        "information_age_max": stats["max_valid_information_age"],
    }
    for name in metric_names:
        mean = float(summary["metrics"][name]["mean"])
        sd = float(summary["metrics"][name]["sd_across_simulations"])
        require(math.isfinite(mean) and math.isfinite(sd), f"Nonfinite metric: {name}")
        require(sd >= 0, f"Negative SD: {name}")
        row[name] = mean
        row[name + "_sd_across_simulations"] = sd
    return row

completed = []

def save_progress(status, error=None):
    if completed:
        path = root / "run_level_results.csv"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(completed[0]))
            writer.writeheader()
            writer.writerows(completed)
        temporary.replace(path)

    atomic_json(root / "execution_metadata.json", {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(PLAN_PATH),
        "plan_sha256": EXPECTED_PLAN_SHA,
        "git_commit": plan["git_commit"],
        "expected_jobs": len(plan["jobs"]),
        "validated_completed_jobs": len(completed),
        "completed_job_ids": [row["job_id"] for row in completed],
        "error": error,
    })

def run_evaluation(job, run):
    env = os.environ.copy()

    # Prevent inherited evaluation settings from leaking into this job.
    for key in list(env):
        if key.startswith("PAPER2_"):
            env.pop(key)
    env.pop("PYTHONPATH", None)

    settings = {
        "PAPER2_PROPOSED_RUN": str(run),
        "PAPER2_TRAIN_RUN": job["train_run"],
        "PAPER2_CHECKPOINT_PREFIX": job["checkpoint_prefix"],
        "PAPER2_EVAL_SCENARIO": job["scenario"],
        "PAPER2_NUM_AGENTS": str(job["agents"]),
        "PAPER2_NUM_SIMULATIONS": "32",
        "PAPER2_COMMUNICATION_MODE": job["communication_mode"],
        "PAPER2_COMMUNICATION_PERIOD_STEPS": (
            "none" if job["communication_period_steps"] is None
            else str(job["communication_period_steps"])
        ),
        "PAPER2_COMMUNICATION_MAX_AGE_STEPS": (
            "none" if job["communication_max_age_steps"] is None
            else str(job["communication_max_age_steps"])
        ),
        "PAPER2_COMMUNICATION_DELAY_STEPS": "0",
        "PAPER2_COMMUNICATION_LOSS_PROBABILITY": "0",
        "PAPER2_COMMUNICATION_OUTAGE_WINDOWS": "[]",
        "PAPER2_COMMUNICATION_RANDOM_SEED": "0",
        "PYTHONUNBUFFERED": "1",
    }
    env.update(settings)

    with (run / "evaluation.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(repo / "evaluate_proposed_certificate.py")],
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

    (run / "runner_exit_status.txt").write_text(str(returncode) + "\n")
    require(returncode == 0, f"Evaluator exited with status {returncode}")

try:
    check_source()
    require(len(plan["jobs"]) == 63, "Unexpected number of planned jobs")
    for job in plan["jobs"]:
        check_checkpoint(job)

    print("PASS: source and all planned checkpoints verified", flush=True)
    print("TOTAL JOBS: 63", flush=True)
    print("OUTPUT ROOT:", root, flush=True)

    for index, job in enumerate(plan["jobs"], start=1):
        check_source()
        run = root / job["job_id"]

        if run.exists():
            try:
                row = validate_run(job, run)
            except Exception as exc:
                print(f"INCOMPLETE/INVALID {job['job_id']}: {exc}", flush=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archived = run.with_name(run.name + ".incomplete_" + stamp)
                run.rename(archived)
                print("PRESERVED:", archived, flush=True)
            else:
                completed.append(row)
                save_progress("running")
                print(f"SKIP [{index}/63]: validated {job['job_id']}", flush=True)
                continue

        free = shutil.disk_usage(root).free
        require(
            free >= MIN_FREE_BYTES,
            f"Only {free / 1024**3:.2f} GiB free; free space and restart this runner",
        )

        run.mkdir()
        atomic_json(run / "planned_job.json", job)
        print(
            f"\nRUNNING [{index}/63] {job['job_id']} "
            f"role={job['role']} free_GiB={free / 1024**3:.2f}",
            flush=True,
        )
        run_evaluation(job, run)
        check_source()
        row = validate_run(job, run)
        completed.append(row)
        save_progress("running")
        print(
            f"PASS [{index}/63] {job['job_id']}: "
            f"messages={row['messages_sent']}, "
            f"collision={row['collision_union_frames_pct']:.4f}%",
            flush=True,
        )

    save_progress("completed")
    print("\nPASS: all 63 corrected evaluations completed and validated")
    print("RUN-LEVEL CSV:", root / "run_level_results.csv")
    print("METADATA:", root / "execution_metadata.json")

except KeyboardInterrupt:
    save_progress("interrupted", "Interrupted by user")
    print("\nStopped. Restart the same command to resume.", flush=True)
    sys.exit(130)
except Exception as exc:
    save_progress("failed", str(exc))
    print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
    print("Completed run directories are preserved.", flush=True)
    sys.exit(1)
