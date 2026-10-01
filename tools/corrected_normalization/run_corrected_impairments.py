#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path("/home/arsany/paper2/corrected_normalization")
PLAN_PATH = WORKSPACE / "manifests/corrected_impairment_plan_s0_s2.json"
EXPECTED_PLAN_SHA = (
    "af2dacf72fce1a31b339016a78c4299b23f079d6e24e035055772588b8e73556"
)
RESERVE = 2 * 1024**3

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
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

require(sha256(PLAN_PATH) == EXPECTED_PLAN_SHA, "Frozen plan hash mismatch")
plan = read_json(PLAN_PATH)
repo = Path(plan["repo"])
root = Path(plan["output_root"])
root.mkdir(parents=True, exist_ok=True)

require(len(plan["jobs"]) == 36, "Expected 36 new jobs")
require(len(plan["clean_references"]) == 12, "Expected 12 clean references")

def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()

def check_source():
    require(git("branch", "--show-current") == plan["git_branch"],
            "Source branch differs from plan")
    require(git("rev-parse", "HEAD") == plan["git_commit"],
            "Source commit differs from plan")
    require(not git("diff", "HEAD", "--"), "Tracked source changes detected")

def check_checkpoint(job):
    for suffix, expected in job["checkpoint_sha256"].items():
        path = (
            Path(job["train_run"]) / "model"
            / f"{job['checkpoint_prefix']}_{suffix}"
        )
        require(path.is_file(), f"Missing checkpoint: {path}")
        require(sha256(path) == expected, f"Checkpoint hash mismatch: {path}")

references = {
    (ref["scenario"], ref["training_seed"]): ref
    for ref in plan["clean_references"]
}
require(len(references) == 12, "Duplicate clean reference keys")

def verify_reference(ref):
    run = Path(ref["run_directory"])
    for filename, hash_key in (
        ("manifest.json", "manifest_sha256"),
        ("communication_statistics.json", "statistics_sha256"),
        ("summary_untrimmed.json", "summary_sha256"),
    ):
        require(sha256(run / filename) == ref[hash_key],
                f"Clean reference changed: {run / filename}")
    return read_json(run / "manifest.json")

metric_names = (
    "collision_union_frames_pct",
    "mean_speed_m_s",
    "mean_centerline_deviation_cm",
)

def result_row(run, scenario, seed, condition, policy_hash):
    stats = read_json(run / "communication_statistics.json")
    summary = read_json(run / "summary_untrimmed.json")
    sent = int(stats["messages_sent"])
    received = int(stats["messages_received"])
    dropped = int(stats["dropped_messages"])

    require(sent > 0, "No attempted messages recorded")
    require(0 <= received <= sent, "Invalid received-message count")
    require(0 <= dropped <= sent, "Invalid dropped-message count")
    require(received + dropped <= sent, "Message accounting exceeds attempts")
    require(stats["payload_dim"] == 11, "Unexpected payload dimension")
    require(stats["payload_element_size_bytes"] == 4, "Unexpected payload size")
    require(stats["bytes_sent"] == sent * 44, "Sent-byte mismatch")
    require(stats["bytes_received"] == received * 44, "Received-byte mismatch")

    row = {
        "scenario": scenario,
        "training_seed": seed,
        "condition": condition,
        "simulations": 32,
        "run_directory": str(run),
        "policy_sha256": policy_hash,
        "delay_steps": stats["delay_steps"],
        "loss_probability": stats["loss_probability"],
        "outage_windows": json.dumps(stats["outage_windows"]),
        "channel_random_seed": stats["random_seed"],
        "messages_sent": sent,
        "messages_received": received,
        "bytes_sent": stats["bytes_sent"],
        "bytes_received": stats["bytes_received"],
        "dropped_messages": dropped,
        "delayed_messages": stats["delayed_messages"],
        "outage_blocked_messages": stats["outage_blocked_messages"],
        "sent_minus_received_minus_dropped": sent - received - dropped,
        "delivery_pct": 100.0 * received / sent,
        "fallback_events": stats["fallback_events"],
        "eligible_slots": stats["eligible_slots"],
        "valid_slots": stats["valid_slots"],
        "information_age_mean": stats["mean_valid_information_age"],
        "information_age_max": stats["max_valid_information_age"],
    }
    for name in metric_names:
        for field, output in (
            ("mean", name),
            ("sd_across_simulations", name + "_sd_across_simulations"),
        ):
            value = float(summary["metrics"][name][field])
            require(math.isfinite(value) and value >= 0,
                    f"Invalid metric: {name}/{field}")
            row[output] = value
    return row

def validate_run(job, run):
    manifest = read_json(run / "manifest.json")
    parameters = read_json(run / "effective_evaluation_parameters.json")
    stats = read_json(run / "communication_statistics.json")

    expected_manifest = {
        "status": "completed",
        "git_branch": plan["git_branch"],
        "git_commit": plan["git_commit"],
        "scenario": job["scenario"],
        "agents": job["agents"],
        "parallel_simulations": 32,
        "configured_steps": 1200,
        "evaluation_seed": 0,
        "communication_mode": "certificate",
    }
    for key, value in expected_manifest.items():
        require(manifest.get(key) == value, f"Manifest mismatch: {key}")

    require(
        Path(manifest["training_run"]).resolve()
        == Path(job["train_run"]).resolve(),
        "Training-run provenance mismatch",
    )
    prefix = job["checkpoint_prefix"]
    require(manifest["checkpoint"] == f"{prefix}_policy.pth",
            "Checkpoint selection mismatch")
    for suffix, expected in job["checkpoint_sha256"].items():
        name = f"{prefix}_{suffix}"
        require(manifest["source_sha256"].get(name) == expected,
                f"Recorded checkpoint hash mismatch: {name}")
    require(
        sha256(run / "model" / f"{prefix}_policy.pth")
        == job["checkpoint_sha256"]["policy.pth"],
        "Copied policy hash mismatch",
    )

    expected_parameters = {
        "scenario_type": job["scenario"],
        "n_agents": job["agents"],
        "num_vmas_envs": 32,
        "max_steps": 1200,
        "communication_mode": "certificate",
        "communication_period_steps": None,
        "communication_max_age_steps": None,
        "is_continue_train": False,
        "is_testing_mode": True,
    }
    for key, value in expected_parameters.items():
        require(parameters.get(key) == value,
                f"Effective parameter mismatch: {key}")

    for parameter, statistic in (
        ("communication_delay_steps", "delay_steps"),
        ("communication_loss_probability", "loss_probability"),
        ("communication_outage_windows", "outage_windows"),
        ("communication_random_seed", "random_seed"),
    ):
        require(parameters.get(parameter) == job[parameter],
                f"Effective impairment mismatch: {parameter}")
        require(manifest.get(parameter) == job[parameter],
                f"Manifest impairment mismatch: {parameter}")
        require(stats.get(statistic) == job[parameter],
                f"Statistics impairment mismatch: {statistic}")

    require(stats["mode"] == "certificate", "Unexpected mode")
    require(stats["active"] is True, "Inactive communication accounting")
    require(stats["cache_active"] is True, "Inactive cache")
    require(stats["certificate_evaluations"] > 0, "No certificate evaluations")

    condition = job["condition"]
    if condition == "delay2":
        require(stats["delayed_messages"] > 0, "No delayed messages")
        require(stats["dropped_messages"] == 0, "Unexpected channel drops")
        require(stats["outage_blocked_messages"] == 0, "Unexpected outage")
        # In-flight messages and reset-discarded queued messages may
        # prevent sent and received totals from matching.
    elif condition == "loss10":
        require(stats["dropped_messages"] > 0, "No packet loss observed")
        require(stats["delayed_messages"] == 0, "Unexpected delay")
        require(stats["outage_blocked_messages"] == 0, "Unexpected outage")
        require(
            stats["messages_sent"]
            == stats["messages_received"] + stats["dropped_messages"],
            "Immediate packet-loss accounting mismatch",
        )
    elif condition == "outage":
        require(stats["outage_blocked_messages"] > 0, "No outage observed")
        require(stats["delayed_messages"] == 0, "Unexpected delay")
        require(
            stats["dropped_messages"] == stats["outage_blocked_messages"],
            "Outage/drop accounting mismatch",
        )
        require(
            stats["messages_sent"]
            == stats["messages_received"] + stats["dropped_messages"],
            "Immediate outage accounting mismatch",
        )
    else:
        raise RuntimeError(f"Unknown condition: {condition}")

    require(manifest["actual_steps"] == 1199, "Unexpected saved frame count")
    require(
        manifest["observation_shape"] == [32, 1199, job["agents"], 32],
        "Observation shape mismatch",
    )
    require(
        manifest["action_shape"] == [32, 1199, job["agents"], 2],
        "Action shape mismatch",
    )
    rollouts = list((run / "model").glob("*_out_td_*.pth"))
    require(len(rollouts) == 1, "Expected one saved rollout")
    require(
        rollouts[0].resolve() == Path(manifest["raw_rollout"]).resolve(),
        "Rollout path mismatch",
    )
    require(sha256(rollouts[0]) == manifest["raw_rollout_sha256"],
            "Rollout hash mismatch")
    with (run / "per_simulation_metrics.csv").open(newline="") as stream:
        require(len(list(csv.DictReader(stream))) == 32,
                "Expected 32 per-simulation metric rows")

    return result_row(
        run, job["scenario"], job["training_seed"],
        condition, job["checkpoint_sha256"]["policy.pth"],
    )

def run_evaluation(job, run):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("PAPER2_"):
            env.pop(key)
    env.pop("PYTHONPATH", None)
    env.update({
        "PAPER2_PROPOSED_RUN": str(run),
        "PAPER2_TRAIN_RUN": job["train_run"],
        "PAPER2_CHECKPOINT_PREFIX": job["checkpoint_prefix"],
        "PAPER2_EVAL_SCENARIO": job["scenario"],
        "PAPER2_NUM_AGENTS": str(job["agents"]),
        "PAPER2_NUM_SIMULATIONS": "32",
        "PAPER2_COMMUNICATION_MODE": "certificate",
        "PAPER2_COMMUNICATION_PERIOD_STEPS": "none",
        "PAPER2_COMMUNICATION_MAX_AGE_STEPS": "none",
        "PAPER2_COMMUNICATION_DELAY_STEPS":
            str(job["communication_delay_steps"]),
        "PAPER2_COMMUNICATION_LOSS_PROBABILITY":
            str(job["communication_loss_probability"]),
        "PAPER2_COMMUNICATION_OUTAGE_WINDOWS":
            json.dumps(job["communication_outage_windows"]),
        "PAPER2_COMMUNICATION_RANDOM_SEED":
            str(job["communication_random_seed"]),
        "PYTHONUNBUFFERED": "1",
    })

    with (run / "evaluation.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(repo / "evaluate_proposed_certificate.py")],
            cwd=str(repo), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

    (run / "runner_exit_status.txt").write_text(str(code) + "\n")
    require(code == 0, f"Evaluator exit status: {code}")

clean_rows = []
completed = []
completed_ids = []

def save_progress(status, error=None):
    write_csv(root / "run_level_results.csv", clean_rows + completed)
    atomic_json(root / "execution_metadata.json", {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(PLAN_PATH),
        "plan_sha256": EXPECTED_PLAN_SHA,
        "git_commit": plan["git_commit"],
        "expected_new_jobs": 36,
        "validated_new_jobs": len(completed),
        "validated_clean_references": len(clean_rows),
        "analysis_rows": len(clean_rows) + len(completed),
        "completed_new_job_ids": completed_ids,
        "error": error,
    })

try:
    check_source()

    for ref in plan["clean_references"]:
        manifest = verify_reference(ref)
        require(manifest["status"] == "completed", "Incomplete clean reference")
        clean_rows.append(result_row(
            Path(ref["run_directory"]), ref["scenario"],
            ref["training_seed"], "clean", ref["policy_sha256"],
        ))

    for job in plan["jobs"]:
        check_checkpoint(job)

    print("PASS: source, corrected checkpoints, and clean references verified")
    print("NEW JOBS: 36")
    print("REUSED CLEAN REFERENCES: 12")
    print("OUTPUT ROOT:", root, flush=True)
    save_progress("running")

    for index, job in enumerate(plan["jobs"], start=1):
        check_source()
        run = root / job["job_id"]

        if run.exists():
            try:
                row = validate_run(job, run)
            except Exception as exc:
                # Preserve completed-but-invalid outputs for diagnosis.
                manifest_path = run / "manifest.json"
                finished = False
                if manifest_path.exists():
                    try:
                        finished = read_json(manifest_path).get("status") == "completed"
                    except Exception:
                        pass
                require(
                    not finished,
                    f"Completed run failed validation: {run}: {exc}. "
                    "Inspect it before rerunning.",
                )
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archived = run.with_name(run.name + ".incomplete_" + stamp)
                run.rename(archived)
                print("PRESERVED INCOMPLETE RUN:", archived, flush=True)
            else:
                completed.append(row)
                completed_ids.append(job["job_id"])
                save_progress("running")
                print(f"SKIP [{index}/36]: validated {job['job_id']}", flush=True)
                continue

        ref = references[(job["scenario"], job["training_seed"])]
        reference_manifest = verify_reference(ref)
        rollout = Path(reference_manifest["raw_rollout"])
        require(rollout.is_file(), f"Missing clean reference rollout: {rollout}")

        # Allow twice the reference rollout size, plus a 2 GiB reserve.
        required_free = RESERVE + 2 * rollout.stat().st_size
        free = shutil.disk_usage(root).free
        require(
            free >= required_free,
            f"Insufficient free space for next job: {free / 1024**3:.2f} GiB "
            f"available; threshold {required_free / 1024**3:.2f} GiB. "
            "Free space and resume.",
        )

        run.mkdir()
        atomic_json(run / "planned_job.json", job)
        print(
            f"\nRUNNING [{index}/36] {job['job_id']} "
            f"free_GiB={free / 1024**3:.2f}",
            flush=True,
        )
        run_evaluation(job, run)
        check_source()
        row = validate_run(job, run)
        completed.append(row)
        completed_ids.append(job["job_id"])
        save_progress("running")
        print(
            f"PASS [{index}/36] {job['job_id']}: "
            f"sent={row['messages_sent']}, "
            f"received={row['messages_received']}, "
            f"collision={row['collision_union_frames_pct']:.4f}%",
            flush=True,
        )

    require(len(clean_rows) + len(completed) == 48, "Expected 48 analysis rows")
    save_progress("completed")
    print("\nPASS: all 36 corrected impairment evaluations completed")
    print("PASS: 48 analysis rows including 12 reused clean references")
    print("RUN-LEVEL CSV:", root / "run_level_results.csv")
    print("METADATA:", root / "execution_metadata.json")

except KeyboardInterrupt:
    save_progress("interrupted", "Interrupted by user")
    print("\nStopped. Restart the same command to resume.", flush=True)
    sys.exit(130)
except Exception as exc:
    save_progress("failed", str(exc))
    print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
    print("Existing run outputs have been preserved.", flush=True)
    sys.exit(1)
