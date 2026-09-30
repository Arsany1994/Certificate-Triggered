
#!/usr/bin/env python3

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/home/arsany/paper2/corrected_normalization")
OUTPUT_ROOT = (
    WORKSPACE
    / "runs"
    / "evaluation"
    / "frozen_policy_impact_s0_s2"
)
RUNS_ROOT = OUTPUT_ROOT / "runs"

NUM_SIMULATIONS = 2

TRAINING_RUNS = {
    0: {
        "path": Path(
            "/home/arsany/paper2/runs/"
            "proposed_certificate_train_s0_T6MifT"
        ),
        "checkpoint_prefix": "reward3.53",
    },
    1: {
        "path": Path(
            "/home/arsany/paper2/runs/"
            "proposed_certificate_train_s1_qZ8sk6"
        ),
        "checkpoint_prefix": "reward3.45",
    },
    2: {
        "path": Path(
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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def job_key(seed, scenario):
    return f"seed{seed}_{scenario}"


def checkpoint_for(seed):
    item = TRAINING_RUNS[seed]
    checkpoint = (
        item["path"]
        / "model"
        / f"{item['checkpoint_prefix']}_policy.pth"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    return checkpoint


def load_json(path):
    return json.loads(path.read_text())


def locate_statistics(run_dir):
    candidates = (
        run_dir / "communication_statistics.json",
        run_dir / "model" / "communication_statistics.json",
    )

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"No communication statistics found in {run_dir}"
    )


def parse_completed_row(seed, scenario, run_dir):
    manifest = load_json(run_dir / "manifest.json")
    stats = load_json(locate_statistics(run_dir))
    summary = load_json(run_dir / "summary_untrimmed.json")

    if manifest.get("status") != "completed":
        raise RuntimeError(
            f"{run_dir}: manifest status is "
            f"{manifest.get('status')}"
        )

    if manifest.get("git_branch") != "corrected-normalization":
        raise RuntimeError(
            f"{run_dir}: wrong branch "
            f"{manifest.get('git_branch')}"
        )

    metrics = summary["metrics"]

    return {
        "seed": seed,
        "scenario": scenario,
        "n_agents": SCENARIOS[scenario],
        "mode": "certificate",
        "num_simulations": NUM_SIMULATIONS,
        "policy_sha256": manifest["source_sha256"].get(
            f"{TRAINING_RUNS[seed]['checkpoint_prefix']}_policy.pth",
            "",
        ),
        "git_branch": manifest["git_branch"],
        "git_commit": manifest["git_commit"],
        "messages_sent": stats["messages_sent"],
        "messages_received": stats["messages_received"],
        "bytes_sent": stats["bytes_sent"],
        "bytes_received": stats["bytes_received"],
        "certificate_refresh_requests": stats[
            "certificate_refresh_requests"
        ],
        "certificate_evaluations": stats[
            "certificate_evaluations"
        ],
        "mean_valid_information_age": stats[
            "mean_valid_information_age"
        ],
        "collision_union_frames_pct": metrics[
            "collision_union_frames_pct"
        ]["mean"],
        "mean_speed_m_s": metrics[
            "mean_speed_m_s"
        ]["mean"],
        "mean_centerline_deviation_cm": metrics[
            "mean_centerline_deviation_cm"
        ]["mean"],
        "run_dir": str(run_dir),
    }


def write_csv(rows):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed",
        "scenario",
        "n_agents",
        "mode",
        "num_simulations",
        "policy_sha256",
        "git_branch",
        "git_commit",
        "messages_sent",
        "messages_received",
        "bytes_sent",
        "bytes_received",
        "certificate_refresh_requests",
        "certificate_evaluations",
        "mean_valid_information_age",
        "collision_union_frames_pct",
        "mean_speed_m_s",
        "mean_centerline_deviation_cm",
        "run_dir",
    ]

    path = OUTPUT_ROOT / "run_level_results.csv"

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in sorted(
            rows.values(),
            key=lambda item: (
                int(item["seed"]),
                item["scenario"],
            ),
        ):
            writer.writerow(row)

    return path


def write_metadata(rows, failures):
    metadata = {
        "status": "completed"
        if len(rows) == 12 and not failures
        else "incomplete",
        "repository": str(REPO),
        "branch_required": "corrected-normalization",
        "run_type": "frozen-policy impact study",
        "training_performed": False,
        "communication_mode": "certificate",
        "channel": "clean",
        "num_simulations": NUM_SIMULATIONS,
        "expected_jobs": 12,
        "completed_jobs": len(rows),
        "failed_jobs": failures,
        "training_checkpoints": {
            str(seed): {
                "training_run": str(item["path"]),
                "checkpoint_prefix": item["checkpoint_prefix"],
                "checkpoint_sha256": sha256(
                    checkpoint_for(seed)
                ),
            }
            for seed, item in TRAINING_RUNS.items()
        },
        "scenarios": SCENARIOS,
        "rows": sorted(rows),
    }

    path = OUTPUT_ROOT / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    return path


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    rows = {}
    failures = []

    for seed in sorted(TRAINING_RUNS):
        checkpoint = checkpoint_for(seed)
        checkpoint_hash = sha256(checkpoint)

        for scenario, n_agents in SCENARIOS.items():
            key = job_key(seed, scenario)
            run_dir = RUNS_ROOT / key

            if (
                (run_dir / "communication_statistics.json").is_file()
                and (run_dir / "manifest.json").is_file()
                and (run_dir / "summary_untrimmed.json").is_file()
            ):
                try:
                    rows[key] = parse_completed_row(
                        seed,
                        scenario,
                        run_dir,
                    )
                    print(f"SKIP {key}: already completed", flush=True)
                    continue
                except Exception as exc:
                    archive = run_dir.with_name(
                        run_dir.name
                        + f".incomplete_{int(time.time())}"
                    )
                    run_dir.rename(archive)
                    print(
                        f"ARCHIVED incomplete {run_dir} -> {archive}",
                        flush=True,
                    )

            run_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.pop("PAPER2_COMMUNICATION_PERIOD_STEPS", None)
            env.pop("PAPER2_COMMUNICATION_MAX_AGE_STEPS", None)

            env.update(
                {
                    "PAPER2_PROPOSED_RUN": str(run_dir),
                    "PAPER2_TRAIN_RUN": str(
                        TRAINING_RUNS[seed]["path"]
                    ),
                    "PAPER2_CHECKPOINT_PREFIX": (
                        TRAINING_RUNS[seed]["checkpoint_prefix"]
                    ),
                    "PAPER2_EVAL_SCENARIO": scenario,
                    "PAPER2_NUM_AGENTS": str(n_agents),
                    "PAPER2_NUM_SIMULATIONS": str(
                        NUM_SIMULATIONS
                    ),
                    "PAPER2_COMMUNICATION_MODE": "certificate",
                    "PAPER2_COMMUNICATION_DELAY_STEPS": "0",
                    "PAPER2_COMMUNICATION_LOSS_PROBABILITY": "0",
                    "PAPER2_COMMUNICATION_OUTAGE_WINDOWS": "[]",
                    "PAPER2_COMMUNICATION_RANDOM_SEED": "0",
                    "PYTHONPATH": str(REPO),
                }
            )

            print(
                f"RUNNING {key}: "
                f"agents={n_agents}, "
                f"checkpoint_sha256={checkpoint_hash}",
                flush=True,
            )

            log_path = run_dir / "evaluation.log"

            with log_path.open("w") as log:
                result = subprocess.run(
                    [sys.executable, "evaluate_proposed_certificate.py"],
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            if result.returncode != 0:
                failures.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "returncode": result.returncode,
                        "run_dir": str(run_dir),
                    }
                )
                print(
                    f"FAIL {key}: returncode={result.returncode}",
                    flush=True,
                )
                continue

            try:
                rows[key] = parse_completed_row(
                    seed,
                    scenario,
                    run_dir,
                )
                print(
                    f"PASS {key}: "
                    f"messages={rows[key]['messages_sent']}, "
                    f"collision="
                    f"{rows[key]['collision_union_frames_pct']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "error": repr(exc),
                        "run_dir": str(run_dir),
                    }
                )
                print(f"FAIL {key}: {exc}", flush=True)

            write_csv(rows)

    csv_path = write_csv(rows)
    metadata_path = write_metadata(rows, failures)

    print()
    print(f"COMPLETED JOBS: {len(rows)}")
    print(f"FAILED JOBS: {len(failures)}")
    print(f"RUN-LEVEL CSV: {csv_path}")
    print(f"METADATA: {metadata_path}")

    if failures:
        sys.exit(1)

    if len(rows) != 12:
        raise SystemExit(
            f"Expected 12 completed jobs, found {len(rows)}"
        )

    print("PASS: all 12 frozen-policy impact jobs completed")


if __name__ == "__main__":
    main()
