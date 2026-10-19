#!/usr/bin/env python3
"""Launch one frozen evaluation job using relocated input/output paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = "363881d5cde7e39a0b218fea44936324382e55de"
PLANS = {
    "clean": (
        "corrected_full_evaluation_plan_s0_s2.json",
        "f933939f02f8fe3db0a685d8db8237ecc4e6f49d7bc8ac0912d13dba492e9d9e",
    ),
    "impairments": (
        "corrected_impairment_plan_s0_s2.json",
        "af2dacf72fce1a31b339016a78c4299b23f079d6e24e035055772588b8e73556",
    ),
}

def require(condition, message):
    if not condition:
        raise SystemExit("STOP: " + message)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--suite", choices=PLANS, required=True)
parser.add_argument("--job", required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--training-root", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()

release = Path(__file__).resolve().parents[2]
name, expected_hash = PLANS[args.suite]
plan_path = release / "tools/corrected_normalization/plans" / name
require(digest(plan_path) == expected_hash, "Frozen plan hash mismatch")
plan = json.loads(plan_path.read_text())
require(plan["git_commit"] == COMMIT, "Unexpected planned source commit")

matches = [j for j in plan["jobs"] if j["job_id"] == args.job]
require(len(matches) == 1, "Job ID must identify exactly one planned job")
job = matches[0]
require(Path(args.job).name == args.job, "Invalid job ID")
require(job["evaluation_seed"] == 0, "Evaluator fixes evaluation seed to zero")
require(job["configured_steps"] == 1200, "Evaluator fixes steps to 1200")
require(job["parallel_simulations"] == 32, "Expected 32 simulations")

repo = args.source.resolve()
training_root = args.training_root.resolve()
train = training_root / f"seed{job['training_seed']}"
output_root = args.output_root.resolve()
run = output_root / args.job

def git(*values):
    return subprocess.check_output(
        ["git", "-C", str(repo), *values], text=True
    ).strip()

require(git("rev-parse", "HEAD") == COMMIT, "Wrong experimental commit")
require(
    git("branch", "--show-current") == plan["git_branch"],
    "Wrong experimental branch",
)
require(not git("diff", "HEAD", "--"), "Tracked experimental source changed")
require(
    (repo / "evaluate_proposed_certificate.py").is_file(),
    "Evaluator missing",
)

protected = [repo, release, training_root]
for suffix, expected in job["checkpoint_sha256"].items():
    checkpoint = train / "model" / f"{job['checkpoint_prefix']}_{suffix}"
    require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
    require(digest(checkpoint) == expected, f"Hash mismatch: {checkpoint}")
    protected.append(checkpoint.resolve().parent)

for directory in protected:
    require(
        run != directory and directory not in run.parents,
        f"Output must be outside input/source directory: {directory}",
    )
require(not run.exists(), f"Output already exists: {run}")

def optional(value):
    return "none" if value is None else str(value)

settings = {
    "PAPER2_PROPOSED_RUN": str(run),
    "PAPER2_TRAIN_RUN": str(train),
    "PAPER2_CHECKPOINT_PREFIX": job["checkpoint_prefix"],
    "PAPER2_EVAL_SCENARIO": job["scenario"],
    "PAPER2_NUM_AGENTS": str(job["agents"]),
    "PAPER2_NUM_SIMULATIONS": str(job["parallel_simulations"]),
    "PAPER2_COMMUNICATION_MODE": job["communication_mode"],
    "PAPER2_COMMUNICATION_PERIOD_STEPS": optional(job["communication_period_steps"]),
    "PAPER2_COMMUNICATION_MAX_AGE_STEPS": optional(job["communication_max_age_steps"]),
    "PAPER2_COMMUNICATION_DELAY_STEPS": str(job.get("communication_delay_steps", 0)),
    "PAPER2_COMMUNICATION_LOSS_PROBABILITY": str(job.get("communication_loss_probability", 0.0)),
    "PAPER2_COMMUNICATION_OUTAGE_WINDOWS": json.dumps(job.get("communication_outage_windows", [])),
    "PAPER2_COMMUNICATION_RANDOM_SEED": str(job.get("communication_random_seed", 0)),
    "PYTHONUNBUFFERED": "1",
}

print("PASS: frozen plan, experimental source, and checkpoint hashes")
print(json.dumps(settings, indent=2))

if not args.execute:
    print("DRY RUN: no evaluation started and no output directory created.")
    raise SystemExit(0)

env = {
    key: value for key, value in os.environ.items()
    if not key.startswith("PAPER2_") and key != "PYTHONPATH"
}
env.update(settings)
run.mkdir(parents=True, exist_ok=False)
(run / "reproduction_invocation.json").write_text(json.dumps({
    "plan_sha256": expected_hash,
    "experimental_commit": COMMIT,
    "job": job,
    "relocated_environment": settings,
    "python_executable": sys.executable,
    "launcher_sha256": digest(Path(__file__).resolve()),
}, indent=2) + "\n")

with (run / "evaluation.log").open("w") as log:
    result = subprocess.run(
        [sys.executable, "-u", str(repo / "evaluate_proposed_certificate.py")],
        cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
(run / "runner_exit_status.txt").write_text(str(result.returncode) + "\n")
require(result.returncode == 0, f"Evaluation failed; inspect {run / 'evaluation.log'}")
manifest = json.loads((run / "manifest.json").read_text())
require(manifest["status"] == "completed", "Evaluator did not report completion")
print(f"Evaluation completed: {run}")
print("Numerical comparison with archived results is a separate validation step.")
