#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

WORKSPACE = Path("/home/arsany/paper2/corrected_normalization")
ROOT = WORKSPACE / "runs/evaluation/corrected_full_comparison_s0_s2"
OUT = WORKSPACE / "runs/aggregates/corrected_full_comparison_s0_s2"
PLAN = WORKSPACE / "manifests/corrected_full_evaluation_plan_s0_s2.json"
EXPECTED_PLAN_SHA = (
    "f933939f02f8fe3db0a685d8db8237ecc4e6f49d7bc8ac0912d13dba492e9d9e"
)

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
    return json.loads(path.read_text())

def write_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def optional_int(value):
    return None if value in ("", None) else int(value)

require(sha256(PLAN) == EXPECTED_PLAN_SHA, "Frozen plan hash mismatch")
plan = read_json(PLAN)
execution = read_json(ROOT / "execution_metadata.json")

require(execution["status"] == "completed", "Execution is not completed")
require(execution["validated_completed_jobs"] == 63, "Expected 63 completed jobs")
require(execution["plan_sha256"] == EXPECTED_PLAN_SHA, "Execution plan mismatch")
require(execution["git_commit"] == plan["git_commit"], "Source commit mismatch")

with (ROOT / "run_level_results.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

jobs = {job["job_id"]: job for job in plan["jobs"]}
require(len(rows) == len(jobs) == 63, "Expected 63 unique planned jobs")
require(len({r["job_id"] for r in rows}) == 63, "Duplicate result job IDs")
require(
    {r["job_id"] for r in rows}
    == set(jobs)
    == set(execution["completed_job_ids"]),
    "Plan, execution, and CSV coverage differ",
)
require(
    Counter(r["role"] for r in rows)
    == {"primary": 48, "budget_bracket": 15},
    "Unexpected primary/additional run counts",
)

metrics = {
    "messages": "messages_sent",
    "payload_bytes": "bytes_sent",
    "collision_union_pct": "collision_union_frames_pct",
    "speed_m_s": "mean_speed_m_s",
    "deviation_cm": "mean_centerline_deviation_cm",
    "information_age_steps": "information_age_mean",
}

for row in rows:
    job = jobs[row["job_id"]]
    for field in ("role", "scenario"):
        require(row[field] == job[field], f"{row['job_id']}: {field} mismatch")
    require(row["mode"] == job["communication_mode"], "Mode mismatch")
    require(int(row["training_seed"]) == job["training_seed"], "Seed mismatch")
    require(int(row["simulations"]) == 32, "Expected 32 simulations")
    require(
        optional_int(row["period_steps"]) == job["communication_period_steps"],
        "Periodic schedule mismatch",
    )
    require(
        optional_int(row["max_age_steps"]) == job["communication_max_age_steps"],
        "Age schedule mismatch",
    )
    require(
        row["policy_sha256"] == job["checkpoint_sha256"]["policy.pth"],
        "Policy hash mismatch",
    )
    require(
        Path(row["run_directory"]).resolve() == (ROOT / row["job_id"]).resolve(),
        "Unexpected run directory",
    )

    for column in metrics.values():
        value = float(row[column])
        require(math.isfinite(value) and value >= 0, f"Invalid value: {column}")
    require(float(row["collision_union_frames_pct"]) <= 100, "Invalid collision %")
    require(
        int(row["bytes_sent"]) == 44 * int(row["messages_sent"]),
        "Payload-byte accounting mismatch",
    )

    for column in (
        "collision_union_frames_pct_sd_across_simulations",
        "mean_speed_m_s_sd_across_simulations",
        "mean_centerline_deviation_cm_sd_across_simulations",
    ):
        value = float(row[column])
        require(math.isfinite(value) and value >= 0, f"Invalid SD: {column}")

# Match each run to the certificate and perfect runs from the same
# scenario and training seed.
reference = {}
for row in rows:
    if row["mode"] in ("certificate", "perfect"):
        key = (row["scenario"], row["training_seed"], row["mode"])
        require(key not in reference, "Duplicate reference run")
        reference[key] = row

seed_comparisons = []
for row in rows:
    key = (row["scenario"], row["training_seed"])
    cert = reference[(*key, "certificate")]
    perfect = reference[(*key, "perfect")]
    messages = int(row["messages_sent"])
    cert_messages = int(cert["messages_sent"])
    perfect_messages = int(perfect["messages_sent"])
    require(cert_messages > 0 and perfect_messages > 0, "Zero reference count")

    difference = 100.0 * (messages / cert_messages - 1.0)
    reduction = 100.0 * (1.0 - messages / perfect_messages)
    seed_comparisons.append({
        **row,
        "message_difference_vs_certificate_pct": difference,
        "within_5pct_certificate_budget": abs(difference) <= 5.0,
        "message_reduction_vs_perfect_pct": reduction,
        "payload_byte_reduction_vs_perfect_pct": reduction,
    })

# Group by scenario, role, method AND schedule.
groups = defaultdict(list)
for row in seed_comparisons:
    key = (
        row["scenario"], row["role"], row["mode"],
        row["period_steps"], row["max_age_steps"],
    )
    groups[key].append(row)

require(len(groups) == 21, "Expected 21 distinct comparison groups")

aggregates = []
for key, group in groups.items():
    scenario, role, mode, period, age = key
    require(len(group) == 3, f"Expected three runs: {key}")
    require(
        {int(r["training_seed"]) for r in group} == {0, 1, 2},
        f"Incorrect training seeds: {key}",
    )
    result = {
        "scenario": scenario,
        "role": role,
        "mode": mode,
        "period_steps": period,
        "max_age_steps": age,
        "n_training_seeds": 3,
        "simulations_per_seed": 32,
    }

    for label, column in metrics.items():
        values = [float(r[column]) for r in group]
        result[label + "_mean"] = mean(values)
        result[label + "_sd_across_seeds"] = stdev(values)

    result["information_age_max_across_runs"] = max(
        int(r["information_age_max"]) for r in group
    )

    cert_mean = mean(
        int(reference[(scenario, str(seed), "certificate")]["messages_sent"])
        for seed in (0, 1, 2)
    )
    perfect_mean = mean(
        int(reference[(scenario, str(seed), "perfect")]["messages_sent"])
        for seed in (0, 1, 2)
    )
    result["message_difference_vs_certificate_pct"] = (
        100.0 * (result["messages_mean"] / cert_mean - 1.0)
    )
    result["message_reduction_vs_perfect_pct"] = (
        100.0 * (1.0 - result["messages_mean"] / perfect_mean)
    )
    result["payload_byte_reduction_vs_perfect_pct"] = (
        result["message_reduction_vs_perfect_pct"]
    )
    result["within_5pct_certificate_budget_aggregate"] = (
        abs(result["message_difference_vs_certificate_pct"]) <= 5.0
    )
    result["n_seeds_within_5pct_certificate_budget"] = sum(
        r["within_5pct_certificate_budget"] for r in group
    )

    # Keep the mean of paired seed-level percentages separate from
    # the percentage computed from aggregate message means.
    for column in (
        "message_difference_vs_certificate_pct",
        "message_reduction_vs_perfect_pct",
    ):
        values = [float(r[column]) for r in group]
        result[column + "_seedwise_mean"] = mean(values)
        result[column + "_seedwise_sd"] = stdev(values)

    aggregates.append(result)

scenario_order = {
    "CPM_mixed": 0, "intersection_2": 1,
    "on_ramp_1": 2, "roundabout_1": 3,
}
mode_order = {"certificate": 0, "periodic": 1, "age": 2, "perfect": 3}
aggregates.sort(key=lambda r: (
    scenario_order[r["scenario"]],
    0 if r["role"] == "primary" else 1,
    mode_order[r["mode"]],
    int(r["period_steps"] or r["max_age_steps"] or 0),
))
primary = [r for r in aggregates if r["role"] == "primary"]
require(len(primary) == 16, "Expected 16 primary rows")
require(
    Counter(r["scenario"] for r in primary)
    == {scenario: 4 for scenario in scenario_order},
    "Primary scenario grouping error",
)

OUT.mkdir(parents=True, exist_ok=True)
write_csv(OUT / "primary_comparison.csv", primary)
write_csv(OUT / "all_schedule_comparison.csv", aggregates)
write_csv(OUT / "seed_level_comparisons.csv", seed_comparisons)

def label(row):
    value = row["period_steps"] or row["max_age_steps"]
    return row["mode"] + (f" ({value})" if value else "")

def pm(row, metric, digits):
    return (
        f"{row[metric + '_mean']:.{digits}f} ± "
        f"{row[metric + '_sd_across_seeds']:.{digits}f}"
    )

lines = [
    "# Corrected communication comparison",
    "",
    "Performance values are means ± sample SD across three training seeds.",
    "Each seed-level performance value averages 32 simulations.",
    "The error measure is not a confidence interval.",
    "Message counts and payload bytes are totals per 32-simulation run, "
    "summarized across training seeds.",
    "Reductions compare aggregate mean counts with perfect communication "
    "in the same scenario.",
    "Payload accounting uses 44 bytes per message and excludes protocol overhead.",
    "Budget differences compare aggregate mean message counts with certificate "
    "communication. Seed-level budget checks are provided separately.",
    "",
]

for scenario in scenario_order:
    lines += [
        f"## {scenario}",
        "",
        "| Method | Role | Messages | Reduction (%) | Budget difference (%) | "
        "Collision union (%) | Speed (m/s) | Deviation (cm) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        if row["scenario"] != scenario:
            continue
        lines.append(
            f"| {label(row)} | {row['role']} | "
            f"{pm(row, 'messages', 1)} | "
            f"{row['message_reduction_vs_perfect_pct']:.2f} | "
            f"{row['message_difference_vs_certificate_pct']:+.2f} | "
            f"{pm(row, 'collision_union_pct', 3)} | "
            f"{pm(row, 'speed_m_s', 4)} | "
            f"{pm(row, 'deviation_cm', 3)} |"
        )
    lines.append("")

(OUT / "comparison_summary.md").write_text("\n".join(lines) + "\n")

outputs = [
    "primary_comparison.csv",
    "all_schedule_comparison.csv",
    "seed_level_comparisons.csv",
    "comparison_summary.md",
]
metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": plan["git_commit"],
    "plan_sha256": sha256(PLAN),
    "run_level_csv_sha256": sha256(ROOT / "run_level_results.csv"),
    "execution_metadata_sha256": sha256(ROOT / "execution_metadata.json"),
    "aggregation_script_sha256": sha256(Path(__file__)),
    "input_runs": 63,
    "primary_groups": 16,
    "all_schedule_groups": 21,
    "training_seeds": [0, 1, 2],
    "simulations_per_run": 32,
    "performance_aggregation": (
        "Equal-weight mean and sample SD (ddof=1) of three "
        "training-seed run means; each run mean covers 32 simulations."
    ),
    "communication_aggregation": (
        "Mean and sample SD of total counts per 32-simulation run."
    ),
    "reduction_formula": "100 * (1 - method_mean / perfect_mean)",
    "budget_difference_formula": "100 * (method_mean / certificate_mean - 1)",
    "limitations": [
        "No confidence intervals or significance tests are computed.",
        "Within-simulation SDs are not averaged to obtain across-seed SD.",
        "Payload bytes exclude network protocol overhead.",
        "Aggregation validates plan/CSV consistency, not raw rollout contents.",
        "Calibration and evaluation are not claimed to use independent random draws.",
    ],
    "output_sha256": {name: sha256(OUT / name) for name in outputs},
}
write_json(OUT / "aggregation_metadata.json", metadata)

print("PASS: 63 run-level rows match the frozen plan")
print("PASS: policy hashes, schedules, seeds, and payload accounting checked")
print("PASS: 16 primary comparison rows written")
print("PASS: 21 all-schedule comparison rows written")
print("PASS: 63 seed-level communication comparisons written")
print("PASS: across-seed sample SD computed from seed-level values")
print("PASS: Markdown summary and provenance metadata written")
print("OUTPUT DIRECTORY:", OUT)

print("\nPRIMARY CERTIFICATE RESULTS")
for row in primary:
    if row["mode"] == "certificate":
        print(
            f"{row['scenario']:15s} "
            f"collision={pm(row, 'collision_union_pct', 4)}%  "
            f"payload_reduction={row['message_reduction_vs_perfect_pct']:.2f}%"
        )
