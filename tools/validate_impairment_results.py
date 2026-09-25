#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path

PACKAGE = Path(
    "/home/arsany/paper2/runs/proposed_impairment_results_package"
)
MATRIX = Path(
    "/home/arsany/paper2/runs/proposed_impairment_matrix_s0_s2_full"
)

SCENARIOS = {
    "CPM_mixed",
    "intersection_2",
    "on_ramp_1",
    "roundabout_1",
}

CONDITIONS = {
    "certificate_clean",
    "delay2",
    "loss10",
    "outage",
}

PACKAGE_FILES = [
    "impairment_summary.json",
    "impairment_summary.md",
    "communication_impairment_summary.png",
    "communication_impairment_summary_errorbars.png",
    "package_metadata.json",
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


for filename in PACKAGE_FILES:
    path = PACKAGE / filename
    assert path.is_file() and path.stat().st_size > 0, (
        f"Missing or empty package file: {path}"
    )

run_csv = MATRIX / "run_level_results.csv"
aggregate_csv = MATRIX / "aggregate_results.csv"
metadata_json = MATRIX / "metadata.json"

assert run_csv.is_file(), run_csv
assert aggregate_csv.is_file(), aggregate_csv
assert metadata_json.is_file(), metadata_json

runs = read_csv(run_csv)
aggregates = read_csv(aggregate_csv)
metadata = json.loads(metadata_json.read_text())

assert len(runs) == 48, f"Expected 48 run rows, found {len(runs)}"
assert len(aggregates) == 16, (
    f"Expected 16 aggregate rows, found {len(aggregates)}"
)

scenario_counts = Counter(row["scenario"] for row in runs)
condition_counts = Counter(row["condition"] for row in runs)
pair_counts = Counter(
    (row["scenario"], row["condition"])
    for row in runs
)

assert set(scenario_counts) == SCENARIOS, scenario_counts
assert set(condition_counts) == CONDITIONS, condition_counts
assert all(value == 12 for value in scenario_counts.values())
assert all(value == 12 for value in condition_counts.values())
assert len(pair_counts) == 16
assert all(value == 3 for value in pair_counts.values())

aggregate_pairs = {
    (row["scenario"], row["condition"])
    for row in aggregates
}
assert aggregate_pairs == set(pair_counts), aggregate_pairs

statuses = [
    row.get("status")
    for row in runs
    if row.get("status")
]
if statuses:
    assert set(statuses) == {"completed"}, set(statuses)

job_ids = [
    row.get("job_id")
    for row in runs
    if row.get("job_id")
]
if job_ids:
    assert len(job_ids) == len(set(job_ids)), (
        "Duplicate job IDs detected"
    )

assert metadata.get("expected_jobs") == 48
assert metadata.get("completed_jobs") == 48

print("PASS: impairment package files validated")
print("PASS: 48 run-level rows validated")
print("PASS: 16 aggregate rows validated")
print("PASS: four scenarios and four conditions validated")
print("PASS: every scenario-condition pair has three seeds")
print("PASS: matrix metadata reports 48 completed jobs")
print("PASS: uncertainty-aware figure is present")
print(f"PACKAGE: {PACKAGE}")
print(f"MATRIX: {MATRIX}")
