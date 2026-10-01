#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path("/home/arsany/paper2/corrected_normalization")
ROOT = WORKSPACE / "runs/impairment/corrected_certificate_s0_s2"
OUT = WORKSPACE / "runs/aggregates/corrected_impairment_results_package"
PLAN_PATH = WORKSPACE / "manifests/corrected_impairment_plan_s0_s2.json"
EXPECTED_PLAN_SHA = (
    "af2dacf72fce1a31b339016a78c4299b23f079d6e24e035055772588b8e73556"
)

SCENARIOS = ("CPM_mixed", "intersection_2", "on_ramp_1", "roundabout_1")
CONDITIONS = ("clean", "delay2", "loss10", "outage")
LABELS = ("Clean", "Delay: 2 steps", "Loss: 10%", "Outages")

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

def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

require(sha256(PLAN_PATH) == EXPECTED_PLAN_SHA, "Frozen plan hash mismatch")
plan = read_json(PLAN_PATH)
metadata_path = ROOT / "execution_metadata.json"
execution = read_json(metadata_path)
require(execution["status"] == "completed", "Impairment execution incomplete")
require(execution["validated_new_jobs"] == 36, "Expected 36 impairment runs")
require(execution["validated_clean_references"] == 12, "Expected 12 clean runs")
require(execution["analysis_rows"] == 48, "Expected 48 analysis rows")
require(execution["plan_sha256"] == EXPECTED_PLAN_SHA, "Plan mismatch")
require(execution["git_commit"] == plan["git_commit"], "Source commit mismatch")
require(
    set(execution["completed_new_job_ids"])
    == {job["job_id"] for job in plan["jobs"]},
    "Completed jobs differ from plan",
)

source_csv = ROOT / "run_level_results.csv"
with source_csv.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
require(len(rows) == 48, "Expected 48 CSV rows")

def key(row):
    return row["scenario"], int(row["training_seed"]), row["condition"]

by_key = {key(row): row for row in rows}
expected_keys = {
    (scenario, seed, condition)
    for scenario in SCENARIOS
    for seed in (0, 1, 2)
    for condition in CONDITIONS
}
require(len(by_key) == 48 and set(by_key) == expected_keys,
        "Missing or duplicate scenario/seed/condition")

jobs = {
    (job["scenario"], job["training_seed"], job["condition"]): job
    for job in plan["jobs"]
}
references = {
    (ref["scenario"], ref["training_seed"]): ref
    for ref in plan["clean_references"]
}

metrics = {
    "collision_union_pct": "collision_union_frames_pct",
    "speed_m_s": "mean_speed_m_s",
    "deviation_cm": "mean_centerline_deviation_cm",
    "delivery_pct": "delivery_pct",
    "information_age_steps": "information_age_mean",
    "messages_sent": "messages_sent",
    "messages_received": "messages_received",
    "payload_bytes_sent": "bytes_sent",
    "payload_bytes_received": "bytes_received",
    "dropped_messages": "dropped_messages",
    "delayed_messages": "delayed_messages",
    "outage_blocked_messages": "outage_blocked_messages",
    "delivery_accounting_residual": "sent_minus_received_minus_dropped",
    "fallback_events": "fallback_events",
}

for row in rows:
    scenario, seed, condition = key(row)
    ref = references[(scenario, seed)]
    require(row["policy_sha256"] == ref["policy_sha256"], "Policy hash mismatch")
    require(int(row["simulations"]) == 32, "Expected 32 simulations")
    require(int(row["channel_random_seed"]) == 0, "Unexpected channel seed")

    if condition == "clean":
        expected_delay, expected_loss, expected_outages = 0, 0.0, []
        expected_run = Path(ref["run_directory"])
        for filename, hash_key in (
            ("manifest.json", "manifest_sha256"),
            ("communication_statistics.json", "statistics_sha256"),
            ("summary_untrimmed.json", "summary_sha256"),
        ):
            require(
                sha256(expected_run / filename) == ref[hash_key],
                f"Clean reference changed: {expected_run / filename}",
            )
    else:
        job = jobs[key(row)]
        expected_delay = job["communication_delay_steps"]
        expected_loss = job["communication_loss_probability"]
        expected_outages = job["communication_outage_windows"]
        expected_run = ROOT / job["job_id"]

    require(Path(row["run_directory"]).resolve() == expected_run.resolve(),
            "Run directory mismatch")
    require(int(row["delay_steps"]) == expected_delay, "Delay mismatch")
    require(float(row["loss_probability"]) == expected_loss, "Loss mismatch")
    require(json.loads(row["outage_windows"]) == expected_outages,
            "Outage mismatch")

    for column in metrics.values():
        value = float(row[column])
        require(math.isfinite(value) and value >= 0,
                f"Invalid value: {key(row)} / {column}")

    sent = int(row["messages_sent"])
    received = int(row["messages_received"])
    dropped = int(row["dropped_messages"])
    residual = int(row["sent_minus_received_minus_dropped"])
    require(sent > 0, "Zero message denominator")
    require(sent - received - dropped == residual >= 0,
            "Message accounting mismatch")
    require(int(row["bytes_sent"]) == 44 * sent, "Sent-byte mismatch")
    require(int(row["bytes_received"]) == 44 * received, "Received-byte mismatch")
    require(
        math.isclose(float(row["delivery_pct"]), 100 * received / sent,
                     rel_tol=1e-10, abs_tol=1e-10),
        "Delivery percentage mismatch",
    )
    require(
        int(row["valid_slots"]) + int(row["fallback_events"])
        == int(row["eligible_slots"]),
        "Validity/fallback accounting mismatch",
    )
    if condition != "delay2":
        require(residual == 0, "Unexpected immediate-delivery residual")

groups = defaultdict(list)
for row in rows:
    groups[(row["scenario"], row["condition"])].append(row)

aggregates = []
for scenario in SCENARIOS:
    for condition in CONDITIONS:
        group = groups[(scenario, condition)]
        require(
            len(group) == 3
            and {int(row["training_seed"]) for row in group} == {0, 1, 2},
            "Each group must contain exactly three training seeds",
        )
        result = {
            "scenario": scenario,
            "condition": condition,
            "n_training_seeds": 3,
            "simulations_per_seed": 32,
        }
        for label, column in metrics.items():
            values = [float(row[column]) for row in group]
            result[label + "_mean"] = mean(values)
            result[label + "_sd_across_seeds"] = stdev(values)
        result["information_age_max_across_runs"] = max(
            int(row["information_age_max"]) for row in group
        )
        aggregates.append(result)

# Pair each impaired run with the same scenario and training seed.
paired = []
for scenario in SCENARIOS:
    for condition in CONDITIONS[1:]:
        for seed in (0, 1, 2):
            row = by_key[(scenario, seed, condition)]
            clean = by_key[(scenario, seed, "clean")]
            paired.append({
                "scenario": scenario,
                "condition": condition,
                "training_seed": seed,
                "collision_change_pp":
                    float(row["collision_union_frames_pct"])
                    - float(clean["collision_union_frames_pct"]),
                "speed_change_m_s":
                    float(row["mean_speed_m_s"])
                    - float(clean["mean_speed_m_s"]),
                "deviation_change_cm":
                    float(row["mean_centerline_deviation_cm"])
                    - float(clean["mean_centerline_deviation_cm"]),
                "delivery_change_pp":
                    float(row["delivery_pct"]) - float(clean["delivery_pct"]),
                "information_age_change_steps":
                    float(row["information_age_mean"])
                    - float(clean["information_age_mean"]),
                "messages_sent_change_pct":
                    100 * (int(row["messages_sent"])
                           / int(clean["messages_sent"]) - 1),
            })

paired_groups = defaultdict(list)
for row in paired:
    paired_groups[(row["scenario"], row["condition"])].append(row)

paired_aggregate = []
change_columns = [
    column for column in paired[0]
    if column not in ("scenario", "condition", "training_seed")
]
for scenario in SCENARIOS:
    for condition in CONDITIONS[1:]:
        group = paired_groups[(scenario, condition)]
        result = {
            "scenario": scenario,
            "condition": condition,
            "n_training_seeds": 3,
        }
        for column in change_columns:
            values = [row[column] for row in group]
            result[column + "_mean"] = mean(values)
            result[column + "_sd_across_seeds"] = stdev(values)
        paired_aggregate.append(result)

require(len(aggregates) == 16, "Expected 16 aggregate groups")
require(len(paired) == 36, "Expected 36 seed-level paired comparisons")
require(len(paired_aggregate) == 12, "Expected 12 paired aggregate groups")

OUT.mkdir(parents=True, exist_ok=True)
write_csv(OUT / "impairment_aggregate.csv", aggregates)
write_csv(OUT / "paired_changes_by_seed.csv", paired)
write_csv(OUT / "paired_changes_aggregate.csv", paired_aggregate)

def pm(row, label, digits):
    return (
        f"{row[label + '_mean']:.{digits}f} ± "
        f"{row[label + '_sd_across_seeds']:.{digits}f}"
    )

lines = [
    "# Corrected certificate impairment evaluation",
    "",
    "Values are means ± sample SD across three training seeds.",
    "Each performance value for a training seed averages 32 simulations.",
    "Error bars and ± values are not confidence intervals.",
    "Delivery percentage is the mean of the three run-level "
    "received/sent percentages.",
    "Paired changes subtract the clean result for the same training seed "
    "and scenario; they are not episode-level paired comparisons.",
    "Message counts are totals per 32-simulation run.",
    "",
    "| Scenario | Condition | Collision union (%) | Speed (m/s) | "
    "Delivery (%) | Information age (steps) |",
    "|---|---|---:|---:|---:|---:|",
]
for row in aggregates:
    lines.append(
        f"| {row['scenario']} | {row['condition']} | "
        f"{pm(row, 'collision_union_pct', 3)} | "
        f"{pm(row, 'speed_m_s', 4)} | "
        f"{pm(row, 'delivery_pct', 2)} | "
        f"{pm(row, 'information_age_steps', 3)} |"
    )
lines += [
    "",
    "The delay accounting residual is sent minus received minus recorded "
    "drops. It is not labeled as packet loss or queue occupancy because "
    "the counters do not separate pending and reset-discarded messages.",
    "",
    "The study uses certificate communication only, with one channel random "
    "seed. It does not establish comparative robustness against other schedulers.",
]
(OUT / "impairment_summary.md").write_text("\n".join(lines) + "\n")

# Four panels; offsets separate overlapping error bars.
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "pdf.fonttype": 42,
})
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = (
    ("collision_union_pct", "Collision outcome", "Collision union (%)"),
    ("delivery_pct", "Communication delivery", "Received / sent (%)"),
    ("speed_m_s", "Traffic efficiency", "Mean speed (m/s)"),
    ("information_age_steps", "Information freshness", "Information age (steps)"),
)
lookup = {
    (row["scenario"], row["condition"]): row for row in aggregates
}
colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
markers = ("o", "s", "^", "D")
display_names = ("CPM mixed", "Intersection 2", "On-ramp 1", "Roundabout 1")

for ax, (metric, title, ylabel) in zip(axes.flat, panels):
    for index, scenario in enumerate(SCENARIOS):
        values = [lookup[(scenario, c)] for c in CONDITIONS]
        x = [position + (index - 1.5) * 0.055 for position in range(4)]
        ax.errorbar(
            x,
            [row[metric + "_mean"] for row in values],
            yerr=[row[metric + "_sd_across_seeds"] for row in values],
            fmt=markers[index],
            linestyle="none",
            color=colors[index],
            capsize=4,
            markersize=5,
            label=display_names[index],
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(4), LABELS)
    ax.set_xlim(-0.3, 3.3)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

fig.suptitle(
    "Corrected frozen-policy communication impairment evaluation\n"
    "Mean ± sample SD across three training seeds",
    fontsize=14,
    y=0.99,
)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           bbox_to_anchor=(0.5, 0.915), ncol=4, frameon=False)
fig.text(
    0.5, 0.015,
    "32 simulations per training seed and condition. "
    "Error bars are sample SD, not confidence intervals.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.86))

figure_stem = "communication_impairment_summary_errorbars"
fig.savefig(OUT / f"{figure_stem}.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / f"{figure_stem}.pdf", bbox_inches="tight")
plt.close(fig)

outputs = [
    "impairment_aggregate.csv",
    "paired_changes_by_seed.csv",
    "paired_changes_aggregate.csv",
    "impairment_summary.md",
    f"{figure_stem}.png",
    f"{figure_stem}.pdf",
]
write_json(OUT / "package_metadata.json", {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": plan["git_commit"],
    "plan_sha256": sha256(PLAN_PATH),
    "run_level_csv_sha256": sha256(source_csv),
    "execution_metadata_sha256": sha256(metadata_path),
    "builder_sha256": sha256(Path(__file__)),
    "analysis_rows": 48,
    "aggregate_groups": 16,
    "paired_seed_comparisons": 36,
    "paired_aggregate_groups": 12,
    "training_seeds": [0, 1, 2],
    "simulations_per_run": 32,
    "channel_random_seeds": [0],
    "uncertainty": "Across-training-seed sample SD, ddof=1; not confidence intervals",
    "delivery_aggregation": "Arithmetic mean of run-level received/sent percentages",
    "paired_changes": "Within scenario and training seed, impaired minus clean",
    "validation_scope": "Plan, CSV, accounting, and frozen clean-reference hashes",
    "output_sha256": {name: sha256(OUT / name) for name in outputs},
})

print("PASS: 48 rows and all scenario-condition-seed combinations validated")
print("PASS: 16 impairment aggregate rows written")
print("PASS: 36 seed-level paired changes written")
print("PASS: 12 paired aggregate rows written")
print("PASS: Markdown summary generated")
print("PASS: uncertainty-aware PNG and PDF generated")
print("PASS: package metadata and checksums written")
print("PACKAGE:", OUT)
print("FIGURE:", OUT / f"{figure_stem}.png")
