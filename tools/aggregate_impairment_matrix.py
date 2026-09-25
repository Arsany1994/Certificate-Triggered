#!/usr/bin/env python3
"""Aggregate completed impairment runs across the three training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


SCENARIOS = {
    "CPM_mixed": 4,
    "intersection_2": 6,
    "on_ramp_1": 8,
    "roundabout_1": 8,
}

CONDITIONS = {
    "certificate_clean",
    "delay2",
    "loss10",
    "outage",
}

JOB_RE = re.compile(
    r"^(?P<scenario>CPM_mixed|intersection_2|on_ramp_1|"
    r"roundabout_1)_(?P<condition>certificate_clean|delay2|"
    r"loss10|outage)_seed(?P<seed>[0-9]+)$"
)


def read_json(path: Path):
    return json.loads(path.read_text())


def metric_mean(summary, name: str) -> float:
    return float(summary["metrics"][name]["mean"])


def read_job(job_dir: Path):
    match = JOB_RE.match(job_dir.name)
    if match is None:
        return None

    scenario = match.group("scenario")
    condition = match.group("condition")
    seed = int(match.group("seed"))

    manifest_path = job_dir / "manifest.json"
    stats_path = job_dir / "communication_statistics.json"
    summary_path = job_dir / "summary_untrimmed.json"

    if not (
        manifest_path.is_file()
        and stats_path.is_file()
        and summary_path.is_file()
    ):
        raise RuntimeError(
            f"Incomplete job directory: {job_dir}"
        )

    manifest = read_json(manifest_path)
    stats = read_json(stats_path)
    summary = read_json(summary_path)

    assert manifest.get("status") == "completed", job_dir

    sent = int(stats["messages_sent"])
    received = int(stats["messages_received"])

    delivery = 0.0 if sent == 0 else 100.0 * received / sent

    return {
        "job_id": job_dir.name,
        "scenario": scenario,
        "condition": condition,
        "seed": seed,
        "agents": SCENARIOS[scenario],
        "status": "completed",
        "run_directory": str(job_dir),
        "policy_sha256": manifest.get("policy_sha256", ""),
        "messages_sent": sent,
        "messages_received": received,
        "delivery_pct": delivery,
        "bytes_sent": int(stats["bytes_sent"]),
        "bytes_received": int(stats["bytes_received"]),
        "collision_union_mean": metric_mean(
            summary,
            "collision_union_frames_pct",
        ),
        "speed_mean": metric_mean(summary, "mean_speed_m_s"),
        "deviation_mean_cm": metric_mean(
            summary,
            "mean_centerline_deviation_cm",
        ),
        "information_age_mean": float(
            stats["mean_valid_information_age"]
        ),
        "max_information_age": float(
            stats["max_valid_information_age"]
        ),
    }


def sample_sd(values):
    return 0.0 if len(values) < 2 else statistics.stdev(values)


def aggregate_rows(run_rows):
    grouped = defaultdict(list)

    for row in run_rows:
        grouped[(row["scenario"], row["condition"])].append(row)

    rows = []

    for scenario in sorted(SCENARIOS):
        for condition in sorted(CONDITIONS):
            values = grouped[(scenario, condition)]

            if len(values) != 3:
                raise RuntimeError(
                    f"{scenario}/{condition} has "
                    f"{len(values)} seeds instead of 3"
                )

            def vals(name):
                return [float(item[name]) for item in values]

            rows.append(
                {
                    "scenario": scenario,
                    "condition": condition,
                    "n_seeds": len(values),
                    "messages_mean": statistics.mean(
                        vals("messages_sent")
                    ),
                    "messages_sd": sample_sd(
                        vals("messages_sent")
                    ),
                    "messages_received_mean": statistics.mean(
                        vals("messages_received")
                    ),
                    "messages_received_sd": sample_sd(
                        vals("messages_received")
                    ),
                    "delivery_pct_mean": statistics.mean(
                        vals("delivery_pct")
                    ),
                    "delivery_pct_sd": sample_sd(
                        vals("delivery_pct")
                    ),
                    "bytes_mean": statistics.mean(
                        vals("bytes_sent")
                    ),
                    "bytes_sd": sample_sd(vals("bytes_sent")),
                    "collision_union_mean": statistics.mean(
                        vals("collision_union_mean")
                    ),
                    "collision_union_sd": sample_sd(
                        vals("collision_union_mean")
                    ),
                    "speed_mean": statistics.mean(
                        vals("speed_mean")
                    ),
                    "speed_sd": sample_sd(
                        vals("speed_mean")
                    ),
                    "deviation_mean_cm": statistics.mean(
                        vals("deviation_mean_cm")
                    ),
                    "deviation_sd_cm": sample_sd(
                        vals("deviation_mean_cm")
                    ),
                    "information_age_mean": statistics.mean(
                        vals("information_age_mean")
                    ),
                    "information_age_sd": sample_sd(
                        vals("information_age_mean")
                    ),
                    "max_information_age_mean": statistics.mean(
                        vals("max_information_age")
                    ),
                    "max_information_age_sd": sample_sd(
                        vals("max_information_age")
                    ),
                }
            )

    return rows


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError("No rows to write")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows):
    lines = [
        "# Communication Impairment Results",
        "",
        "Means and sample SD across three training seeds.",
        "",
        "| Scenario | Condition | Delivery (%) | "
        "Collision union (%) | Speed (m/s) | Age (steps) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['condition']} | "
            f"{row['delivery_pct_mean']:.2f} ± "
            f"{row['delivery_pct_sd']:.2f} | "
            f"{row['collision_union_mean']:.3f} ± "
            f"{row['collision_union_sd']:.3f} | "
            f"{row['speed_mean']:.4f} ± "
            f"{row['speed_sd']:.4f} | "
            f"{row['information_age_mean']:.3f} ± "
            f"{row['information_age_sd']:.3f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []

    for job_dir in sorted(root.iterdir()):
        if not job_dir.is_dir():
            continue
        if ".incomplete_" in job_dir.name:
            continue

        row = read_job(job_dir)
        if row is not None:
            run_rows.append(row)

    expected = 3 * len(SCENARIOS) * len(CONDITIONS)
    if len(run_rows) != expected:
        raise RuntimeError(
            f"Expected {expected} completed jobs, "
            f"found {len(run_rows)}"
        )

    run_rows.sort(
        key=lambda row: (
            row["scenario"],
            row["condition"],
            row["seed"],
        )
    )

    aggregate = aggregate_rows(run_rows)

    write_csv(output_dir / "run_level_results.csv", run_rows)
    write_csv(output_dir / "aggregate_results.csv", aggregate)
    write_markdown(
        output_dir / "aggregate_results.md",
        aggregate,
    )

    metadata_path = output_dir / "metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = read_json(metadata_path)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    metadata.update(
        {
            "status": "completed",
            "expected_jobs": expected,
            "completed_jobs": len(run_rows),
            "aggregate_rows": len(aggregate),
            "git_commit": commit,
            "scenarios": SCENARIOS,
            "conditions": sorted(CONDITIONS),
        }
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    (output_dir / "aggregate_results.json").write_text(
        json.dumps(aggregate, indent=2)
    )

    print(f"PASS: {len(run_rows)} run-level rows written")
    print(f"PASS: {len(aggregate)} aggregate rows written")
    print(f"RUN-LEVEL CSV: {output_dir / 'run_level_results.csv'}")
    print(f"AGGREGATE CSV: {output_dir / 'aggregate_results.csv'}")


if __name__ == "__main__":
    main()
