#!/usr/bin/env python3
"""Build mean-only and uncertainty-aware impairment figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCENARIOS = [
    "CPM_mixed",
    "intersection_2",
    "on_ramp_1",
    "roundabout_1",
]

CONDITIONS = [
    "certificate_clean",
    "delay2",
    "loss10",
    "outage",
]

CONDITION_LABELS = {
    "certificate_clean": "Clean",
    "delay2": "Delay 2",
    "loss10": "Loss 10%",
    "outage": "Outage",
}

COLORS = {
    "CPM_mixed": "#1f77b4",
    "intersection_2": "#ff7f0e",
    "on_ramp_1": "#2ca02c",
    "roundabout_1": "#d62728",
}


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(rows, scenario, condition, name):
    matches = [
        row
        for row in rows
        if row["scenario"] == scenario
        and row["condition"] == condition
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one row for {scenario}/{condition}, "
            f"found {len(matches)}"
        )

    return float(matches[0][name])


def build_figure(rows, output: Path, with_errorbars: bool):
    x = list(range(len(CONDITIONS)))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11),
    )

    panels = [
        (
            axes[0, 0],
            "collision_union_mean",
            "collision_union_sd",
            "Collision union (%)",
            "Safety outcome",
        ),
        (
            axes[0, 1],
            "delivery_pct_mean",
            "delivery_pct_sd",
            "Message delivery (%)",
            "Communication delivery",
        ),
        (
            axes[1, 0],
            "speed_mean",
            "speed_sd",
            "Mean speed (m/s)",
            "Efficiency outcome",
        ),
        (
            axes[1, 1],
            "information_age_mean",
            "information_age_sd",
            "Information age (steps)",
            "Information freshness",
        ),
    ]

    for axis, mean_key, sd_key, ylabel, title in panels:
        for scenario in SCENARIOS:
            means = [
                value(rows, scenario, condition, mean_key)
                for condition in CONDITIONS
            ]

            if with_errorbars:
                errors = [
                    value(rows, scenario, condition, sd_key)
                    for condition in CONDITIONS
                ]
                axis.errorbar(
                    x,
                    means,
                    yerr=errors,
                    marker="o",
                    linewidth=2.5,
                    capsize=4,
                    label=scenario,
                    color=COLORS[scenario],
                )
            else:
                axis.plot(
                    x,
                    means,
                    marker="o",
                    linewidth=2.5,
                    label=scenario,
                    color=COLORS[scenario],
                )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [CONDITION_LABELS[item] for item in CONDITIONS],
            rotation=20,
            ha="right",
        )
        axis.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best")

    title = (
        "Frozen-Policy Communication Impairment Evaluation"
        + (" (mean ± across-seed SD)" if with_errorbars else "")
    )
    figure.suptitle(title, fontsize=18)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if len(rows) != 16:
        raise RuntimeError(
            f"Expected 16 aggregate rows, found {len(rows)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mean_path = (
        args.output_dir / "communication_impairment_summary.png"
    )
    errorbar_path = (
        args.output_dir
        / "communication_impairment_summary_errorbars.png"
    )

    build_figure(rows, mean_path, with_errorbars=False)
    build_figure(rows, errorbar_path, with_errorbars=True)

    print("PASS: mean-only figure generated")
    print("PASS: uncertainty-aware figure generated")
    print(f"MEAN FIGURE: {mean_path}")
    print(f"ERRORBAR FIGURE: {errorbar_path}")


if __name__ == "__main__":
    main()
