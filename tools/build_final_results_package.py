#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path


SCENARIOS = [
    "CPM_mixed",
    "intersection_2",
    "on_ramp_1",
    "roundabout_1",
]
MODES = ["certificate", "age", "periodic", "perfect"]
OUTPUT_DIR = Path(
    "/home/arsany/paper2/runs/final_results_package_corrected"
)

SOURCE_FILES = {
    "CPM_mixed": Path(
        "/home/arsany/paper2/runs/proposed_seed_aggregate_s0_s2/"
        "seed_aggregate_results.csv"
    ),
    "intersection_2": Path(
        "/home/arsany/paper2/runs/proposed_seed_aggregate_intersection2_s0_s2/"
        "seed_aggregate_results.csv"
    ),
    "on_ramp_1": Path(
        "/home/arsany/paper2/runs/proposed_extended_evaluations_s0_s2_lo5h3g3m/"
        "scenario_mode_aggregate.csv"
    ),
    "roundabout_1": Path(
        "/home/arsany/paper2/runs/proposed_extended_evaluations_s0_s2_lo5h3g3m/"
        "scenario_mode_aggregate.csv"
    ),
}


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, *names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Missing {names}; available={sorted(row)}")


def normalize(scenario, row):
    method = row.get("method") or row.get("mode")
    if method not in MODES:
        raise ValueError(f"Unexpected method {method!r}")

    return {
        "scenario": scenario,
        "method": method,
        "n_seeds": int(number(row, "n_seeds")),
        "messages_mean": number(
            row, "messages_sent_mean", "messages_mean"
        ),
        "messages_sd": number(
            row, "messages_sent_sd", "messages_sd"
        ),
        "bytes_mean": number(row, "bytes_sent_mean", "bytes_mean"),
        "bytes_sd": number(row, "bytes_sent_sd", "bytes_sd"),
        "bandwidth_reduction_vs_perfect_pct": None,
        "collision_mean": number(
            row, "collision_union_mean", "collision_mean"
        ),
        "collision_sd": number(
            row,
            "collision_union_sd_across_seeds",
            "collision_sd",
        ),
        "speed_mean": number(row, "speed_mean"),
        "speed_sd": number(
            row, "speed_sd_across_seeds", "speed_sd"
        ),
        "deviation_mean_cm": number(row, "deviation_mean_cm"),
        "deviation_sd_cm": number(
            row, "deviation_sd_across_seeds", "deviation_sd_cm"
        ),
        "information_age_mean": number(
            row, "information_age_mean", "mean_information_age"
        ),
        "information_age_sd": number(
            row,
            "information_age_sd_across_seeds",
            "information_age_sd",
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in SCENARIOS:
        source_rows = read_rows(SOURCE_FILES[scenario])

        # The extended source contains both held-out scenarios.
        if "scenario" in source_rows[0]:
            source_rows = [
                row for row in source_rows
                if row["scenario"] == scenario
            ]

        if len(source_rows) != 4:
            raise RuntimeError(
                f"{scenario}: expected 4 rows, found {len(source_rows)}"
            )

        rows.extend(normalize(scenario, row) for row in source_rows)

    rows.sort(
        key=lambda row: (
            SCENARIOS.index(row["scenario"]),
            MODES.index(row["method"]),
        )
    )

    counts = Counter(row["scenario"] for row in rows)
    expected = Counter({scenario: 4 for scenario in SCENARIOS})
    if counts != expected:
        raise RuntimeError(f"Unexpected scenario counts: {counts}")

    for scenario in SCENARIOS:
        perfect = next(
            row for row in rows
            if row["scenario"] == scenario
            and row["method"] == "perfect"
        )
        for row in rows:
            if row["scenario"] == scenario:
                row["bandwidth_reduction_vs_perfect_pct"] = (
                    100.0
                    * (1.0 - row["bytes_mean"] / perfect["bytes_mean"])
                )

    fields = list(rows[0])

    with (OUTPUT_DIR / "combined_results.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT_DIR / "combined_results.json").open("w") as handle:
        json.dump(
            {
                "scenario_counts": dict(counts),
                "methods": MODES,
                "results": rows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    with (OUTPUT_DIR / "combined_results.md").open("w") as handle:
        handle.write("# Combined Results\n\n")
        handle.write(
            "Values are means across three training seeds; "
            "± denotes across-seed sample SD.\n\n"
        )

        for scenario in SCENARIOS:
            handle.write(f"## {scenario}\n\n")
            handle.write(
                "| Method | Messages | Bytes | Bandwidth reduction | "
                "Collision union (%) | Speed (m/s) | Deviation (cm) |\n"
            )
            handle.write(
                "|---|---:|---:|---:|---:|---:|---:|\n"
            )

            for row in rows:
                if row["scenario"] != scenario:
                    continue
                handle.write(
                    f"| {row['method']} "
                    f"| {row['messages_mean']:.1f} ± "
                    f"{row['messages_sd']:.1f} "
                    f"| {row['bytes_mean']:.1f} ± "
                    f"{row['bytes_sd']:.1f} "
                    f"| {row['bandwidth_reduction_vs_perfect_pct']:.2f}% "
                    f"| {row['collision_mean']:.3f} ± "
                    f"{row['collision_sd']:.3f} "
                    f"| {row['speed_mean']:.4f} ± "
                    f"{row['speed_sd']:.4f} "
                    f"| {row['deviation_mean_cm']:.3f} ± "
                    f"{row['deviation_sd_cm']:.3f} |\n"
                )
            handle.write("\n")

    with (OUTPUT_DIR / "metadata.json").open("w") as handle:
        json.dump(
            {
                "scenario_counts": dict(counts),
                "methods": MODES,
                "total_rows": len(rows),
                "sources": {
                    scenario: str(path)
                    for scenario, path in SOURCE_FILES.items()
                },
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            2, 2, figsize=(13, 9), squeeze=False
        )
        axes = list(axes.ravel())

        for index, scenario in enumerate(SCENARIOS):
            selected = [
                row for row in rows
                if row["scenario"] == scenario
            ]
            labels = [row["method"] for row in selected]
            collision = [row["collision_mean"] for row in selected]
            bandwidth = [
                row["bandwidth_reduction_vs_perfect_pct"]
                for row in selected
            ]
            positions = list(range(len(labels)))
            axis = axes[index]

            axis.bar(positions, collision, color="#4472C4")
            axis.set_title(scenario)
            axis.set_ylabel("Collision union (%)")
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=25)
            axis.grid(axis="y", alpha=0.25)

            secondary = axis.twinx()
            secondary.plot(
                positions,
                bandwidth,
                color="#C00000",
                marker="o",
            )
            secondary.set_ylabel("Bandwidth reduction (%)")
            secondary.set_ylim(0, 100)

        figure.suptitle("Communication and Safety Summary")
        figure.tight_layout()
        figure.savefig(
            OUTPUT_DIR / "communication_safety_summary.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)
        print("PASS: summary figure generated")
    except Exception as error:
        print(f"WARNING: figure skipped: {error}")

    print("SCENARIO COUNTS:", dict(counts))
    print(f"PASS: wrote {len(rows)} correctly grouped rows")
    print(f"OUTPUT DIRECTORY: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
