#!/usr/bin/env python3
"""Run archived analysis with explicitly relocated paths."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

data = args.data.resolve()
output = args.output.resolve()
repo = Path(__file__).resolve().parents[2]
historical = repo / "tools/corrected_normalization"

if output.exists():
    raise SystemExit("STOP: output directory must not already exist.")
for protected in (data, repo):
    if output == protected or protected in output.parents:
        raise SystemExit("STOP: choose output outside data and repository.")

old = "/home/arsany/paper2/corrected_normalization"
clean = "runs/evaluation/corrected_full_comparison_s0_s2"
impaired = "runs/impairment/corrected_certificate_s0_s2"
training = "runs/training/corrected_certificate_s0_s2_0lbr4r69"

scripts = {
    "aggregate_corrected_comparison.py":
        "48ae2bc4320411fba188dcb24d2a5ca682a5abeec924930742a172225fcd2211",
    "build_corrected_impairment_package.py":
        "dcc3bde9860c80135f0e9de5f769b1be50cd1fbf683b098a9b9f8d8db9938688",
    "plot_corrected_training.py":
        "8c7e4f15efd26135c3ad2f548d8605dbac44834f4fafe52d7594d04d3840fd82",
}

links = {
    clean + "/run_level_results.csv":
        data / "results/clean/run_level_results.csv",
    clean + "/execution_metadata.json":
        data / "provenance/full_evaluation_execution.json",
    impaired + "/run_level_results.csv":
        data / "results/impairments/run_level_results.csv",
    impaired + "/execution_metadata.json":
        data / "provenance/impairment_evaluation_execution.json",
    training: data / "training",
}
for name in (
    "corrected_full_evaluation_plan_s0_s2.json",
    "corrected_impairment_plan_s0_s2.json",
):
    links["manifests/" + name] = data / "plans" / name

for suite, relative, expected in (
    ("clean", clean, 63),
    ("impairments", impaired, 36),
):
    manifests = sorted((data / "runs" / suite).glob("*/manifest.json"))
    if len(manifests) != expected:
        raise SystemExit(f"STOP: unexpected {suite} run count.")
    for manifest in manifests:
        links[relative + "/" + manifest.parent.name] = manifest.parent

for name, expected in scripts.items():
    if digest(historical / name) != expected:
        raise SystemExit(f"STOP: historical script hash mismatch: {name}")
for target in links.values():
    if not target.exists():
        raise SystemExit(f"STOP: missing input: {target}")

output.mkdir(parents=True)
workspace = output / "workspace"
adapted = output / "adapted_scripts"
adapted.mkdir()

for relative, target in links.items():
    link = workspace / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())

helper = f'''
def relocated_run(value):
    original = Path(value)
    historical_root = Path({old!r})
    try:
        relative = original.relative_to(historical_root)
    except ValueError:
        raise RuntimeError("Unexpected historical run path: " + str(original))
    if ".." in relative.parts:
        raise RuntimeError("Invalid historical run path")
    return WORKSPACE / relative

'''

records = []
for name, expected in scripts.items():
    source = historical / name
    text = source.read_text()
    marker = f'WORKSPACE = Path("{old}")'
    if text.count(marker) != 1:
        raise SystemExit(f"STOP: unexpected workspace assignment: {name}")
    text = text.replace(
        marker, f"WORKSPACE = Path({str(workspace)!r})\n" + helper, 1
    )

    replacements = []
    if name != "plot_corrected_training.py":
        replacements.append((
            'Path(row["run_directory"])',
            'relocated_run(row["run_directory"])',
        ))
    if name == "build_corrected_impairment_package.py":
        replacements.append((
            'Path(ref["run_directory"])',
            'relocated_run(ref["run_directory"])',
        ))

    for before, after in replacements:
        if text.count(before) != 1:
            raise SystemExit(f"STOP: unexpected path expression: {name}")
        text = text.replace(before, after, 1)

    target = adapted / name
    target.write_text(text)
    records.append({
        "script": name,
        "historical_sha256": expected,
        "adapted_sha256": digest(target),
        "changes": [
            "Relocate WORKSPACE to the separate analysis workspace.",
            "Add historical run-path relocation helper.",
        ] + [f"{before} -> {after}" for before, after in replacements],
    })

record = {
    "status": "prepared",
    "data": str(data),
    "output": str(output),
    "python": sys.executable,
    "launcher_sha256": digest(Path(__file__).resolve()),
    "scripts": records,
    "note": (
        "Historical scripts and inputs are unchanged. Adapted copies change "
        "path handling only; statistical calculations and checks are retained."
    ),
}
record_path = output / "analysis_execution.json"
record_path.write_text(json.dumps(record, indent=2) + "\n")

for item in records:
    name = item["script"]
    print(f"RUNNING: {name}", flush=True)
    with (output / (name + ".log")).open("w") as log:
        result = subprocess.run(
            [sys.executable, str(adapted / name)],
            stdout=log, stderr=subprocess.STDOUT,
        )
    item["returncode"] = result.returncode
    if result.returncode:
        record["status"] = "failed"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise SystemExit(
            f"STOP: {name} failed. Inspect {output / (name + '.log')}"
        )
    print(f"PASS: {name}", flush=True)

record["status"] = "completed"
record_path.write_text(json.dumps(record, indent=2) + "\n")
print("PASS: all three analysis scripts completed.")
print("Generated results:", workspace / "runs/aggregates")
print("Numerical comparison against original aggregates remains a separate step.")
