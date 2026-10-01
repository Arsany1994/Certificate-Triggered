#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path("/home/arsany/paper2/corrected_normalization")
TRAINING = (
    WORKSPACE / "runs/training/"
    "corrected_certificate_s0_s2_0lbr4r69"
)
OUT = WORKSPACE / "runs/aggregates/corrected_training_curves"

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Read iteration-associated rewards, not "Saved model" messages,
# which report the best checkpoint reward rather than each iteration.
pattern = re.compile(
    r"Episode mean reward\s*=\s*([-+0-9.eE]+):"
    r"[^\r\n]*?\|\s*(\d+)/250\s*\["
)

histories = {}
sources = []
expected_maxima = {0: 5.04, 1: 5.31, 2: 5.57}

for seed in (0, 1, 2):
    path = TRAINING / f"seed{seed}" / "training.log"
    require(path.is_file(), f"Training log not found: {path}")

    text = path.read_text(errors="replace")
    history = {}
    for reward_text, iteration_text in pattern.findall(text):
        iteration = int(iteration_text)
        reward = float(reward_text)
        require(math.isfinite(reward), "Nonfinite reward")
        if iteration in history:
            require(
                history[iteration] == reward,
                f"Conflicting duplicate iteration {iteration} in {path}",
            )
        history[iteration] = reward

    require(
        set(history) == set(range(1, 251)),
        f"Expected exactly iterations 1–250 for seed {seed}; "
        f"found {len(history)} unique iterations",
    )
    values = [history[i] for i in range(1, 251)]
    require(
        math.isclose(max(values), expected_maxima[seed], abs_tol=1e-9),
        f"Unexpected logged maximum for seed {seed}",
    )
    histories[seed] = values
    sources.append({
        "training_seed": seed,
        "log": str(path),
        "sha256": sha256(path),
        "iterations": len(values),
        "maximum_logged_reward": max(values),
        "final_logged_reward": values[-1],
    })
    print(
        f"PASS seed {seed}: 250 iterations, "
        f"max={max(values):.2f}, final={values[-1]:.2f}"
    )

rows = []
for index in range(250):
    values = [histories[seed][index] for seed in (0, 1, 2)]
    rows.append({
        "training_iteration": index + 1,
        "seed0_logged_reward": values[0],
        "seed1_logged_reward": values[1],
        "seed2_logged_reward": values[2],
        "reward_mean": mean(values),
        "reward_sd_across_seeds": stdev(values),
    })

OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "training_reward_by_iteration.csv"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "pdf.fonttype": 42,
})

iterations = list(range(1, 251))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
colors = ("#0072B2", "#D55E00", "#009E73")

for seed, color in zip((0, 1, 2), colors):
    axes[0].plot(
        iterations, histories[seed],
        color=color, linewidth=0.9, label=f"Seed {seed}",
    )
axes[0].set_title("(a) Individual training seeds")
axes[0].legend(frameon=False, loc="lower right")

means = [row["reward_mean"] for row in rows]
sds = [row["reward_sd_across_seeds"] for row in rows]
axes[1].fill_between(
    iterations,
    [m - sd for m, sd in zip(means, sds)],
    [m + sd for m, sd in zip(means, sds)],
    color="#0072B2", alpha=0.20, label="± sample SD",
)
axes[1].plot(
    iterations, means,
    color="#0072B2", linewidth=1.2, label="Three-seed mean",
)
axes[1].set_title("(b) Mean and across-seed variability")
axes[1].legend(frameon=False, loc="lower right")

# Use the same reward limits in both panels.
lower = min(
    min(min(values) for values in histories.values()),
    min(m - sd for m, sd in zip(means, sds)),
)
upper = max(
    max(max(values) for values in histories.values()),
    max(m + sd for m, sd in zip(means, sds)),
)
padding = 0.05 * (upper - lower)

for ax in axes:
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Logged episode mean reward")
    ax.set_xlim(1, 250)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xticks([1, 50, 100, 150, 200, 250])
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

fig.tight_layout()
stem = "corrected_training_reward"
fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
plt.close(fig)

caption = (
    "Training reward histories for the corrected certificate-triggered "
    "policies trained in CPM_mixed with four agents. "
    "(a) Logged episode mean reward for training seeds 0, 1, and 2 over "
    "250 training iterations. (b) Arithmetic mean across the three seeds "
    "at each iteration; shading denotes plus or minus one across-seed "
    "sample standard deviation (ddof=1), not a confidence interval. "
    "Curves are unsmoothed and use rewards recorded to two decimal places "
    "in the training logs. Evaluation uses the selected best saved "
    "intermediate checkpoint from each training run."
)
(OUT / "figure_caption.txt").write_text(caption + "\n")

outputs = [
    "training_reward_by_iteration.csv",
    f"{stem}.png",
    f"{stem}.pdf",
    "figure_caption.txt",
]
metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "training_batch": str(TRAINING),
    "sources": sources,
    "builder_sha256": sha256(Path(__file__)),
    "iterations_per_seed": 250,
    "training_seeds": [0, 1, 2],
    "smoothing": "none",
    "reward_source": "Iteration-associated Episode mean reward log entries",
    "reward_precision": "Two decimal places as recorded in logs",
    "aggregation": "Arithmetic mean and sample SD across seeds at each iteration",
    "uncertainty": "Sample SD, ddof=1; not a confidence interval",
    "notes": [
        "Training reward is distinct from evaluation collision and speed metrics.",
        "The plot does not establish mathematical convergence.",
        "Logged maxima do not identify exact checkpoint iterations when rounded values tie.",
    ],
    "output_sha256": {name: sha256(OUT / name) for name in outputs},
}
(OUT / "training_curve_metadata.json").write_text(
    json.dumps(metadata, indent=2, allow_nan=False) + "\n"
)

print("PASS: all 750 iteration-associated reward values extracted")
print("PASS: unsmoothed seed curves and mean ± sample SD generated")
print("PASS: CSV, PNG, PDF, caption, and metadata saved")
print("OUTPUT DIRECTORY:", OUT)
print("FIGURE:", OUT / f"{stem}.png")
