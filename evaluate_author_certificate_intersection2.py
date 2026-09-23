import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

run = Path(os.environ["PAPER2_AUTHOR_RUN"]).resolve()
repo = Path.home() / "paper2/development/sigmarl-proposed"
source_model = Path.home() / "paper2/vendor/SigmaRL/checkpoints/itsc24/M0 (our)"
model = run / "model"
model.mkdir()

expected_commit = "5fe715bdfba4ff3e33d901d69dfa220f1222c060"
commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
branch = subprocess.check_output(
    ["git", "-C", str(repo), "branch", "--show-current"],
    text=True,
).strip()

assert branch == "proposed-certificate-comm", (
    f"Unexpected proposed branch: {branch}"
)
assert commit != expected_commit, (
    "The evaluator is using the frozen baseline commit instead "
    "of the proposed worktree."
)

tracked_diff = subprocess.check_output(
    ["git", "-C", str(repo), "diff", "HEAD", "--"], text=True
)
assert not tracked_diff, "Tracked source changes detected; stop and inspect."

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

filenames = [
    "reward6.03_policy.pth",
    "reward6.03_critic.pth",
    "reward6.03_data.json",
]

source_hashes = {}
for name in filenames:
    source = source_model / name
    assert source.is_file(), f"Missing author file: {source}"
    source_hashes[name] = sha256(source)
    shutil.copy2(source, model / name)

# Keep an untouched copy outside the evaluator's model directory.
shutil.copy2(
    source_model / "reward6.03_data.json",
    run / "author_original_data.json",
)

data_path = model / "reward6.03_data.json"
data = json.loads(data_path.read_text())
assert data["parameters"]["is_add_noise"] is True
data["parameters"]["where_to_save"] = str(model) + "/"
data_path.write_text(json.dumps(data, indent=2))

manifest = {
    "status": "started",
    "git_commit": commit,
    "checkpoint": "reward6.03_policy.pth",
    "source_sha256": source_hashes,
    "scenario": "intersection_2",
    "agents": 6,
    "parallel_simulations": 32,
    "configured_steps": 1200,
    "evaluation_seed": 0,
    "seed_note": "Explicit seed for this standalone M0 evaluation.",
    "fresh_rollout": True,
    "device": "cpu",
}

def save_manifest():
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))

save_manifest()
start = time.time()

try:
    # The original code resolves map assets relative to the repository.
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    from torchrl.objectives import ClipPPOLoss
    from utilities.evaluation_base import Evaluation
    from utilities.constants import SCENARIOS

    # Upstream imports set torch's seed: reseed after those imports.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    print("RUN DIRECTORY:", run, flush=True)
    print("POLICY SHA256:", source_hashes["reward6.03_policy.pth"])
    print("ClipPPOLoss signature:", inspect.signature(ClipPPOLoss.__init__))

    assert SCENARIOS["intersection_2"]["n_agents"] == 6

    evaluator = Evaluation(
        scenario_type="intersection_2",
        model_paths=[str(model) + "/"],
        fitst_model_index=0,
        num_agents=6,
        simulation_steps=1200,
        num_simulations_per_model=32,
        is_render=False,
        is_save_simulation_video=False,
        legends=["Author M0"],
        render_titles=["Author M0"],
        video_names=["author_m0"],
        where_to_save_eva_results=str(run / "figures"),
        where_to_save_logging=str(run / "native_summary.txt"),
    )

    # Invoke the original evaluation path for one selected model.
    evaluator.model_i = 0
    evaluator.model_idx = 0
    evaluator.model_i_path = str(model) + "/"
    evaluator._load_parameters()
    communication_mode = os.environ.get(
        "PAPER2_COMMUNICATION_MODE",
        "certificate",
    ).lower()

    allowed_communication_modes = {
        "perfect",
        "periodic",
        "age",
        "certificate",
    }

    if communication_mode not in allowed_communication_modes:
        raise ValueError(
            f"Unsupported communication mode: {communication_mode}"
        )

    evaluator.parameters.communication_mode = communication_mode
    evaluator.parameters.communication_period_steps = None
    evaluator.parameters.communication_max_age_steps = None
    evaluator._adjust_parameters()

    (run / "effective_evaluation_parameters.json").write_text(
        json.dumps(evaluator.parameters.to_dict(), indent=2)
    )

    assert not list(model.glob("*_out_td_*.pth")), \
        "Unexpected cached rollout in the fresh run directory."

    print("Starting fresh author M0 simulation...", flush=True)
    td = evaluator._get_simulation_outputs()

    communication_statistics_path = (
        model / "communication_statistics.json"
    )
    assert communication_statistics_path.is_file(), (
        f"Missing communication sidecar: "
        f"{communication_statistics_path}"
    )

    communication_statistics = json.loads(
        communication_statistics_path.read_text()
    )

    assert communication_statistics["mode"] == communication_mode
    assert communication_statistics["active"] is True

    if communication_mode == "perfect":
        assert communication_statistics["cache_active"] is False
    else:
        assert communication_statistics["cache_active"] is True

    if communication_mode == "certificate":
        assert communication_statistics["certificate_evaluations"] > 0

    assert communication_statistics["messages_sent"] > 0

    manifest["communication_mode"] = communication_mode
    manifest["communication_statistics"] = communication_statistics

    (run / "communication_statistics.json").write_text(
        json.dumps(
            communication_statistics,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "COMMUNICATION STATISTICS:",
        json.dumps(
            communication_statistics,
            indent=2,
            sort_keys=True,
        ),
    )

    # Read the same current-state info fields as the original evaluator.
    positions = td["agents", "info", "pos"].detach().cpu()
    velocities = td["agents", "info", "vel"].detach().cpu()
    aa = td["agents", "info", "is_collision_with_agents"].detach().cpu()
    al = td["agents", "info", "is_collision_with_lanelets"].detach().cpu()
    deviation = td["agents", "info", "distance_ref"].detach().cpu()

    print("Position shape:", tuple(positions.shape))
    print("Observation shape:", tuple(td["agents", "observation"].shape))
    print("Action shape:", tuple(td["agents", "action"].shape))

    batch, steps, agents = positions.shape[:3]
    assert (batch, steps, agents) == (32, 1199, 6), \
        f"Unexpected rollout dimensions: {(batch, steps, agents)}"

    assert torch.isfinite(positions).all()
    assert torch.isfinite(velocities).all()
    assert torch.isfinite(deviation).all()

    aa_frames = aa.bool().reshape(batch, steps, agents).any(dim=-1)
    al_frames = al.bool().reshape(batch, steps, agents).any(dim=-1)

    aa_pct = 100 * aa_frames.float().mean(dim=1)
    al_pct = 100 * al_frames.float().mean(dim=1)
    union_pct = 100 * (aa_frames | al_frames).float().mean(dim=1)
    speed = velocities.norm(dim=-1).mean(dim=(1, 2))
    cd_cm = 100 * deviation.reshape(batch, steps, agents).mean(dim=(1, 2))

    rows = []
    for i in range(batch):
        rows.append({
            "simulation_index": i,
            "actual_steps": steps,
            "agents": agents,
            "aa_collision_frames_pct": aa_pct[i].item(),
            "al_collision_frames_pct": al_pct[i].item(),
            "aa_plus_al_pct": (aa_pct[i] + al_pct[i]).item(),
            "collision_union_frames_pct": union_pct[i].item(),
            "mean_speed_m_s": speed[i].item(),
            "mean_centerline_deviation_cm": cd_cm[i].item(),
        })

    csv_path = run / "per_simulation_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "aggregation": "Untrimmed mean and sample SD across 32 simulations",
        "note": "Not the native trimmed/relative-unit median summary.",
        "metrics": {},
    }

    for name in list(rows[0])[3:]:
        values = np.array([row[name] for row in rows], dtype=float)
        summary["metrics"][name] = {
            "mean": float(values.mean()),
            "sd_across_simulations": float(values.std(ddof=1)),
        }

    (run / "summary_untrimmed.json").write_text(
        json.dumps(summary, indent=2)
    )

    rollouts = list(model.glob("*_out_td_*.pth"))
    assert len(rollouts) == 1, f"Expected one saved rollout, found {rollouts}"

    manifest.update({
        "status": "completed",
        "actual_steps": steps,
        "observation_shape": list(td["agents", "observation"].shape),
        "action_shape": list(td["agents", "action"].shape),
        "raw_rollout": str(rollouts[0]),
        "raw_rollout_sha256": sha256(rollouts[0]),
    })

    (run / "pip-freeze.txt").write_text(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
    )

    print(json.dumps(summary, indent=2))
    print("PASS: fresh author M0 rollout and numerical extraction completed.")

except BaseException as exc:
    manifest["status"] = "failed"
    manifest["error"] = repr(exc)
    raise
finally:
    manifest["duration_seconds"] = time.time() - start
    save_manifest()
