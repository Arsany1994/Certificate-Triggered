#!/usr/bin/env python3
"""Animate a trusted, completed SigmaRL rollout without running a simulation.

Requires the original Torch/TensorDict environment, Matplotlib, and ffmpeg.
The .pth file is loaded as a trusted local Python pickle. Only load your own
experiment files. No policy, simulator, environment, or evaluator is created.

Each video frame uses one recorded frame from agents/info, in order. There is
no interpolation, trajectory smoothing, physics stepping, or collision
recalculation. Normalization is reversed using the recorded source version and evaluation-scenario world dimensions.
The map parser and map asset are read from that same Git commit. Outputs are
written to a NEW directory outside the original run and repository.

The recorded global lane-boundary flag is displayed at environment level.
Agent-agent collision flags are displayed per agent. The video reproduces
saved state samples at their stored precision, not original rendered pixels.
"""

import argparse
import ast
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def git_blob(repo, commit, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Unsafe repository-relative path: " + relative)
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", commit + ":" + relative]
    )


def constant_dict(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                result = ast.literal_eval(node.value)
                if not isinstance(result, dict):
                    break
                return result
    raise ValueError("Cannot read literal dictionary " + name)


def expression_id(text):
    return ast.dump(ast.parse(text, mode="eval").body, include_attributes=False)


def verify_source_contract(source):
    """Fail rather than silently guessing when saved-state semantics change."""
    tree = ast.parse(source)
    info = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "info"), None)
    if info is None:
        raise ValueError("The recorded source has no info() function.")
    fields = {}
    for node in ast.walk(info):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    fields[key.value] = ast.dump(value, include_attributes=False)
    expected = {
        "pos": "agent.state.pos / self.normalizers.pos_world",
        "rot": "angle_eliminate_two_pi(agent.state.rot) / self.normalizers.rot",
        "vel": "agent.state.vel / self.normalizers.v",
    }
    for name, expression in expected.items():
        if fields.get(name) != expression_id(expression):
            raise ValueError("Unrecognized saved " + name + " normalization; inspect source.")
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assignments[ast.dump(target, include_attributes=False)] = node.value

    def assigned(target):
        # Assignment targets use Store context rather than Load context.
        parsed = ast.parse(target + " = None").body[0].targets[0]
        key = ast.dump(parsed, include_attributes=False)
        if key not in assignments:
            raise ValueError("Missing recorded assignment: " + target)
        return assignments[key]

    for target, expression in {
        "self.world_x_dim": 'SCENARIOS[scenario_type]["world_x_dim"]',
        "self.world_y_dim": 'SCENARIOS[scenario_type]["world_y_dim"]',
        "self.agent_length": 'AGENTS["length"]',
        "self.agent_width": 'AGENTS["width"]',
        "self.max_speed": 'kwargs.pop("max_speed", AGENTS["max_speed"])',
    }.items():
        if ast.dump(assigned(target), include_attributes=False) != expression_id(expression):
            raise ValueError("Unrecognized recorded assignment: " + target)
    norm = assigned("self.normalizers")
    if not isinstance(norm, ast.Call):
        raise ValueError("Unrecognized normalizer construction.")
    kwargs = {k.arg: k.value for k in norm.keywords}
    for name, expression in {
        "pos_world": "[self.world_x_dim, self.world_y_dim]",
        "rot": "2 * torch.pi", "v": "self.max_speed",
    }.items():
        value = kwargs.get(name)
        if (not isinstance(value, ast.Call) or not value.args
                or ast.dump(value.func, include_attributes=False) != expression_id("torch.tensor")
                or ast.dump(value.args[0], include_attributes=False) != expression_id(expression)):
            raise ValueError("Unrecognized normalizer: " + name)
        dtype = next((k.value for k in value.keywords if k.arg == "dtype"), None)
        if dtype is None or ast.dump(dtype, include_attributes=False) != expression_id("torch.float32"):
            raise ValueError("Expected float32 recorded normalizer: " + name)
    lane_assignment = next((n.value for n in ast.walk(info)
                            if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "is_collision_with_lanelets"
                                    for t in n.targets)), None)
    if (lane_assignment is None or ast.dump(lane_assignment, include_attributes=False)
            != expression_id("self.collisions.with_lanelets.any(dim=-1)")):
        raise ValueError("Lane-collision scope differs from the inspected implementation.")


def load_saved_arrays(rollout, simulation):
    import torch
    import tensordict  # Required for TensorDict checkpoint deserialization.

    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    td = torch.load(rollout, **options)
    info = td["agents", "info"]
    pos = info["pos"]
    if pos.ndim != 4 or pos.shape[-1] != 2:
        raise ValueError("Expected position shape (simulations, frames, agents, 2).")
    batch, count, agents, _ = pos.shape
    if not 0 <= simulation < batch:
        raise ValueError("Simulation index is outside the stored batch.")
    arrays = {}
    for key, width in (("pos", 2), ("rot", 1), ("vel", 2),
                       ("is_collision_with_agents", 1),
                       ("is_collision_with_lanelets", 1)):
        tensor = info[key]
        if tuple(tensor.shape) != (batch, count, agents, width):
            raise ValueError("Unexpected shape for " + key + ": " + str(tuple(tensor.shape)))
        value = tensor[simulation].detach().cpu().numpy().copy()
        if not np.isfinite(value).all():
            raise ValueError("Non-finite recorded field: " + key)
        arrays[key] = value
    for key in ("pos", "rot", "vel"):
        if arrays[key].dtype != np.float32:
            raise ValueError("Expected float32 recorded poses and velocities.")
    for key in ("is_collision_with_agents", "is_collision_with_lanelets"):
        if not np.isin(arrays[key], [0, 1]).all():
            raise ValueError("Non-binary recorded collision flag: " + key)
        arrays[key] = arrays[key][..., 0].astype(bool)
    lane = arrays["is_collision_with_lanelets"]
    if not np.all(lane == lane[:, :1]):
        raise ValueError("Lane flags are not repeated globally as the recorded source specifies.")
    del td, info
    return arrays, int(batch), {
        "torch": torch.__version__,
        "tensordict": getattr(tensordict, "__version__", "unknown"),
    }


def world_arrays(saved, position_scale, max_speed):
    scales = np.asarray(position_scale, dtype=np.float32)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid position normalizers.")
    # Float32 multiplication mirrors the precision of the recorded tensors.
    positions = saved["pos"] * scales
    headings = saved["rot"][..., 0] * np.float32(2 * math.pi)
    velocity = saved["vel"] * np.float32(max_speed)
    return positions, headings, velocity


def calibrate_position_scale(saved, lanes, scenarios):
    """Infer the position normalizer from the archived lane geometry.

    Some historical evaluator checkpoints retain the training environment's
    ``pos_world`` normalizer while being evaluated on another map.  In that
    case using the target scenario dimensions shifts every replayed vehicle.
    The saved initial states are sampled from lane centerlines, so the correct
    scale is identifiable without rerunning the policy.
    """
    centers = [lane["center_line"] for lane in lanes]
    samples = saved["pos"][:: max(1, saved["pos"].shape[0] // 12)]
    samples = samples.reshape(-1, 2)
    candidates = []
    for data in scenarios.values():
        scale = (float(data["world_x_dim"]), float(data["world_y_dim"]))
        if scale not in candidates:
            candidates.append(scale)
    scores = []
    for scale in candidates:
        points = samples * np.asarray(scale, dtype=np.float32)
        distances = []
        for point in points:
            distances.append(min(
                float(np.linalg.norm(center - point, axis=1).min())
                for center in centers
            ))
        scores.append((float(np.median(distances)), float(np.mean(distances)), scale))
    scores.sort()
    best = scores[0]
    if best[0] > 0.15:
        raise ValueError(
            "Could not calibrate saved position normalization against the "
            "archived lane geometry; best median centerline distance is "
            f"{best[0]:.3f} m. Candidate scores: {scores}"
        )
    return best[2], scores


def archived_map(repo, commit, scenario_name, scenario):
    """Instantiate only the historical map parser, from a private snapshot."""
    parser_file = "utilities/parse_xml.py" if "CPM" in scenario_name else "utilities/parse_osm.py"
    files = ["utilities/constants.py", "utilities/colors.py",
             "utilities/helper_scenario.py", "utilities/parse_map_base.py",
             parser_file, scenario["map_path"]]
    blobs = {name: git_blob(repo, commit, name) for name in files}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in blobs.items()}
    old_directory, old_path = Path.cwd(), sys.path[:]
    old_modules = {name: value for name, value in sys.modules.items()
                   if name == "utilities" or name.startswith("utilities.")}
    with tempfile.TemporaryDirectory(prefix="paper2_replay_map_") as folder:
        root = Path(folder)
        for name, content in blobs.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (root / "utilities" / "__init__.py").write_text("")
        try:
            for name in old_modules:
                del sys.modules[name]
            sys.path.insert(0, str(root))
            os.chdir(root)
            import matplotlib
            with matplotlib.rc_context():
                module = importlib.import_module(parser_file[:-3].replace("/", "."))
                cls = module.ParseXML if "CPM" in scenario_name else module.ParseOSM
                parser = cls(scenario_type=scenario_name, device="cpu",
                             is_visualize_map=False, is_plt_show=False, is_save_fig=False,
                             is_visualize_random_agents=False)
                lanes = []
                for lane in parser.lanelets_all:
                    lanes.append({key: lane[key].detach().cpu().numpy().copy()
                                  for key in ("center_line", "left_boundary", "right_boundary")})
        finally:
            os.chdir(old_directory)
            sys.path[:] = old_path
            for name in list(sys.modules):
                if name == "utilities" or name.startswith("utilities."):
                    del sys.modules[name]
            sys.modules.update(old_modules)
    if not lanes:
        raise ValueError("The archived map parser produced no lanelets.")
    for lane in lanes:
        for points in lane.values():
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError("Invalid archived map geometry.")
    return lanes, hashes


def rectangle_vertices(position, heading, length, width):
    local = np.array([[length / 2, width / 2], [-length / 2, width / 2],
                      [-length / 2, -width / 2], [length / 2, -width / 2]])
    cosine, sine = math.cos(float(heading)), math.sin(float(heading))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + position


def write_frame_csv(path, saved, positions, headings, velocities, indices, simulation, dt):
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_frame", "recorded_frame", "simulation_index", "agent_index",
                         "time_since_first_recorded_frame_s", "saved_x_normalized",
                         "saved_y_normalized", "saved_heading_normalized", "x_m", "y_m",
                         "heading_rad", "vx_m_s", "vy_m_s", "speed_m_s",
                         "recorded_agent_agent_collision", "recorded_environment_lane_collision"])
        for video_frame, frame in enumerate(indices):
            for agent in range(positions.shape[1]):
                writer.writerow([
                    video_frame, frame, simulation, agent, frame * dt,
                    *map(float, saved["pos"][frame, agent]),
                    float(saved["rot"][frame, agent, 0]),
                    *map(float, positions[frame, agent]), float(headings[frame, agent]),
                    *map(float, velocities[frame, agent]),
                    float(np.linalg.norm(velocities[frame, agent])),
                    int(saved["is_collision_with_agents"][frame, agent]),
                    int(saved["is_collision_with_lanelets"][frame, 0]),
                ])


def render_video(path, poster, positions, headings, velocities, aa, al, lanes,
                 scenario, agents, dt, indices, simulation, label, playback_speed):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Polygon

    if not FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg is unavailable in the active environment.")
    count = positions.shape[1]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.usetex": False, "axes.grid": False}):
        fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="#ffffff")
        ax = fig.add_axes([0.055, 0.105, 0.61, 0.77])
        panel = fig.add_axes([0.70, 0.105, 0.28, 0.77])
        panel.axis("off")
        fig.suptitle("Recorded trajectory playback", x=0.055, ha="left", y=0.974,
                     fontsize=20, fontweight="bold", color="#16233b")
        fig.text(0.055, 0.925, label, fontsize=12, color="#536078")
        ax.set_facecolor("#eef2f6")
        for lane in lanes:
            polygon = np.concatenate([lane["left_boundary"], lane["right_boundary"][::-1]])
            ax.add_patch(Polygon(polygon, closed=True, facecolor="white", edgecolor="none", zorder=0))
        for lane in lanes:
            for key in ("left_boundary", "right_boundary"):
                points = lane[key]
                ax.plot(points[:, 0], points[:, 1], color="#8591a3", lw=0.7, alpha=0.6, zorder=1)
            points = lane["center_line"]
            ax.plot(points[:, 0], points[:, 1], color="#b3bac6", lw=0.6,
                    linestyle=(0, (4, 5)), alpha=0.7, zorder=1)
        all_points = np.vstack([p for lane in lanes for p in lane.values()]
                               + [positions.reshape(-1, 2)])
        margin = max(float(agents["length"]), 0.1)
        lower = all_points.min(axis=0) - margin
        upper = all_points.max(axis=0) + margin
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.tick_params(labelsize=9, colors="#536078")
        for spine in ax.spines.values():
            spine.set_color("#c7cfdb")
        palette = ["#2763ae", "#128479", "#7558a7", "#ac5687", "#367780", "#74643a"]
        bodies, labels, noses = [], [], []
        for i in range(count):
            body = Polygon(np.zeros((4, 2)), closed=True, facecolor=palette[i % len(palette)],
                           edgecolor="#17283e", linewidth=1.0, zorder=4)
            ax.add_patch(body)
            bodies.append(body)
            labels.append(ax.text(0, 0, str(i), ha="center", va="center", fontsize=8,
                                  color="white", fontweight="bold", zorder=6))
            noses.append(ax.plot([], [], color="white", linewidth=1.4, zorder=5)[0])
        panel.text(0, 1, "SAVED FRAME", fontsize=10, color="#68758b", va="top")
        clock = panel.text(0, 0.955, "", va="top", fontsize=18, fontweight="bold", color="#16233b")
        panel.text(0, 0.855, "RECORDED COLLISION FLAGS", fontsize=10, color="#68758b", va="top")
        aa_text = panel.text(0, 0.800, "", va="top", fontsize=11)
        al_text = panel.text(0, 0.690, "", va="top", fontsize=11)
        panel.text(0, 0.572, "RECORDED SPEEDS", fontsize=10, color="#68758b", va="top")
        speed_text = panel.text(0, 0.522, "", va="top", fontsize=11,
                                family="DejaVu Sans Mono", linespacing=1.5, color="#263a51")
        union_pct = 100.0 * np.logical_or(aa.any(axis=1), al[:, 0]).mean()
        panel.text(0, 0.22, "Whole saved simulation\nCollision-union frames: %.3f%%" % union_pct,
                   va="top", fontsize=10, color="#536078", linespacing=1.6)
        panel.text(0, 0.105, "Red outline: agent-agent flag\nLane flag: environment level\nAgent IDs are zero-based",
                   va="top", fontsize=9, color="#536078", linespacing=1.6)
        fig.text(0.055, 0.045,
                 "Saved samples only | No interpolation | No policy execution | Playback %.2gx" % playback_speed,
                 fontsize=9, color="#536078")

        def draw(frame):
            for i, body in enumerate(bodies):
                xy, theta = positions[frame, i], headings[frame, i]
                body.set_xy(rectangle_vertices(xy, theta, agents["length"], agents["width"]))
                body.set_edgecolor("#e22744" if aa[frame, i] else "#17283e")
                body.set_linewidth(3 if aa[frame, i] else 1)
                labels[i].set_position(xy)
                forward = np.array([math.cos(float(theta)), math.sin(float(theta))])
                start = xy + forward * agents["length"] * 0.22
                end = xy + forward * agents["length"] * 0.46
                noses[i].set_data([start[0], end[0]], [start[1], end[1]])
            clock.set_text("%d / %d\nElapsed %.2f s" % (frame, len(positions) - 1, frame * dt))
            ids = np.flatnonzero(aa[frame])
            aa_text.set_text("Agent-agent: " + ("YES\nIDs: " + ", ".join(map(str, ids)) if len(ids) else "no"))
            aa_text.set_color("#c61e3a" if len(ids) else "#263a51")
            al_text.set_text("Lane boundary: " + ("YES" if al[frame, 0] else "no") + "\n(environment flag)")
            al_text.set_color("#a65d06" if al[frame, 0] else "#263a51")
            speeds = np.linalg.norm(velocities[frame], axis=-1)
            speed_text.set_text("\n".join("Agent %d   %.3f m/s" % (i, v) for i, v in enumerate(speeds)))

        writer = FFMpegWriter(fps=playback_speed / dt, codec="libx264",
                              extra_args=["-crf", "18", "-pix_fmt", "yuv420p"],
                              metadata={"title": label, "comment": "Offline playback of saved samples."})
        try:
            draw(indices[0])
            fig.savefig(poster, dpi=100, facecolor=fig.get_facecolor())
            with writer.saving(fig, str(path), dpi=100):
                for offset, frame in enumerate(indices):
                    draw(frame)
                    writer.grab_frame(facecolor=fig.get_facecolor())
                    if offset == 0 or (offset + 1) % 200 == 0 or offset + 1 == len(indices):
                        print("RENDERED %d/%d saved frames" % (offset + 1, len(indices)), flush=True)
        finally:
            plt.close(fig)


def encoded_frame_count(path):
    if not shutil.which("ffprobe"):
        return None
    value = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)
    ], text=True))
    return int(value["streams"][0]["nb_read_frames"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; must not exist.")
    parser.add_argument("--simulation", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive; default includes all remaining frames.")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    args = parser.parse_args()
    repo, run, output = args.repo.resolve(), args.run.resolve(), args.output.resolve()
    if output == run or run in output.parents or output == repo or repo in output.parents:
        raise ValueError("Choose an output directory outside the run and repository.")
    if output.exists():
        raise FileExistsError("Output directory already exists; choose a new --output path: " + str(output))
    if not math.isfinite(args.playback_speed) or args.playback_speed <= 0:
        raise ValueError("Playback speed must be finite and positive.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not available in the active environment.")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("The source manifest does not report a completed run.")
    commit = manifest["git_commit"]
    if not (isinstance(commit, str) and 7 <= len(commit) <= 64
            and all(c in "0123456789abcdefABCDEF" for c in commit)):
        raise ValueError("Invalid recorded Git commit.")
    files = sorted((run / "model").glob("*_out_td_*.pth"))
    if len(files) != 1:
        raise ValueError("Expected exactly one saved rollout; found " + str(files))
    rollout = files[0]
    original_hash = digest_file(rollout)
    if original_hash != manifest.get("raw_rollout_sha256"):
        raise ValueError("Saved rollout SHA256 does not match its original manifest.")
    print("PASS: original rollout hash matches its manifest", flush=True)
    constants = git_blob(repo, commit, "utilities/constants.py").decode("utf-8")
    road_source = git_blob(repo, commit, "scenarios/road_traffic.py").decode("utf-8")
    verify_source_contract(road_source)
    scenario_name = manifest["scenario"]
    scenarios = constant_dict(constants, "SCENARIOS")
    scenario = scenarios[scenario_name]
    agents = constant_dict(constants, "AGENTS")
    params_path = run / "effective_evaluation_parameters.json"
    params = json.loads(params_path.read_text())
    params = params.get("parameters", params)
    if params.get("scenario_type") != scenario_name:
        raise ValueError("Manifest and effective parameters disagree on scenario.")
    dt = float(params["dt"])
    vmax = float(params.get("max_speed", agents["max_speed"]))
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(vmax) or vmax <= 0:
        raise ValueError("Invalid recorded dt or max_speed.")
    saved, batch_size, versions = load_saved_arrays(rollout, args.simulation)
    count, n_agents = saved["pos"].shape[:2]
    if n_agents != manifest["agents"] or n_agents != scenario["n_agents"]:
        raise ValueError("Agent counts disagree across saved data and configuration.")
    if batch_size != manifest["parallel_simulations"]:
        raise ValueError("Simulation count disagrees with the original manifest.")
    end = count if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end <= count:
        raise ValueError("Invalid requested frame interval.")
    indices = list(range(args.start_frame, end))
    print("Loading map geometry from recorded commit " + commit, flush=True)
    import matplotlib
    matplotlib.use("Agg", force=True)
    lanes, source_hashes = archived_map(repo, commit, scenario_name, scenario)
    source_hashes["scenarios/road_traffic.py"] = hashlib.sha256(road_source.encode("utf-8")).hexdigest()
    source_hashes["replay_script"] = digest_file(Path(__file__).resolve())
    position_scale = (
        float(scenario["world_x_dim"]),
        float(scenario["world_y_dim"]),
    )
    scale_scores = []
    positions, headings, velocities = world_arrays(saved, position_scale, vmax)
    print("PASS: saved-state conversions verified against recorded source", flush=True)
    print("POSITION SCALE SELECTED: [%.6g, %.6g]" % position_scale, flush=True)
    print("POSITION SCALE CANDIDATES: " + json.dumps([
        {"scale": list(scale), "median_centerline_distance_m": median,
         "mean_centerline_distance_m": mean}
        for median, mean, scale in scale_scores
    ]), flush=True)
    print("PASS: archived map loaded; no simulation was executed", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "replay_metadata.json"
    metadata = {
        "status": "started", "source_run": str(run), "source_rollout": str(rollout),
        "source_rollout_sha256": original_hash, "source_manifest_sha256": digest_file(manifest_path),
        "source_effective_parameters_sha256": digest_file(params_path),
        "source_git_commit": commit, "source_file_sha256": source_hashes,
        "scenario": scenario_name, "communication_mode": manifest.get("communication_mode"),
        "simulation_index": args.simulation, "original_simulations": batch_size,
        "original_frames": count, "agents": n_agents, "recorded_dt_s": dt,
        "first_recorded_frame": args.start_frame, "end_recorded_frame_exclusive": end,
        "video_frames": len(indices), "playback_speed": args.playback_speed,
        "video_fps": args.playback_speed / dt,
        "time_origin": "Zero is the first stored agents/info sample; no reset timestamp is inferred.",
        "position_scale": list(position_scale),
        "position_scale_source": "archived evaluation-scenario world dimensions; no map fitting",
        "position_scale_candidates": [
            {"scale": list(scale), "median_centerline_distance_m": median,
             "mean_centerline_distance_m": mean}
            for median, mean, scale in scale_scores
        ],
        "position_scale_float32": [float(np.float32(value))
                                    for value in position_scale],
        "heading_scale_float32": float(np.float32(2 * math.pi)), "velocity_scale": vmax,
        "vehicle_length_m": agents["length"], "vehicle_width_m": agents["width"],
        "state_key_prefix": ["agents", "info"], "interpolation": False,
        "new_simulation": False, "collision_recomputation": False,
        "lane_collision_scope": "Recorded environment-level flag, repeated per agent in the source.",
        "fidelity": "One frame per saved sample; normalization reversed at stored float32 precision.",
        "versions": dict(versions, numpy=np.__version__, matplotlib=matplotlib.__version__),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    video = output / "recorded_replay.mp4"
    partial = output / "recorded_replay.partial.mp4"
    frames_csv = output / "recorded_frame_data.csv"
    poster = output / "first_frame.png"
    try:
        write_frame_csv(frames_csv, saved, positions, headings, velocities, indices, args.simulation, dt)
        label = "%s | %s | simulation %d | %d agents" % (
            scenario_name, manifest.get("communication_mode", "unknown"), args.simulation, n_agents)
        render_video(partial, poster, positions, headings, velocities,
                     saved["is_collision_with_agents"], saved["is_collision_with_lanelets"],
                     lanes, scenario, agents, dt, indices, args.simulation, label, args.playback_speed)
        encoded_count = encoded_frame_count(partial)
        if encoded_count is not None and encoded_count != len(indices):
            raise RuntimeError("Encoded video frame count does not match the saved frame selection.")
        if digest_file(rollout) != original_hash:
            raise RuntimeError("The source rollout changed during rendering.")
        partial.rename(video)
        metadata.update(status="completed", source_unchanged=True,
                        ffprobe_frame_count=encoded_count,
                        output_sha256={p.name: digest_file(p) for p in (video, frames_csv, poster)})
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("PASS: rendered %d saved frames with no interpolation" % len(indices))
    print("PASS: original rollout remains unchanged")
    print("VIDEO: " + str(video))
    print("FRAME DATA: " + str(frames_csv))
    print("METADATA: " + str(metadata_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)

