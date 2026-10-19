# Supplementary video replay

Videos S1–S8 render saved evaluation trajectories. The scripts do not
execute the policy, step the simulator, interpolate trajectories, or
recompute collisions.

## Requirements

Use the archived experiment environment with PyTorch, TensorDict, NumPy,
Matplotlib, and the historical map-parser dependencies. Git, ffmpeg with
libx264, and ffprobe must be available.

The repository must contain experimental commit:
363881d5cde7e39a0b218fea44936324382e55de

The separate replay_payload archive supplies the original rollouts,
manifests, and effective evaluation parameters. Only load trusted .pth
files because these files use Python pickle deserialization.

A fresh-environment replay has not yet been validated.

## Input verification

From inside replay_payload, run:

    sha256sum -c SHA256SUMS

## Video assignments

Simulation indices are zero-based.

| Video | Scenario | Agents | Training seed | Simulation | Script suffix |
|---|---|---:|---:|---:|---|
| S1 | CPM_mixed | 4 | 2 | 2 | source_scale |
| S2 | intersection_2 | 6 | 2 | 16 | verified_scale |
| S3 | on_ramp_1 | 8 | 0 | 0 | verified_scale |
| S4 | roundabout_1 | 8 | 2 | 26 | verified_scale |
| S5 | CPM_mixed | 4 | 0 | 20 | source_scale |
| S6 | intersection_2 | 6 | 1 | 15 | verified_scale |
| S7 | on_ramp_1 | 8 | 0 | 23 | verified_scale |
| S8 | roundabout_1 | 8 | 1 | 27 | verified_scale |

S3 and S7 share a full-batch rollout but select different simulations.
Script hashes and original locations appear in script_provenance.json.

The source_scale script uses CPM scenario dimensions. The verified_scale
script uses the archived default CPM position normalizer with the
evaluated scenario's map. Neither calls its map-based calibration helper.
Preserve these script assignments for the archived videos.

## Replay commands

Activate the environment and set paths for your installation:

    conda activate paper2-sigmarl-itsc24
    CT_REPO="/home/arsany/Certificate-Triggered"
    CT_REPLAY="/home/arsany/Certificate-Triggered-archive/replay_payload"
    CT_OUTPUT="/home/arsany/Certificate-Triggered-archive/replayed-videos"

Define this Bash helper:

    replay_video() {
        python "$CT_REPO/tools/video_replay/animate_recorded_rollout_${2}.py" --repo "$CT_REPO" --run "$CT_REPLAY/runs/$3" --output "$CT_OUTPUT/$1" --simulation "$4" --playback-speed 1
    }

Run the desired command, checking successful completion before the next:

    replay_video S1 source_scale CPM_mixed_certificate_seed2 2
    replay_video S2 verified_scale intersection_2_certificate_seed2 16
    replay_video S3 verified_scale on_ramp_1_certificate_seed0 0
    replay_video S4 verified_scale roundabout_1_certificate_seed2 26
    replay_video S5 source_scale CPM_mixed_certificate_seed0 20
    replay_video S6 verified_scale intersection_2_certificate_seed1 15
    replay_video S7 verified_scale on_ramp_1_certificate_seed0 23
    replay_video S8 verified_scale roundabout_1_certificate_seed1 27

Each output directory must be new and outside the repository and input
run directories. Full replay uses 1,199 recorded frames at 20 frames/s.

## Outputs

Each replay produces:

- recorded_replay.mp4
- first_frame.png
- recorded_frame_data.csv
- replay_metadata.json

The scripts verify rollout hashes against the original manifests, retrieve
map assets and parser code from the recorded commit, check encoded frame
counts when ffprobe is available, and verify that the rollout is unchanged.

Compare regenerated frame CSVs with the corresponding files under
video_payload/videos/SN/. MP4 hashes may differ across rendering and
encoder versions; byte-identical video encoding is not assumed.

video_payload/video_manifest.csv records the distributed filenames and
source-run mapping. replay_payload/replay_input_manifest.json maps eight
videos to seven unique full-batch rollouts.

Historical absolute paths remain unchanged as provenance records.
Command-line arguments select relocated inputs without modifying them.
