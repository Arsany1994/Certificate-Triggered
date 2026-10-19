# Certificate-Triggered

Reproducibility materials for certificate-triggered neighbor-information
refresh in cooperative multi-agent driving.

## Supplementary Videos S1–S8

Recorded trajectories under clean-channel, certificate-triggered
communication. Videos S1–S4 show favorable examples and S5–S8 show
adverse examples selected using collision-union frame percentage.
Simulation indices are zero-based.

Each replay displays all 1,199 recorded frames at 20 frames/s without
rerunning the policy. These selected examples illustrate vehicle
interactions; the quantitative results use all 32 simulations per
evaluation run across three training seeds.

[Download the videos, supporting data, and replay inputs from release v1.0.0](https://github.com/Arsany1994/Certificate-Triggered/releases/tag/v1.0.0).

### Video S1 — CPM_mixed (favorable)

4 agents · Training seed 2 · Simulation 02

https://github.com/user-attachments/assets/54390747-a1bf-4749-82e1-c5304547b1c6

### Video S2 — intersection_2 (favorable)

6 agents · Training seed 2 · Simulation 16

https://github.com/user-attachments/assets/d11d3dab-66ee-45d7-8be3-3e7b6ae4898b

### Video S3 — on_ramp_1 (favorable)

8 agents · Training seed 0 · Simulation 00

https://github.com/user-attachments/assets/17126a81-8354-4e99-9154-50555c81420b

### Video S4 — roundabout_1 (favorable)

8 agents · Training seed 2 · Simulation 26

https://github.com/user-attachments/assets/97b3d1e2-5c1a-4506-94ee-acae6d6dedef

### Video S5 — CPM_mixed (adverse)

4 agents · Training seed 0 · Simulation 20

https://github.com/user-attachments/assets/c603082c-ab8b-4b65-8bae-e1c4077d8b17

### Video S6 — intersection_2 (adverse)

6 agents · Training seed 1 · Simulation 15

https://github.com/user-attachments/assets/554f2999-2899-4f7f-8757-45ad89763c9d

### Video S7 — on_ramp_1 (adverse)

8 agents · Training seed 0 · Simulation 23

https://github.com/user-attachments/assets/f5a4ec79-23fb-47af-94f9-6cf2bb16a03a

### Video S8 — roundabout_1 (adverse)

8 agents · Training seed 1 · Simulation 27

https://github.com/user-attachments/assets/00dbd10a-6fb8-4e11-aa05-b097c70d82de

## Contribution and upstream framework

This repository contains the SigmaRL framework together with the
experimental communication implementation and reproduction tools.
It is a derivative research repository, not an official SigmaRL release.

SigmaRL supplies the underlying multi-agent driving framework, shared-policy
MAPPO baseline, simulator integration, traffic scenarios, and baseline
observation and reward definitions. The contribution studied here is the
certificate-triggered communication mechanism: timestamped neighbor caches,
a clearance calculation using the last delivered scalar inter-agent distance
and an age-dependent uncertainty allowance, and selective refresh requests.

The communication mechanism determines information updates used by the
actor. The existing policy supplies the driving actions.

The original upstream README, including its references and usage guidance,
is preserved in [README-SigmaRL.md](README-SigmaRL.md).
The upstream copyright and MIT license are preserved in
[LICENSE.txt](LICENSE.txt).

Upstream repository recorded by this archive:
https://github.com/bassamlab/SigmaRL

## Recorded source versions

| Role | Version or commit |
|---|---|
| SigmaRL baseline | Version 1.2.0 |
| Baseline commit | 5fe715bdfba4ff3e33d901d69dfa220f1222c060 |
| Experimental implementation | 363881d5cde7e39a0b218fea44936324382e55de |
| Evaluation and analysis archive | 94452411f5535342a6ddbb84a81beb4c8ba8f32e |

The experimental commit identifies the implementation used to generate
the corrected-normalization results. The later archive commit adds the
associated evaluation and analysis scripts and frozen plans.

Release preparation adds documentation and portable reproduction tooling.
It does not change the historical experimental commit or rewrite the
original run records.

## Archive contents

The source repository contains:

- The SigmaRL-derived experimental source and Git history.
- Historical scripts and plans under tools/corrected_normalization/.
- Portable launchers under tools/reproduction/.
- Historical video replay scripts under tools/video_replay/.
- Environment exports under environment/.

The separately packaged supplementary directories contain:

| Directory | Contents |
|---|---|
| data_payload | Selected checkpoints, training configurations and logs, frozen plans, run-level results, per-run evidence, and reproduction validation records |
| video_payload | Videos S1–S8, frame data, original replay metadata, and filename mapping |
| replay_payload | Seven original full-batch rollouts and the run records needed to replay the eight videos |

S3 and S7 select different simulations from the same rollout.
Large binary assets are packaged separately from Git source.

## Experimental environment

The recorded CPU environment uses:

| Component | Version |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.1.0+cpu |
| TorchRL | 0.2.1 |
| TensorDict | 0.2.1 |
| VMAS | 1.4.1 |
| NumPy | 1.26.1 |
| Gym | 0.26.2 |

Environment exports document the installed environment. The Conda explicit
export alone does not capture the pip-installed packages. Original exports
are retained; requirements-portable.txt replaces the machine-local pip
package reference with its recorded version.

Fresh-environment installation has not yet been validated.

## Reproduce evaluation

tools/reproduction/run_job.py selects a job from a frozen evaluation plan,
verifies its checkpoint hashes, and executes the historical evaluator from
a separate source checkout.

The source checkout must be at experimental commit
363881d5cde7e39a0b218fea44936324382e55de on branch corrected-normalization.
The launcher defaults to a dry run; evaluation requires --execute.

Inspect the available arguments:

    python tools/reproduction/run_job.py --help

The training-root layout expected by the evaluator is:

    seed0/model/reward5.04_policy.pth
    seed0/model/reward5.04_critic.pth
    seed0/model/reward5.04_data.json

Use corresponding seed1/reward5.31 and seed2/reward5.57 files.
The supplied checkpoint files are under data_payload/checkpoints/seedN;
copy or link each seed directory into the required model location.

Historical runners contain original machine-specific paths. Use the
portable launcher for relocated execution.

## Reproduce analysis

With the experiment environment active, run:

    python tools/reproduction/run_analysis.py --data /absolute/path/to/data_payload --output /absolute/path/to/new-analysis-output

The output directory must not already exist and must be outside the
repository and data_payload.

The launcher verifies historical script hashes and creates separate
adapted copies that relocate paths. Original scripts, frozen plans, and
input CSVs remain unchanged. It records the adaptations and execution
status and produces clean-comparison tables, impairment tables and figures,
and the training-reward table and figure.

## Replay supplementary videos

See [video replay instructions](tools/video_replay/README.md).
Replaying saved trajectories does not require another policy evaluation.

The replay scripts and original rollout hashes are preserved. Encoded
video bytes can differ across rendering and encoder versions.

## Validation performed during archive preparation

On the original experiment machine and environment:

- One relocated clean evaluation job, CPM_mixed_certificate_seed0,
  matched all 32 per-simulation metric rows, its summary, and communication
  statistics exactly.
- All seven generated analysis CSV tables matched the original tables.
  Historical run-directory strings remained unchanged.
- All eight archived videos passed provenance, output-hash, and encoded
  frame-count checks.
- The seven source rollouts required for video replay were copied and
  verified against their recorded hashes.

This records the checks actually performed. It does not claim that every
evaluation job or a fresh environment installation was rerun.

## Statistical reporting

The clean comparison contains 63 evaluation runs. The impairment analysis
contains 36 impaired runs and 12 reused clean references, giving 48 rows.

Each evaluation run contains 32 simulations. Reported across-seed
statistics summarize three training-seed means; sample standard deviation
is not a confidence interval.

## Historical paths and provenance

Absolute paths in original plans, manifests, and CSVs document the original
execution. They are retained unchanged. Portable launchers handle relocated
inputs separately and record the new execution paths.

The upstream README describes SigmaRL usage. For this study's corrected
experiments, use the pinned commits, archived plans, selected checkpoints,
and reproduction tools described here.
