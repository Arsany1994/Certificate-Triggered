# Recorded environment and installation

## Scope

The experiments used Python 3.9.25 and CPU-only PyTorch 2.1.0+cpu
on Linux x86-64. The exports capture the environment at archive preparation.

The existing experiment environment passed pip check. Relocated evaluation
and analysis were tested in that environment. The installation procedure
below has not yet been validated in a fresh environment.

## Export files

- conda-explicit-linux-64.txt: exact Conda package builds and download URLs.
  This file does not include the pip-installed experiment dependencies.
- environment-original.yml: original Conda YAML, including pip packages,
  the original environment name, local prefix, and PYTHONPATH setting.
- requirements-lock.txt: original Python package inventory.
- requirements-portable.txt: the same inventory with the local pip package
  reference replaced by pip==26.0.1.
- runtime-details.txt, cpu-information.txt, pip-check.txt: recorded runtime,
  hardware, and dependency-check information.

The portable requirements file pins versions but does not contain wheel
hashes. Installation requires the recorded packages to remain available.

## Installation on Linux x86-64

Install Conda first. Choose a new environment name; do not reuse the
original experiment environment.

Set the repository location for your machine:

    CT_REPO="/home/arsany/Certificate-Triggered"

Create the Conda portion from the explicit export:

    conda create --name certificate-triggered-repro --file "$CT_REPO/environment/conda-explicit-linux-64.txt"

Activate it and clear inherited Python path overrides:

    conda activate certificate-triggered-repro
    unset PYTHONPATH

Install the recorded CPU PyTorch build without resolving dependencies yet:

    python -m pip install --no-deps "torch==2.1.0+cpu" --index-url https://download.pytorch.org/whl/cpu

Install the full pinned Python package inventory from PyPI:

    python -m pip install --index-url https://pypi.org/simple -r "$CT_REPO/environment/requirements-portable.txt"

The installed CPU PyTorch build satisfies the torch pin in that file.
The other pinned packages provide its dependencies.

Run each command separately and stop if a command fails. Do not remove
version pins to work around a failure without recording the change.

## System tools

On Ubuntu, install the tools used for repository handling and replay:

    sudo apt install git ffmpeg

Confirm that git, ffmpeg, and ffprobe are available:

    command -v git
    command -v ffmpeg
    command -v ffprobe

Video rendering uses the libx264 encoder:

    ffmpeg -hide_banner -encoders | grep libx264

System tools are not included in the Conda or pip exports.

## Verify the installation

Check dependency consistency:

    python -m pip check

Print the Python and core package versions:

    python -c 'import sys; from importlib.metadata import version; print(sys.version); print({p: version(p) for p in ("torch", "torchrl", "tensordict", "vmas", "numpy", "gym")})'

Expected core versions:

| Package | Version |
|---|---|
| Python | 3.9.25 |
| torch | 2.1.0+cpu |
| torchrl | 0.2.1 |
| tensordict | 0.2.1 |
| vmas | 1.4.1 |
| numpy | 1.26.1 |
| gym | 0.26.2 |

Check core imports and a CPU tensor operation:

    python -c 'import torch, torchrl, tensordict, vmas, numpy, gym; print(torch.ones(2).sum()); print("CUDA build:", torch.version.cuda)'

Expected tensor result: tensor(2.).
Expected CUDA build: None.

Successful installation and imports establish environment setup.
Numerical reproducibility must be checked separately using the archived
evaluation or analysis inputs.

## References

Official PyTorch installation instructions for previous releases:
https://pytorch.org/get-started/previous-versions/

CPU wheel index:
https://download.pytorch.org/whl/cpu
