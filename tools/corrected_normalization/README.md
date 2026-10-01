# Corrected-normalization evaluation and analysis

Experimental source commit:
363881d5cde7e39a0b218fea44936324382e55de

This directory archives:
- The 63-run clean comparison runner.
- The clean comparison aggregation script.
- The 36-run impairment runner, reusing 12 clean references.
- The impairment aggregation and figure builder.
- The three-seed training-reward figure builder.
- Both frozen evaluation plans.

The original working scripts remain in:
 /home/arsany/paper2/corrected_normalization/development/scripts

Results remain outside Git in:
 /home/arsany/paper2/corrected_normalization/runs

## Provenance

The commit containing this archive is an analysis/archive commit.
It is not the commit at which the experiments were executed.
Do not rewrite historical run manifests or frozen plans to identify
the archive commit as the experimental source commit.

## Re-execution

The evaluation runners enforce the experimental source commit.
They will intentionally refuse execution from a later archive commit.
Use a separate worktree at the recorded experimental commit and run
the external working script to reproduce the original execution setup.

The scripts and plans contain machine-specific absolute paths.
They preserve this experiment's invocation but are not a portable
installation package. Adapt paths through a separately documented
configuration when reproducing on another machine.

## Statistics

Performance is summarized across three training-seed means.
Each evaluation run contains 32 simulations.
Across-seed sample SD is not a confidence interval.
The training figure uses unsmoothed rewards rounded in the logs.

## Scope

This archive covers the listed evaluation and analysis scripts.
It does not itself audit all training/calibration tooling, dependencies,
raw artifacts, or manuscript claims.
