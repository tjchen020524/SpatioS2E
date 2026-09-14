# Clean-room validation records

## Review follow-up

`review_followup_20260913.json` records the reviewed source commit and source
hash, source tests, the archived primary driver trained for six epochs on a
small synthetic panel, source-distribution contents, and a newly built wheel
installed into a fresh target directory. The wheel is imported and used for
synthetic fitting/evaluation outside the repository with isolated Python;
its version, import path and SHA-256 are recorded separately from the base
validation environment's distribution inventory. The pinned CPU environment
is reused, not described as newly created. No full cohort/GPU rerun is implied.

The validation report is committed after the source commit that it identifies;
its own addition does not change that tested source snapshot. A public tag,
release, DOI and source-data deposit remain pending author approval.

## Historical candidate validation

`submission_v1.1.0.json` records an earlier dirty v1.1.0 candidate, dependency checks,
tests, lint and distribution builds. It reuses the previously prepared pinned
CPU environment; it is not a claim that a new environment was created for
this revision. Tests include external-audit synthetic predictions, three-seed
aggregation, missing-condition rejection and signed MSE changes. No external
weights are loaded and no GPU training is performed by that validation. It
predates the review follow-up and does not validate the present source or wheel.

The two older reports below remain historical records for v1.0.0. Their
checksums are preserved unchanged, rather than relabelled as v1.1.0 results.

## Historical clean-room and real-data checks

The publication release distinguishes a newly validated release environment
from the mutable environments in which the historical manuscript experiments
were run. The latter cannot be reconstructed exactly and are not inferred from
this lock.

`requirements-lock-linux-x86_64-py310.txt` pins every Python package in the
supported clean-room environment. `environment-lock.yml` also fixes the Python
patch version. The validation report records the lock checksum, source
snapshot, operating system, CPU, visible GPUs, PyTorch CUDA/cuDNN metadata,
installed distributions, commands, captured logs and exit codes.
`checksums.sha256` provides a compact integrity record for the dependency lock,
gene-partition manifest and both validation outputs; its paths are relative to
the repository root.

To create a new environment rather than reuse an existing one:

```bash
bash scripts/run_cleanroom_rebuild.sh \
  /path/to/python3.10 \
  /new/path/spatios2e-cleanroom \
  validation/cleanroom_validation_linux_x86_64_py310.json
```

The archived report covers installation, dependency verification, checksum
manifest regeneration, unit tests, a synthetic held-out-decoder training and
evaluation, lint, distribution builds and a command-line smoke test. It is a
CPU validation and is not evidence that the complete four-cohort workflow or
CUDA execution was rerun. A future full-workflow record should use the same
reporting convention and additionally freeze the external data, pretrained
weights, trained checkpoints and source-data outputs that licensing permits.

The companion `real_heldout_smoke_hippocampus_seed42` record is a bounded
real-data integration check. It reads frozen hippocampus expression and UNI2-h
feature files, the archived biological and target partitions and Decima
vectors; verifies every spot-manifest sample against the final donor-disjoint
manuscript split; fits matched
pretrained- and random-vector decoders for one epoch using the manuscript
decoder dimensions; evaluates downstream-held-out genes; and writes a TSV
source table. Its deliberately small subset tests data alignment, fitting and
evaluation without being interpreted as a reproduction or estimate of any
reported manuscript result. The command is implemented by
`scripts/validate_real_heldout_smoke.py`; its input paths are supplied by the
operator because source data and third-party vectors are not redistributed.
