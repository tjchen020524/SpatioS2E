# Validation

The records below identify the tested source snapshots, dependencies, commands
and results. CPU checks cover installation, tests and synthetic workflows;
they do not reproduce the full four-cohort GPU analyses.

| Record | Coverage |
| --- | --- |
| [CPU validation, 2026-09-13](review_followup_20260913_final.json) | 44 tests, isolated wheel execution and synthetic primary-workflow training; record cited in Supplementary Methods |
| [Clean installation](cleanroom_validation_linux_x86_64_py310.json) | Fresh CPython 3.10.19 environment, all 38 pinned dependencies and 27 tests |
| [Real-data integration](real_heldout_smoke_hippocampus_seed42.json) | Bounded hippocampus input alignment, one-epoch fitting and held-out-gene evaluation; [output table](real_heldout_smoke_hippocampus_seed42.tsv) |

The real-data check uses small subsets to test the implementation; its metrics
are not manuscript results. Historical reports apply to their recorded source
snapshots. The pinned CPU environment is specified in
[`requirements-lock-linux-x86_64-py310.txt`](../requirements-lock-linux-x86_64-py310.txt)
and [`environment-lock.yml`](../environment-lock.yml).

## Run the checks

From the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check spatios2e tests scripts experiments
python -m build
```

To install the pinned dependencies in a new environment and write a report:

```bash
bash scripts/run_cleanroom_rebuild.sh \
  /path/to/python3.10 \
  /new/path/spatios2e-cleanroom \
  /new/path/cpu-validation.json
```

For the real-data check, run
`python scripts/validate_real_heldout_smoke.py --help` and supply the prepared
hippocampus inputs described in the [reproduction guide](../docs/reproduction.md).

From the repository root, verify the archived dependency lock, partition
manifest, validation records and README figure with:

```bash
sha256sum -c validation/checksums.sha256
```
