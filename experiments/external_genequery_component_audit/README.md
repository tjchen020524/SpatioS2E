# External gene-query audits

Source workflows for Fig. 5c and Extended Data Fig. 10. Run the commands from
the repository root after installing SpatioS2E. They are included in the Git
repository and source distribution, not installed as wheel entry points.

## Scope

- GeneQuery: a reconstruction of the gene-aware head with frozen ResNet-50
  features and a separate end-to-end ResNet-50 training workflow. Both use
  seeds 42, 123 and 456 with independently fitted semantic, identity-shuffled,
  random and constant conditions.
- DeepSpot-M: inference-time token interventions in the released checkpoint,
  using the scGPT pathway and seed 42. No backbone or decoder is refitted.
- A no-image ridge, common component evaluator and paired run summarizer.

GeneQuery uses 613 training and 138 held-out genes; DeepSpot-M uses 601 and
135. Patients A–E train the GeneQuery reconstruction and no-image mappings,
F selects GeneQuery checkpoints, and G–H provide 3,097 test spots. DeepSpot-M
target tokens have upstream spatial-training exposure. These are within-system
audits, not a ranking under identical training conditions. Experimental HBD
analyses are not part of this manuscript release.

## Dependencies and upstream artifacts

All 24 actual run configurations are frozen in
`configs/manuscript/genequery_runs.json`. Use
`python scripts/run_genequery_record.py --help` to render an exact command;
training starts only with `--execute`. Both settings used evaluation batch
size 8. The frozen-feature script's generic default of 16 is not the value
used for these manuscript runs.

For GeneQuery image extraction and end-to-end training:

```bash
python -m pip install -e '.[external-audit]'
```

The core CPU lock supports metric/control tests and synthetic head tests; it
does not lock the optional GPU stack. The inspected external research
environment has PyTorch 2.8.0, torchvision 0.23.0, timm 1.0.27 and safetensors
0.7.0; these observations are not an immutable historical environment lock.
Install DeepSpot-M separately in an environment compatible with its pinned
upstream source. Do not force the core CPU lock onto that environment.

Obtain upstream inputs under their own access and license terms. They are not
covered by this repository's MIT license and are not redistributed here:

| Input | Revision / SHA-256 |
| --- | --- |
| GeneQuery source, `xy-always/GeneQuery` | `7d61d38497316cce384fc11cdf7158cea752d413` |
| `gene-aware/src/her2_gpt_description_emb.npy` | `61f482f4a5e4377f1085d6a95d28ff59cf721a9d4311e1abae9fc7813efbd83d` |
| timm `resnet50.a1_in1k` model artifact | Hugging Face revision `767268603ca0cb0bfe326fa87277f19c419566ef` |
| DeepSpot-M source | `3ec046fffba9bed05974283a1a020ee01ebe6c22` |
| DeepSpot-M full checkpoint | `7c57a60b82f3a32b54433430fccab4af2de400b89ca78c97d07cb63f0c6721b3` |

The GeneQuery head implements the published architecture, retaining its
64 attention heads of width 64. Panel preparation checks the official vector
checksum and count-header-derived order against the 30 published descriptions.
The study uses patient/gene-disjoint partitions and log(CPM+1), rather than
the original random spot split and per-spot min–max target scaling.

## Prepared inputs

Set `SPATIOS2E_HER2_ROOT` to a directory containing:

```text
data/
  sample_manifest.tsv
  expression/<sample>/<split>.npz
  coordinates/<sample>.npz
  source/counts/count-matrices/A1.tsv.gz
  source/images/images/HE/<sample>.jpg
```

The manifest needs `sample`, `patient`, `split` (`train`, `val`, `test`) and
`n_spots`. Expression NPZ files contain gene-by-spot CSR arrays `data`,
`indices`, `indptr`, `shape`, plus ordered string `gene_ids` and `barcodes`.
Values are the prepared log(CPM+1) expression, not raw counts. Coordinates
contain `pixel_x`, `pixel_y` and `barcodes` in the same spot order. This release
consumes these prepared inputs; it does not download or reconstruct the raw
HER2ST preprocessing pipeline automatically.

Set `SPATIOS2E_GENEQUERY_PANEL` to the public 785-symbol panel NPY, and
`SPATIOS2E_GENE_META` to the symbol/Ensembl TSV (symbols in its first column,
Ensembl IDs in `gene_id`). Defaults are under the ignored `data/` directory.
Target partitions come from `configs/manuscript/gene_splits/her2st/`.
Put the timm checkpoint at `weights/resnet50.a1_in1k/model.safetensors`, or
pass `--checkpoint` to image extraction and end-to-end training.

## GeneQuery workflow

```bash
AUDIT=experiments/external_genequery_component_audit
python "$AUDIT/prepare_panel.py" --official-embedding /path/to/her2_gpt_description_emb.npy
python "$AUDIT/extract_resnet50_features.py" --device cuda
python "$AUDIT/extract_resnet50_patches.py"

for seed in 42 123 456; do
  for variant in semantic identity_shuffle random constant; do
    python "$AUDIT/train_genequery_audit.py" --seed "$seed" --variant "$variant" --device cuda
    python "$AUDIT/audit_predictions.py" "$AUDIT/runs/seed_$seed/$variant/predictions.npz"
    python "$AUDIT/train_genequery_trainable_backbone.py" --seed "$seed" --variant "$variant" --device cuda
    python "$AUDIT/audit_predictions.py" "$AUDIT/trainable_backbone_runs/seed_$seed/$variant/predictions.npz"
  done
  python "$AUDIT/mean_only_baseline.py" --seed "$seed"
  python "$AUDIT/audit_predictions.py" "$AUDIT/runs/seed_$seed/semantic_mean_only/predictions.npz"
done
python "$AUDIT/summarize_audit.py" --run-root "$AUDIT/runs" --expected-seeds 42 123 456
python "$AUDIT/summarize_audit.py" --run-root "$AUDIT/trainable_backbone_runs" --expected-seeds 42 123 456 --output-dir "$AUDIT/trainable_backbone_runs/summary_multiseed"
```

These commands launch substantial training; use your own scheduler/resources.
The defaults retain the head dimensions, 100-epoch maximum, patience 12 and
batch size 4. End-to-end training does not add augmentation by default.
`--max-spots` is for integration checks, not manuscript reproduction.

## DeepSpot-M workflow

Install the upstream `deepspotm` package at the recorded source revision and
obtain its gated model directory, including `tokens.csv` and weights:

```bash
python "$AUDIT/deepspotm/prepare_panel.py" --model-root /path/to/DeepSpotM
python "$AUDIT/deepspotm/predict_zero_shot.py" --model-root /path/to/DeepSpotM --source scgpt --seed 42
python "$AUDIT/mean_only_baseline.py" --seed 42 --panel-artifact "$AUDIT/artifacts/deepspotm/her2_panel_scgpt.npz" --output-root "$AUDIT/deepspotm/runs"
for variant in semantic identity_shuffle random constant semantic_mean_only; do
  python "$AUDIT/audit_predictions.py" "$AUDIT/deepspotm/runs/seed_42/$variant/predictions.npz"
done
python "$AUDIT/summarize_audit.py" --run-root "$AUDIT/deepspotm/runs" --expected-seeds 42
```

## Metrics and outputs

The evaluator accepts NPZ files with `pred`, `true` (spot × gene), `gene_ids`,
`symbols`, `sample_ids` and `individual_ids`; model exporters also store
`barcodes`. It reports pooled, per-gene, per-section and per-individual metrics.
Section-centred full-matrix PCC correlates flattened residuals; it is distinct
from averaging section-centred per-gene correlations. External matrix and
gene-mean PCC use zero for zero-variance vectors. Observed-eligible genes with
constant predictions remain in the within-gene PCC denominator with value zero.

Summary MSE reductions are control minus semantic; PCC effects are semantic
minus control. `gene_mean_fraction` is a signed-change ratio retained for
compatibility: when MSE reduction is negative, the ratio describes an error
increase, not a gain. `mse_change_direction` makes this explicit. Fractions
are undefined when total change is effectively zero. Seeds describe
computational variability, not independent biological replication.

Outputs contain local provenance paths and numerical predictions. They are
ignored by Git; use the separately curated manuscript Source Data for public
result tables. No training or model download is performed by the tests.
