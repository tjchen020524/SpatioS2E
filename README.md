# SpatioS2E

SpatioS2E predicts spatial gene expression from H&E-derived spot features, a
spatial neighbor graph, and gene-level molecular priors. The main model uses a
residual form:

```text
predicted log expression = gene prior + spatial residual
```

The repository contains the Python package, command-line tools, and benchmark
configuration used for the hippocampus experiments. Data files, trained
checkpoints, Decima weights, UNI2-h weights, and generated results are kept
outside git.

## Repository Contents

- `spatios2e/`: package code for data loading, preprocessing, models, training,
  residual-scale estimation, and evaluation.
- `configs/hippocampus_spatios2e.yaml`: default hippocampus benchmark config.
- `examples/hippocampus/`: gene splits and a train/evaluate wrapper script.
- `tests/`: smoke tests for public imports.
- `environment.yml` and `pyproject.toml`: reproducible setup files.

## Installation

Create the conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate spatios2e
pip install -e ".[preprocess,dev]"
```

Decima is not vendored in this repository. Install it separately, or add its
source directory to `PYTHONPATH` before running sequence-conditioned models:

```bash
export PYTHONPATH="/path/to/decima/src:${PYTHONPATH:-}"
```

scGPT is also external. It is only needed to build the optional scGPT snRNA
gene-context artifact:

```bash
export PYTHONPATH="/path/to/scGPT:${PYTHONPATH:-}"
```

The default config expects the Decima checkpoint at:

```text
weights/
  decima/
    rep0.ckpt
```

## Expected Data Layout

The hippocampus config uses paths relative to the repository root:

```text
data/
  decima_input/
    gene_inputs_npz/
      <ENSG_ID>.npz
  processed/
    expression_full/
      <sample>/
        train.npz
        val.npz
        test.npz
    multimodal_features/
      modality_stats.json
      <sample>/
        multimodal_features.npz
    spatial_graphs/
      <sample>/
        graph.npz
```

Required file fields:

- expression split files: CSR fields `data`, `indices`, `indptr`, `shape`, plus
  `gene_ids` and `barcodes`;
- multimodal feature files: `barcodes`, `spatial_features`,
  `histology_embeddings`, `celltype_weights`, `spatial_feature_names`,
  `celltype_names`, and/or `combined_features`;
- `modality_stats.json`: `spatial`, `histology`, and `celltype` mean/std arrays;
- graph files: `edge_index`, `edge_weight`, `barcodes`, and optional
  `positions`/`metadata`;
- Decima inputs: one `<ENSG_ID>.npz` per gene containing a `(5, L)`
  one-hot-plus-mask array.

Optional single-cell prior inputs:

- donor baseline/residual-scale files: `gene_ids`, `donor_ids`, `sample_ids`,
  `sample_donor_ids`, `donor_log_mu_base`, and `residual_scale`;
- scGPT gene-context profiles: `gene_ids`, `donor_ids`, `sample_ids`,
  `sample_donor_ids`, `donor_scgpt_gene_context`, and
  `donor_scgpt_gene_expr_sum`.

To build the scGPT gene-context profile file from 10x snRNA matrices, provide a
config whose `paths.source_snrna_base_npz`, `paths.sample_links_csv`, and
`paths.snrna_root` point to the donor baseline artifact, donor/sample metadata,
and raw snRNA directories:

```bash
spatios2e-build-snrna-scgpt-context \
  --config <config.yaml> \
  --model-dir /path/to/scGPT/brain_model \
  --scgpt-root /path/to/scGPT \
  --output data/processed/snrna/snrna_scgpt_gene_context.npz \
  --amp
```

## Run the Hippocampus Benchmark

From the repository root:

```bash
spatios2e-compute-residual-scale configs/hippocampus_spatios2e.yaml
spatios2e-train --config configs/hippocampus_spatios2e.yaml
spatios2e-eval \
  --config configs/hippocampus_spatios2e.yaml \
  --ckpt outputs/hippocampus_spatios2e/checkpoints/best.pt \
  --split test \
  --save-dir outputs/hippocampus_spatios2e/results
```

The same sequence is available as:

```bash
bash examples/hippocampus/run_train_eval.sh
```

## Model Variants

Set `model.variant` in the config:

- `spatios2e`: H&E morphology, spatial graph, DNA sequence prior, and learned
  spatial residual.
- `morphology_graph`: H&E morphology and spatial graph without a molecular
  prior.
- `sequence_prior`: DNA prior baseline without a learned spatial residual.
- `single_cell_prior`: DNA sequence conditioning with a learned scalar gate
  adding donor-matched scGPT snRNA gene-context information.

## Outputs

Evaluation writes:

- `test_overall.json`: pooled MSE and Pearson correlation across evaluated
  spot-gene pairs;
- `test_sample_metrics.tsv`: per-sample metrics;
- `test_gene_metrics.tsv`: per-gene metrics, including gene-level spatial-map
  correlations.

## Scope

Included in git:

- model, training, preprocessing, and evaluation code;
- public ablation variants;
- hippocampus benchmark config and gene splits;
- smoke tests and GitHub Actions configuration.

Not included in git:

- raw Visium data or processed matrices;
- third-party model weights;
- trained checkpoints, predictions, reports, figures, and cluster logs.

## License

The code is released under the MIT License. Citation information can be added
after the manuscript or preprint record is available.
