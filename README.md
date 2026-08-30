# SpatioS2E

SpatioS2E is a controlled framework for asking what transfers in
gene-conditioned virtual spatial transcriptomics. It separates two evaluation
axes that can otherwise be conflated:

- **gene-mean log-expression**: whether genes that are generally high or low
  are correctly ordered and calibrated across targets;
- **within-gene spatial variation**: whether a gene's variation across tissue
  spots is recovered.

The repository contains the fitted-target prediction framework, the separately
trained target-disjoint decoders used for held-out targets, component-resolved
metrics, preprocessing utilities, and an end-to-end hippocampus configuration.
It also freezes the exact four-cohort biological section splits, primary and
sensitivity downstream gene partitions and held-out-assay hyperparameters under
`configs/manuscript/`.
Raw data, third-party encoders and weights, trained checkpoints, and generated
predictions are intentionally not stored in git.

## Experimental systems

### Fitted targets

The fitted-gene framework combines frozen tissue-image features and spatial
coordinates, optional neighbourhood context, and an optional gene
representation. Its sequence-conditioned form uses FiLM conditioning and an
additive gene-conditioned route. Direct, morphology-only, sequence-prior,
and optional single-cell-prior variants remain available through the model
factory.

### Held-out targets

Target-disjoint evaluation uses a separate `FactorizedDotProductDecoder` on
precomputed spot representations and fixed gene vectors. The manuscript uses
two external representation sources: sequence-derived **Decima** vectors and
static gene-token vectors from the **scGPT whole-human checkpoint**. Neither
Decima nor scGPT is the spatial predictor introduced by this package, and
neither dependency is vendored here. Dimension-matched random vectors, a
single constant vector and within-partition identity shuffles provide matched
controls.

The release also exposes the architecture-audit decoders:

- `BiasFreeFactorizedDecoder`, with one intercept shared by all spots and
  genes;
- `ConcatenationMLPDecoder`, a parameter-matched alternative decoder family;
- `SectionCenteredResidualDecoder`, with signed outputs for exactly centred
  residual targets.

These classes reproduce the decoder definitions. Cohort-specific data
partitions, fixed vectors and optimization settings remain the responsibility
of the experiment configuration. The reusable fitting and checkpoint-selection
loop is exposed in `spatios2e.training.heldout`. Utilities in
`spatios2e.models.gene_vectors` standardize pretrained vectors using training
genes only and construct matched random, constant and partition-preserving
identity-permuted controls.

## Installation

SpatioS2E requires Python 3.10 or newer.

For the publication release, the supported clean-room target is CPython
3.10.19 on Linux x86_64 with CPU PyTorch. From the repository root:

```bash
python3.10 -m venv --copies /new/path/spatios2e-cleanroom
/new/path/spatios2e-cleanroom/bin/python -m pip install \
  --requirement requirements-lock-linux-x86_64-py310.txt
/new/path/spatios2e-cleanroom/bin/python -m pip install \
  --no-deps --no-build-isolation .
/new/path/spatios2e-cleanroom/bin/python scripts/validate_cleanroom.py \
  --output validation/local-validation.json
```

`environment-lock.yml` provides the equivalent Python and pip entry point.
The version-pinned lock, its checksum and a successful independent rebuild are
archived under [`validation`](validation/README.md). This validated CPU target
does not retrospectively reconstruct the mutable CUDA environments used for
the manuscript experiments.

For development or platform-specific GPU work, the following broader Conda
specification remains available as a convenience environment rather than an
archival lock:

```bash
conda env create -f environment.yml
conda activate spatios2e
pip install -e ".[preprocess,dev]"
```

Decima, UNI2-h and scGPT checkpoints must be obtained from their respective
upstream projects.
For sequence-conditioned fitted-gene models, install Decima separately or add
its source directory to `PYTHONPATH` and place its checkpoint at the path given
in the experiment config. The static scGPT token extractor reads the packaged
checkpoint tensors and vocabulary directly; the older optional single-cell
context module requires a separate scGPT installation.

## Component-resolved evaluation

For two finite `[spot, gene]` matrices:

```python
import numpy as np
from spatios2e.evaluation import component_metrics

observed = np.load("observed.npy")
predicted = np.load("predicted.npy")
metrics = component_metrics(observed, predicted)

print(metrics["full_matrix_pcc"])
print(metrics["gene_mean_pcc"])
print(metrics["mean_gene_pcc"])
print(metrics["decomposition_error"])
```

The evaluator reports full-matrix PCC and MSE, gene-mean PCC and RMSE, mean,
median and interquartile gene-wise PCC, gene-centred full-matrix PCC, centred
RMSE, and the two exact MSE components. The legacy keys `abundance_pcc` and
`abundance_rmse` remain aliases for compatibility; they refer to mean
normalized log-expression rather than absolute molecule counts. Gene-wise
eligibility is defined only by observed variation. An eligible gene with a
constant prediction contributes PCC zero, so different prediction conditions
retain the same observed-defined denominator. For a rectangular matrix,

```text
full-matrix MSE = gene-mean MSE + centred MSE.
```

The same evaluator is available as a command-line tool:

```bash
spatios2e-eval-components predictions.npz --output component_metrics.json
```

The NPZ must contain `observed` and `predicted` arrays. Add
`--section-key section_ids` to centre both matrices independently within each
tissue section before evaluation.

## Frozen gene representations

Decima features are produced by the upstream package. Static scGPT gene-token
vectors can be extracted reproducibly from the official whole-human checkpoint
without running a cell through the transformer:

```bash
spatios2e-extract-scgpt-tokens \
  --checkpoint weights/scgpt/whole_human_model/best_model.pt \
  --vocabulary weights/scgpt/whole_human_model/vocab.json \
  --output data/scgpt_whole_human_gene_tokens.npz
```

The command writes `gene_symbols`, a `[token, 512]` vector matrix and a JSON
sidecar containing checkpoint and vocabulary SHA-256 digests. Exact hashes,
extraction semantics, vocabulary coverage and downstream comparison boundaries
are frozen in [`configs/manuscript`](configs/manuscript/README.md).

The release also implements the no-image counterfactual used to determine how
much target-disjoint matrix performance is available from gene means alone:

```bash
spatios2e-gene-mean-counterfactual gene_vectors_and_training_means.npz \
  --output heldout_gene_means.npz
```

The input contains `vectors`, training-individual `gene_means`, and downstream
`training_indices` and `heldout_indices`. The fitted ridge receives no image,
coordinate or spot input. Its predicted held-out-target means can be broadcast
over any number of spots with `broadcast_gene_means`; component evaluation of
that map has zero within-gene correlation by construction. Random or
identity-permuted vectors from `spatios2e.models.gene_vectors` provide matched
controls under the same fitting procedure.

## Fitted-target hippocampus example

The public example expects the following external layout:

```text
weights/
  decima/rep0.ckpt
data/
  decima_input/gene_inputs_npz/<ENSG_ID>.npz
  processed/expression_full/<sample>/{train,val,test}.npz
  processed/multimodal_features/modality_stats.json
  processed/multimodal_features/<sample>/multimodal_features.npz
  processed/spatial_graphs/<sample>/graph.npz
```

Run training and evaluation from the repository root:

```bash
bash examples/hippocampus/run_train_eval.sh
```

or invoke the individual commands:

```bash
spatios2e-compute-residual-scale configs/hippocampus_spatios2e.yaml
spatios2e-train --config configs/hippocampus_spatios2e.yaml
spatios2e-eval \
  --config configs/hippocampus_spatios2e.yaml \
  --ckpt outputs/hippocampus_spatios2e/checkpoints/best.pt \
  --split test \
  --save-dir outputs/hippocampus_spatios2e/results
```

For the supplied full fitted-target configuration, the first command computes
the training-spot-weighted gene-mean anchor and residual scale in one artifact;
validation and test expression are not read. The training configuration uses
balanced gene chunks and verifies complete target-gene coverage in every epoch.

Evaluation writes backward-compatible `mse` and `corr` fields together with
explicit `full_matrix_mse`, `full_matrix_pcc`, gene-mean and centred endpoints.
It also reports gene-PCC eligibility, finite-map coverage and training-derived
top-HVG summaries; an eligible gene with a constant predicted map contributes
zero to the primary mean rather than disappearing from the denominator.

## Repository map

- `spatios2e/models/`: fitted-target models and held-out-target decoders;
- `spatios2e/evaluation/`: checkpoint evaluation and component-resolved
  endpoints;
- `spatios2e/preprocessing/`: expression, image-feature and graph preparation;
- `spatios2e/training/`: fitted-target training plus reusable held-out-target
  fitting and evaluation;
- `configs/` and `examples/`: portable hippocampus example;
- `configs/manuscript/`: four-cohort design, exact biological/gene splits,
  held-out-assay protocol and external-model checksums;
- `docs/paper_code_map.md`: mapping from manuscript analyses to public code;
- `docs/heldout_assay.md`: portable target-disjoint batch and control contract;
- `docs/releases/v0.2.0.md`: publication-release notes;
- `tests/`: import, decoder and metric identity tests.

## Reproducibility boundary

Included in git are reusable source code, configuration templates and tests.
Excluded are identifiable or licensed source data, third-party weights,
checkpoints, predictions, cluster logs and manuscript build artifacts. The
manuscript's numerical source data and final figure-assembly scripts are
versioned separately in the submission archive; this repository contains the
portable model, evaluation and control implementations they call.

## License and citation

The code is released under the MIT License. Please use the metadata in
[`CITATION.cff`](CITATION.cff) when citing this software. The version-specific
Zenodo DOI and manuscript DOI will be linked here after they are assigned.
