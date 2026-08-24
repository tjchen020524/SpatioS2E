# SpatioS2E

SpatioS2E is a controlled framework for asking what transfers in
gene-conditioned virtual spatial transcriptomics. It separates two evaluation
axes that can otherwise be conflated:

- **across-gene abundance**: whether genes that are generally high or low are
  correctly ordered and calibrated;
- **within-gene spatial variation**: whether a gene's variation across tissue
  spots is recovered.

The repository contains the fitted-gene prediction framework, the separately
trained target-disjoint decoders used for held-out genes, component-resolved
metrics, preprocessing utilities, and an end-to-end hippocampus configuration.
It also freezes the exact four-cohort biological section splits, primary
downstream gene partitions and held-out-assay hyperparameters under
`configs/manuscript/`.
Raw data, third-party encoders and weights, trained checkpoints, and generated
predictions are intentionally not stored in git.

## Experimental systems

### Fitted genes

The fitted-gene framework combines frozen tissue-image features and spatial
coordinates, optional neighbourhood context, and an optional gene
representation. Its sequence-conditioned form uses FiLM conditioning and an
additive gene-level abundance route. Direct, morphology-only, sequence-prior,
and optional single-cell-prior variants remain available through the model
factory.

### Held-out genes

Target-disjoint evaluation uses a separate `FactorizedDotProductDecoder` on
precomputed spot representations and fixed gene vectors. In the manuscript,
the pretrained vectors came from **Decima**, an external pretrained
reference-sequence model; Decima is not the SpatioS2E predictor and is not
vendored here. Matched random and constant gene vectors serve as controls.

The release also exposes the architecture-audit decoders:

- `BiasFreeFactorizedDecoder`, with one intercept shared by all spots and
  genes;
- `ConcatenationMLPDecoder`, a parameter-matched alternative decoder family;
- `SectionCenteredResidualDecoder`, with signed outputs for exactly centred
  residual targets.

These classes reproduce the decoder definitions. Cohort-specific data
partitions, fixed vectors and optimization settings remain the responsibility
of the experiment configuration. Utilities in `spatios2e.models.gene_vectors`
standardize pretrained vectors using training genes only and construct matched
random, constant and identity-permuted controls.

## Installation

SpatioS2E requires Python 3.10 or newer.

```bash
conda env create -f environment.yml
conda activate spatios2e
pip install -e ".[preprocess,dev]"
```

Decima and UNI2-h must be obtained from their respective upstream projects.
For sequence-conditioned fitted-gene models, install Decima separately or add
its source directory to `PYTHONPATH` and place its checkpoint at the path given
in the experiment config. The optional scGPT preprocessing module likewise
requires a separate scGPT installation.

## Component-resolved evaluation

For two finite `[spot, gene]` matrices:

```python
import numpy as np
from spatios2e.evaluation import component_metrics

observed = np.load("observed.npy")
predicted = np.load("predicted.npy")
metrics = component_metrics(observed, predicted)

print(metrics["full_matrix_pcc"])
print(metrics["abundance_pcc"])
print(metrics["mean_gene_pcc"])
print(metrics["decomposition_error"])
```

The evaluator reports full-matrix PCC and MSE, abundance PCC and RMSE, mean,
median and interquartile gene-wise PCC, gene-centred full-matrix PCC, centred RMSE, and the two
exact MSE components. Gene-wise eligibility is defined only by observed
variation. An eligible gene with a constant prediction contributes PCC zero,
so different prediction conditions retain the same observed-defined
denominator. For a rectangular matrix,

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

The release also implements the no-image counterfactual used to determine how
much target-disjoint matrix performance is available from gene means alone:

```bash
spatios2e-gene-mean-counterfactual gene_vectors_and_training_means.npz \
  --output heldout_gene_means.npz
```

The input contains `vectors`, training-individual `gene_means`, and downstream
`training_indices` and `heldout_indices`. The fitted ridge receives no image,
coordinate or spot input. Its predicted held-out-gene means can be broadcast
over any number of spots with `broadcast_gene_means`; component evaluation of
that map has zero within-gene correlation by construction. Random or
identity-permuted vectors from `spatios2e.models.gene_vectors` provide matched
controls under the same fitting procedure.

## Fitted-gene hippocampus example

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

For the supplied full fitted-gene configuration, the first command computes
the training-spot-weighted abundance anchor and residual scale in one artifact;
validation and test expression are not read. The training configuration uses
balanced gene chunks and verifies complete target-gene coverage in every epoch.

Evaluation writes backward-compatible `mse` and `corr` fields together with
explicit `full_matrix_mse`, `full_matrix_pcc`, abundance and centred endpoints.
It also reports gene-PCC eligibility, finite-map coverage and training-derived
top-HVG summaries; an eligible gene with a constant predicted map contributes
zero to the primary mean rather than disappearing from the denominator.

## Repository map

- `spatios2e/models/`: fitted-gene models and held-out-gene decoders;
- `spatios2e/evaluation/`: checkpoint evaluation and component-resolved
  endpoints;
- `spatios2e/preprocessing/`: expression, image-feature and graph preparation;
- `spatios2e/training/`: fitted-gene training entry point;
- `configs/` and `examples/`: portable hippocampus example;
- `configs/manuscript/`: four-cohort design, exact biological/gene splits,
  held-out-assay protocol and external-model checksums;
- `docs/paper_code_map.md`: mapping from manuscript analyses to public code;
- `tests/`: import, decoder and metric identity tests.

## Reproducibility boundary

Included in git are reusable source code, configuration templates and tests.
Excluded are identifiable or licensed source data, third-party weights,
checkpoints, predictions, cluster logs and manuscript build artifacts. The
manuscript's numerical source-data package is versioned separately from this
software release.

## License and citation

The code is released under the MIT License. Citation metadata will be added
when the manuscript or preprint record is public.
