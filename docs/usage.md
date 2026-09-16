# Evaluation and training

Use the examples below to evaluate prediction matrices or fit a model with
prepared features. For the paper's cohort-specific experiments, follow
[Reproducing the paper](reproduction.md). Run commands from the repository root.

## Evaluate expression predictions

Supply two matrices with spots in rows and genes in columns. Observed and
predicted values must use the same spot order, gene order and expression scale.

```python
import numpy as np

from spatios2e.evaluation import component_metrics

observed = np.load("observed.npy")
predicted = np.load("predicted.npy")
metrics = component_metrics(observed, predicted)

print(metrics["full_matrix_pcc"])    # correlation across all spot-gene pairs
print(metrics["gene_mean_pcc"])      # correlation across gene means
print(metrics["mean_gene_pcc"])      # mean of within-gene correlations
print(metrics["centered_rmse"])      # RMSE after removing each gene's mean
print(metrics["decomposition_error"])  # numerical error in the MSE identity below
```

For finite rectangular matrices, the evaluator verifies

```text
full-matrix MSE = gene-mean MSE + centred MSE.
```

Within-gene correlations include genes with observed standard deviation above
`1e-6`. A constant prediction for an eligible gene contributes a correlation of zero.

To evaluate an NPZ file containing `observed` and `predicted` arrays:

```bash
spatios2e-eval-components predictions.npz \
  --output component_metrics.json
```

To remove observed and predicted means independently within each section, add
section labels to the NPZ and pass `--section-key section_ids`.

## Train with fixed gene vectors

Use `FactorizedDotProductDecoder` for the primary held-out-gene model. The
[training example](heldout_assay.md) shows how to fit on training genes and
evaluate held-out genes with separate data loaders. The architecture controls
in `spatios2e.models` are:

- `BiasFreeFactorizedDecoder`
- `ConcatenationMLPDecoder`
- `SectionCenteredResidualDecoder`

Prepare Decima vectors using the [input preparation commands](reproduction.md).
To extract static scGPT gene-token vectors from the whole-human checkpoint:

```bash
spatios2e-extract-scgpt-tokens \
  --checkpoint weights/scgpt/whole_human_model/best_model.pt \
  --vocabulary weights/scgpt/whole_human_model/vocab.json \
  --output data/scgpt_whole_human_gene_tokens.npz
```

To fit a no-image ridge model that predicts one expression value per gene:

```bash
spatios2e-gene-mean-counterfactual gene_vectors_and_training_means.npz \
  --output heldout_gene_means.npz
```

## Run the fitted-gene hippocampus example

First prepare the inputs listed in the
[hippocampus example](../examples/hippocampus/README.md), then run:

```bash
bash examples/hippocampus/run_train_eval.sh
```

The script runs these steps:

```bash
spatios2e-compute-residual-scale configs/hippocampus_spatios2e.yaml
spatios2e-train --config configs/hippocampus_spatios2e.yaml
spatios2e-eval \
  --config configs/hippocampus_spatios2e.yaml \
  --ckpt outputs/hippocampus_spatios2e/checkpoints/best.pt \
  --split test \
  --save-dir outputs/hippocampus_spatios2e/results
```

## Extra dependencies and tests

Install preprocessing or development dependencies as needed:

```bash
python -m pip install -e ".[preprocess,dev]"
```

Decima-based input preparation additionally requires `decima==0.5.1`.
See [Data and pretrained models](data_availability.md) for the required downloads.

For the pinned Linux/CPU environment and independent installation checks, see
[validation and testing](../validation/README.md).
