# Using SpatioS2E

Run the commands below from the repository root after installation. For
complete manuscript workflows, see [Reproducing the paper](reproduction.md).

## Component-resolved evaluation

Use the Python API with any aligned observed and predicted matrices:

```python
import numpy as np

from spatios2e.evaluation import component_metrics

observed = np.load("observed.npy")
predicted = np.load("predicted.npy")
metrics = component_metrics(observed, predicted)

print(metrics["full_matrix_pcc"])
print(metrics["gene_mean_pcc"])
print(metrics["mean_gene_pcc"])
print(metrics["centered_rmse"])
print(metrics["decomposition_error"])
```

For finite rectangular matrices, the evaluator verifies

```text
full-matrix MSE = gene-mean MSE + centred MSE.
```

The equivalent command-line interface reads an NPZ containing `observed` and
`predicted` arrays:

```bash
spatios2e-eval-components predictions.npz \
  --output component_metrics.json
```

To remove observed and predicted means independently within each section, add
section labels to the NPZ and pass `--section-key section_ids`.

## Gene-conditioned prediction

### Target-disjoint decoders

`spatios2e.models` provides the primary `FactorizedDotProductDecoder` and three
architecture controls:

- `BiasFreeFactorizedDecoder`;
- `ConcatenationMLPDecoder`;
- `SectionCenteredResidualDecoder`.

`spatios2e.training.heldout` exposes reusable fitting, checkpoint selection and
evaluation functions for precomputed spot representations and fixed gene
vectors. The held-out targets enter only the post-fit evaluation call. See the
[held-out assay guide](heldout_assay.md) for the batch interface and
an API example.

The manuscript uses sequence-derived Decima vectors and static scGPT gene-token
vectors. Utilities for vector standardization and matched controls are in
`spatios2e.models.gene_vectors`. Static scGPT token vectors can be extracted
from an upstream checkpoint with:

```bash
spatios2e-extract-scgpt-tokens \
  --checkpoint weights/scgpt/whole_human_model/best_model.pt \
  --vocabulary weights/scgpt/whole_human_model/vocab.json \
  --output data/scgpt_whole_human_gene_tokens.npz
```

The no-image counterfactual is available as both a Python API and a CLI:

```bash
spatios2e-gene-mean-counterfactual gene_vectors_and_training_means.npz \
  --output heldout_gene_means.npz
```

### Fitted-target workflow

The fitted-target implementation combines frozen tissue-image features,
spatial coordinates, optional neighbourhood context and optional gene
conditioning. A donor-disjoint hippocampus configuration is supplied under
`configs/` and `examples/`:

```bash
bash examples/hippocampus/run_train_eval.sh
```

The underlying steps can also be invoked directly:

```bash
spatios2e-compute-residual-scale configs/hippocampus_spatios2e.yaml
spatios2e-train --config configs/hippocampus_spatios2e.yaml
spatios2e-eval \
  --config configs/hippocampus_spatios2e.yaml \
  --ckpt outputs/hippocampus_spatios2e/checkpoints/best.pt \
  --split test \
  --save-dir outputs/hippocampus_spatios2e/results
```

New-cohort prediction requires cohort data and upstream tissue-image and gene
representations. UNI2-h, Decima and scGPT weights are obtained from their
respective projects; trained downstream checkpoints are not distributed in
this repository.

## Optional dependencies and validation

Install preprocessing or development dependencies as needed:

```bash
python -m pip install -e ".[preprocess,dev]"
```

Decima-based input preparation additionally requires `decima==0.5.1`.
The [reproduction guide](reproduction.md) describes the separate upstream
code and artifact requirements.

For the pinned Linux/CPU environment and independent installation checks, see
[validation records](../validation/README.md). Development and testing
instructions are in [CONTRIBUTING.md](../CONTRIBUTING.md).
