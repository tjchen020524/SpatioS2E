# Train and evaluate held-out genes

The example below fits a decoder using prepared image features and fixed gene
vectors. For the full cohort experiments, see [Reproducing the paper](reproduction.md).

Use training genes for both optimization and checkpoint selection. Reserve the
held-out genes and test individuals for final evaluation.

## Prepare the data loaders

Training, validation and test iterables yield either a mapping or a pair:

```python
{
    "spot_features": FloatTensor[number_of_spots, spot_dimension],
    "expression": FloatTensor[number_of_spots, number_of_genes],
}
```

The aliases `x` and `y` are also accepted. Iterables used for more than one
epoch must be re-iterable, such as a `torch.utils.data.DataLoader`.

## Fit the model and evaluate held-out genes

Here, `gene_vectors` contains one row per gene in the same order as the
expression columns. `training_gene_indices` and `heldout_gene_indices` are
disjoint column indices. The three loaders draw spots from separate training,
validation and test individuals.

```python
import torch

from spatios2e.models import FactorizedDotProductDecoder
from spatios2e.training import (
    HeldOutTrainingConfig,
    evaluate_heldout_decoder,
    fit_heldout_decoder,
)

config = HeldOutTrainingConfig(seed=42)
torch.manual_seed(config.seed)  # controls decoder initialization
decoder = FactorizedDotProductDecoder(
    spot_dim=1536,
    gene_dim=gene_vectors.shape[1],
)
record = fit_heldout_decoder(
    decoder,
    training_loader,
    validation_loader,
    gene_vectors,
    training_gene_indices,
    config=config,
    device="cuda",
)
metrics = evaluate_heldout_decoder(
    decoder,
    test_loader,
    gene_vectors,
    heldout_gene_indices,
    device="cuda",
)
```

`fit_heldout_decoder` indexes expression only with `training_gene_indices`.
The validation subset is sampled from those same genes. The held-out indices
enter only the post-fit evaluation call.

## Compare gene-vector conditions

Use the same individual and gene splits for every condition. The paper compares:

- feature-wise standardized pretrained vectors, with statistics estimated from
  downstream training genes only;
- fixed dimension-matched standard-normal vectors;
- one zero vector shared by all genes;
- a pretrained-vector identity shuffle performed independently within the
  downstream training and held-out partitions.

The corresponding utilities are `standardize_from_training_genes`,
`random_gene_vectors`, `constant_gene_vectors` and
`permute_gene_identity_within_partitions`. Seeds and checkpoint-selection
settings are recorded in
[`configs/manuscript/heldout_assay.yaml`](../configs/manuscript/heldout_assay.yaml).

The constant-vector decoder predicts the same gene mean for every target, so
gene-mean PCC is undefined rather than zero. Full-matrix, error and within-gene
endpoints remain defined.
