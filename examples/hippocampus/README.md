# Hippocampus Benchmark Example

This example uses the manuscript's donor-disjoint hippocampus partition: six
training donors (22 sections), two validation donors (8 sections) and two test
donors (4 sections).

Required external files:

Install Decima 0.5.1 in the optional model environment. Use the
[input preparation guide](../../docs/reproduction.md) to generate reference
sequence inputs and UNI2-h vectors; the generic ResNet extractor does not
produce manuscript UNI2-h features.

- `weights/decima/rep0.ckpt`
- `data/decima_input/gene_inputs_npz/<ENSG_ID>.npz`
- `data/processed/expression_full/<sample>/{train,val,test}.npz`
- `data/processed/multimodal_features/<sample>/multimodal_features.npz`
- `data/processed/multimodal_features/modality_stats.json`
- `data/processed/spatial_graphs/<sample>/graph.npz`

Expression NPZ files store a genes-by-spots CSR matrix of `log(CPM+1)` plus
exact `gene_ids` and `barcodes`. Each section belongs to only one biological
split; the selected `<split>.npz` contains that section's complete fitted panel.
The same ordered genes must be present in all sections. This example's genes
are fitted targets, not the held-out-target lists of the separate assay.

Generate the 31 coordinate descriptors from the original Visium position
files, assemble aligned inputs with 18 reserved **zero** composition channels,
fit moments on training sections only, and construct six-neighbour graphs:

```bash
spatios2e-prepare-spatial-features --raw-dir /inputs/visium \
  --output-dir /inputs/spatial_features --num-frequencies 6 --frequency-base 2
python scripts/prepare_fitted_features.py \
  --manifest /inputs/hippocampus_manifest.tsv \
  --biological-split configs/manuscript/biological_splits/hippocampus.json \
  --spatial-root /inputs/spatial_features --embedding-root /inputs/uni2h \
  --output-root data/processed/multimodal_features --celltype-width 18
spatios2e-build-spatial-graph --multimodal-dir data/processed/multimodal_features \
  --output-dir data/processed/spatial_graphs --k 6 --edge-temperature 0.1
```

The feature assembler uses manifest barcode order and rejects duplicates or
missing rows. `multimodal_features.npz` contains `[N,31]` spatial features,
`[N,1536]` image features, `[N,18]` zeros and aligned barcodes; graph node order
must follow these barcodes. No cell-composition measurements are used.
The normalizer file records separate spatial/histology/celltype means and
standard deviations, computed from the archived training sections only.

For all manuscript fitted variants, use the archived cohort preparation,
`prepare_priors`, `write_configs`, `write_multiseed_configs` and
`train_balanced_ablation` drivers listed in the reproduction guide. These
retain the training-only anchors/scales, balanced gene chunks and variant
definitions; the single example below is not the entire ablation study.

Run:

```bash
bash examples/hippocampus/run_train_eval.sh
```

Outputs are written to `outputs/hippocampus_spatios2e/`.
