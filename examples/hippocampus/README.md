# Hippocampus Benchmark Example

This example uses the manuscript's donor-disjoint hippocampus partition: six
training donors (22 sections), two validation donors (8 sections) and two test
donors (4 sections).

Required external files:

- `weights/decima/rep0.ckpt`
- `data/decima_input/gene_inputs_npz/<ENSG_ID>.npz`
- `data/processed/expression_full/<sample>/{train,val,test}.npz`
- `data/processed/multimodal_features/<sample>/multimodal_features.npz`
- `data/processed/multimodal_features/modality_stats.json`
- `data/processed/spatial_graphs/<sample>/graph.npz`

Run:

```bash
bash examples/hippocampus/run_train_eval.sh
```

Outputs are written to `outputs/hippocampus_spatios2e/`.
