# Reproducing the paper

The main held-out-gene experiment uses expression matrices, UNI2-h image
features and fixed Decima gene vectors. The steps below prepare these inputs,
fit one model and summarize the complete set of runs. The final section lists
the scGPT, fitted-gene and other analyses.

Run commands from the repository root. Replace `/inputs`, `/weights` and
`/work/reproduction` with your local paths. `--data-root` keeps experiment
inputs and results in a separate working directory.

## 1. Install dependencies and obtain the inputs

Obtain the cohort data and model files listed in
[Data and pretrained models](data_availability.md). In an environment with a
CUDA-compatible PyTorch installation, install the preprocessing and analysis
dependencies:

```bash
python -m pip install -e '.[preprocess,analysis]'
python -m pip install 'decima==0.5.1'
```

Decima input preparation also needs its metadata AnnData, `rep0.ckpt` and
reference hg38 FASTA; see the
[Decima instructions](https://github.com/Genentech/decima).
UNI2-h requires the [provider's access approval](https://huggingface.co/MahmoodLab/UNI2-h).
The model revisions used in the paper are recorded in
[`external_models.yaml`](../configs/manuscript/external_models.yaml).
The pinned CPU environment is for the [installation tests](../validation/README.md),
not GPU training.

## 2. Prepare image features and gene vectors

### Extract UNI2-h image features

Supply one image and a TSV with unique `barcode`, `pixel_x`, `pixel_y` columns.
For Visium use full-resolution coordinates and the section's Space Ranger spot
diameter, multiplying both by the scale of the supplied image. DLPFC uses the
supplied RGB immunofluorescence rendering. In this example, diameter/scale are
illustrative and must be replaced by that section's actual values:

```bash
python -m spatios2e.preprocessing.extract_uni2h \
  --image /inputs/section/image.png --coordinates /inputs/section/coordinates.tsv \
  --crop-mode visium --spot-diameter-fullres 85.0 --image-scale 0.2 \
  --checkpoint /weights/UNI2-h/pytorch_model.bin \
  --output /features/section/embeddings.npz --device cuda
```

Visium crops use the scaled spot diameter and reflection padding. HER2ST uses
`--crop-mode her2st --image-scale 1`, image-pixel coordinates, half the median
nearest-centre spacing (minimum 32 pixels), and white padding. Both use bilinear
224 × 224 resizing and ImageNet normalization. The NPZ contains `barcodes[N]`
and `embeddings[N,1536]`; a JSON records input/output hashes and crop settings.
Archived cohort-specific extractors retain the original raw-data adapters.

### Extract Decima gene vectors

```bash
python -m spatios2e.preprocessing.prepare_decima_vectors sequences \
  --genes /inputs/all_genes.txt --metadata /inputs/decima_metadata.h5ad \
  --fasta /inputs/hg38.fa --output-dir /inputs/gene_inputs_npz
python -m spatios2e.preprocessing.prepare_decima_vectors vectors \
  --genes /inputs/all_genes.txt --input-dir /inputs/gene_inputs_npz \
  --checkpoint /weights/decima/rep0.ckpt --output /inputs/decima_vectors.npz \
  --batch-size 1 --device cuda
```

Use unique exact Ensembl identifiers: no version stripping or alias substitution.
Each gene input has `seq_mask[5,524288]`, constructed by Decima from strand-aware
windows and a gene-body mask. Mean pooling the final frozen feature map gives
`embeddings[G,1920]`, ordered by `gene_ids`. Missing inputs are rejected. Both
stages record hashes. Downstream standardization is fitted later, on training
genes only.

### Prepare expression matrices and sample assignments

The cohort-specific `prepare_hippocampus.py`, `prepare_dlpfc.py`,
`prepare_nac.py` and `prepare_her2st.py` scripts in `manuscript_workflows/experiments/`
perform the filtering, normalization, identifier matching and zero filling used
in the paper. For other Visium inputs, see `spatios2e-normalize-visium --help`.
The held-out-gene runner expects:

| Prepared input | Contents |
| --- | --- |
| Manifest TSV | `sample`, `barcode`, `split`; unique sample/barcode pairs; frozen biological assignment |
| `<expression-root>/<sample>/<split>.npz` | CSR `data`, `indices`, `indptr`, `shape`; **genes × spots** normalized `log(CPM+1)`; `gene_ids`, `barcodes` |
| `<embedding-root>/<sample>/embeddings.npz` | `barcodes` and finite UNI2-h `embeddings[spots,1536]` |
| Decima NPZ | unique `gene_ids`, finite `embeddings[genes,1920]`; may cover a superset |
| Design records | `configs/manuscript/biological_splits/` and `gene_splits/` |

Expression rows must be the sorted union of the cohort's frozen train/held-out
gene lists. Spot alignment is by barcode, not array position. Historical NPZ
files may contain object arrays: load only trusted preprocessing outputs.

## 3. Train and evaluate the held-out-gene models

The following commands prepare the hippocampus inputs and fit the Decima-vector
model for seed 42:

```bash
python scripts/prepare_manuscript_inputs.py --cohort hippocampus \
  --manifest /inputs/hippocampus/manifest.tsv \
  --expression-root /inputs/hippocampus/expression \
  --embedding-root /inputs/hippocampus/uni2h \
  --gene-vectors /inputs/decima_vectors.npz --data-root /work/reproduction
python manuscript_workflows/launch.py --data-root /work/reproduction \
  experiments.multicohort_geneheldout_decima_clean_split.run_clean_geneheldout \
  --cohort hippocampus_donor_disjoint --seed 42 --variant decima
```

Staging validates the frozen panel/split, records hashes, links expression
inputs and writes feature/metadata adapters without modifying original inputs.
The driver fits six epochs, selects on training genes in validation individuals,
then exports held-out-gene endpoints and per-gene sufficient statistics.
Repeat `decima`, `random`, `constant` for seeds 42, 123, 456 and four cohorts
(`hippocampus_donor_disjoint`, `dlpfc`, `nac`, `her2st`): 36 primary runs.
Within your data workspace, checkpoints are written under
`experiments/multicohort_geneheldout_decima_clean_split/runs/` and per-run metrics
under `experiments/multicohort_geneheldout_decima_clean_split/components/runs/`.
Use a new workspace when changing the inputs.

After all 36 runs finish, summarize the results:

```bash
python manuscript_workflows/launch.py --data-root /work/reproduction \
  experiments.multicohort_geneheldout_decima_clean_split.summarize_clean_replication
```

The `results/` directory under the same experiment contains `per_run.tsv`,
`paired_effects.tsv` and `paired_effects_summary.tsv`. These give the individual
run values, matched pretrained-versus-control differences and seed summaries.
The [figure and table index](paper_code_map.md) connects the analyses to the paper.

## 4. Run the remaining analyses

Run modules with the same `launch.py --data-root ... MODULE [arguments]` form.
Prefixes below are under `experiments.`. Summary scripts without a cohort
option require completed runs from every cohort. Check the input column before
running a secondary analysis; several also need gene annotations or
section-to-individual maps.

| Analysis | Module(s) | Upstream requirements |
| --- | --- | --- |
| Primary summary | `multicohort_geneheldout_decima_clean_split.summarize_clean_replication` | All primary component exports |
| Mean-only ridge and matched controls | `heldout_gene_second_stage.build_gene_mean_baselines`, `build_matched_gene_mean_controls` | Matrices, vectors, primary endpoints |
| Section centring | `multicohort_geneheldout_decima.evaluate_section_centered --partition training_only` | Frozen checkpoints and individual metadata |
| Variance deciles | `heldout_gene_second_stage.build_decima_full_variance_deciles` | Training stratification, per-gene tables |
| Fixed top-50 | `paper.submission.scripts.build_training_defined_hvg_analysis` (no `experiments.` prefix) | Training moments, primary/control outputs |
| Bootstrap | `heldout_gene_second_stage.build_biological_uncertainty` | Individual/section effects and training-only cluster assignments |
| Covariates | `heldout_gene_second_stage.build_sequence_covariates`, `build_detection_and_covariate_controls` | Gene metadata, sequences, decoder outputs |
| Partitions | `heldout_gene_second_stage.build_gene_partitions`, `run_partition_control` | Training stratification, vectors, chromosome annotations |
| Identity/target-fitted | `heldout_gene_second_stage.run_decoder_control` | Primary inputs; `--condition shuffled_pretrained` or `fitted_target_oracle` |
| scGPT | `heldout_gene_second_stage.run_scgpt_gene_token_decoder --checkpoint whole_human` | Checkpoint/vocabulary/arguments, symbol mapping; matched variants fitted by driver |
| scGPT provenance | `heldout_gene_second_stage.audit_scgpt_provenance_and_coverage --checkpoint whole_human` | Upstream files and downstream gene lists |
| Architecture | `heldout_decoder_architecture_audit.run_bias_free`, `run_concat_mlp` | Primary inputs |
| Residual-only | `heldout_decoder_architecture_audit.run_residual_only` | Full sections; original signed targets, section batching, centring and selection retained |
| Branch intervention | `heldout_decoder_architecture_audit.evaluate_branch_intervention` | Frozen jointly fitted checkpoints |
| Fitted-gene variants | `current_full_ablation_common.train_balanced_ablation`, `evaluate_with_coverage`, `summarize_multiseed` | Cohort YAMLs, coordinates/graphs, training-only modality stats, gene anchors/inputs |
| External fitted benchmark | `hippocampus_donor_disjoint_external_benchmark.run_stnet`, `run_hist2st`, `run_bleep` | Cohort and upstream artifacts |
| STPath benchmark | `stpath_finetune_spatios2e.train`, `export_finetuned_outputs`, `eval_stpath_finetuned_only` | Donor-disjoint configs from `hippocampus_donor_disjoint_external_benchmark.prepare_benchmark --fit-stpath-adapter` |
| Tumour programme | `her2st_current_full_ablation.prepare_pathology_program`, `evaluate_pathology_program`, `summarize_multiseed_pathology` | Training programme, annotations, test predictions |

Filenames after a comma share the preceding package prefix. Fitted-gene YAMLs
and `current_full_ablation_common.write_configs` / `write_multiseed_configs`
preserve variant/seed definitions. Copy relevant configs into the data workspace
before using their relative paths. Fitted models additionally need 31 coordinate
descriptors, graphs and training-only modality stats; primary held-out staging
does not generate those inputs. See the fitted-example preparation commands.

For GeneQuery and DeepSpot-M, follow the
[external-model instructions](../experiments/external_genequery_component_audit/README.md).
`configs/manuscript/genequery_runs.json` records the 24 GeneQuery runs, including
evaluation batch size 8.

The archived BLEEP and ST-Net adapters resolve upstream source under
`<data-root>/third_party/BLEEP` and `<data-root>/third_party/ST-Net` by default.
Set `SPATIOS2E_BLEEP_ROOT` or `SPATIOS2E_STNET_ROOT` to use an existing checkout
elsewhere. These paths contain upstream source, not pretrained checkpoints.
