# Manuscript reproduction

This candidate contains the package and a selected archive of the actual custom
analysis scripts. `manuscript_workflows/source_manifest.json` records original
and archived source hashes. Only development-root assignments and import-path
priority were made portable. Full GPU/cohort experiments were not rerun as part
of release validation.

## Environments and preparation

The pinned CPU lock validates the package without external weights. For input
extraction and historical GPU workflows, use a separate environment:

```bash
python -m pip install -e '.[preprocess,analysis]'
python -m pip install 'decima==0.5.1'
```

Decima is a separate code dependency, not just a checkpoint; see its
[official installation and artifact instructions](https://github.com/Genentech/decima).
Obtain its metadata AnnData, `rep0.ckpt` and reference hg38 FASTA separately.
UNI2-h requires the [provider's access approval](https://huggingface.co/MahmoodLab/UNI2-h).
Revisions and hashes are in `configs/manuscript/external_models.yaml`.
Third-party code, weights and derived outputs retain their respective licenses.
The CPU lock is not a historical GPU environment specification.

### UNI2-h

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

### Decima sequences and fixed vectors

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
genes only. Batch size is a memory setting, not a feature definition.

### Expression and identity contract

Archived `prepare_hippocampus.py`, `prepare_dlpfc.py`, `prepare_nac.py` and
`prepare_her2st.py` retain the actual cohort filtering, normalization, identifier
mapping and zero-filling procedures. Raw data must be acquired separately using
the manuscript accessions. For generic Visium conversion, see
`spatios2e-normalize-visium --help`.

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

## Complete primary run path

Stage a cohort into a new data workspace, then run the original driver:

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
Checkpoints and component tables are written under
`experiments/multicohort_geneheldout_decima_clean_split/{runs,components/runs}/`.
Do not reuse output directories for a different input snapshot.

For a bounded real-input CPU check, use
`scripts/validate_real_heldout_smoke.py --help`; its subset/one-epoch metrics are
not manuscript estimates. Weight-free fitting is tested in pytest and in the
independently installed wheel validation outside the repository.

## Other archived scientific workflows

Run modules with the same `launch.py --data-root ... MODULE [arguments]` form.
Prefixes below are under `experiments.`. These are actual research scripts;
their explicit input/output filenames remain visible in source. Aggregators
without a cohort option expect all upstream cohort/run files and do not create
missing results. Secondary workflows also require source metadata such as
gene annotations and section-to-individual maps.

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

GeneQuery/DeepSpot-M remain in top-level
`experiments/external_genequery_component_audit/`; use their dedicated README.
`configs/manuscript/genequery_runs.json` records all 24 historical invocations,
including evaluation batch size 8, rather than relying on defaults.

## Artifact availability

This Git/source archive contains code, configs, partitions and validation.
Numerical Source Data accompany the manuscript submission but are not currently
a separate downloadable deposit from this candidate. Raw images/counts,
third-party weights, trained checkpoints and dense predictions are not included.
Public release and an immutable deposit link await author approval; package
version 1.1.0 does not imply a new public release, tag or DOI.
