# Sample splits, gene lists and model settings

Use these files to match the sample assignments, gene lists and model settings
used in the paper.

- `cohorts.yaml` records accession, image modality, biological unit and section,
  individual and gene-universe counts.
- `biological_splits/*.json` contains the exact section identifiers used for
  training, validation and testing.
- `gene_splits/*/{train,heldout}_genes.txt` contains the exact primary
  downstream gene partition used in the held-out-target analyses.
- `gene_splits_sensitivity/<partition>/*/{train,heldout}_genes.txt` contains
  the exact historical, expression-seed, embedding-cluster and
  chromosome-blocked sensitivity partitions.
- `gene_partition_manifest.json` records the count and SHA-256 digest of every
  primary and sensitivity gene list, together with the primary--historical
  overlap audit.
- `heldout_assay.yaml` records the matched decoder, optimization protocol,
  Decima and scGPT vector settings, controls and evaluation rules.
- `external_models.yaml` records upstream model versions and checksums without
  redistributing licensed checkpoints.
- `genequery_runs.json` gives the arguments for all 24 GeneQuery runs.

The primary gene lists are ordered stable Ensembl identifiers. Visium features
were intersected with the Decima-supported identifiers by exact string equality;
the mapping did not strip version suffixes or substitute aliases. HER2ST symbols
were retained only when they mapped uniquely to one identifier, and the universe
was restricted to symbols observed among training patients.

The scGPT analyses use the genes covered by its vocabulary within these same
partitions. Every scGPT vector condition uses the same covered genes.

Check the recorded splits with
`python scripts/build_manuscript_partition_manifest.py --check`.
