# Data and pretrained models

Download the cohort data and model weights separately before preparing the
inputs. They are not bundled with the code.

## Cohort data

| Cohort | Data source |
| --- | --- |
| Anterior hippocampus | [GSE264692](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264692) |
| DLPFC | [GSE307403](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307403) |
| Nucleus accumbens | [GSE307586](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307586); [processed data](https://doi.org/10.5281/zenodo.17089020) |
| HER2ST | [Count matrices, images and annotations](https://doi.org/10.5281/zenodo.4751624) |

Sample assignments and gene lists are in
[`configs/manuscript/`](../configs/manuscript/README.md).
Follow each provider's access terms and use the paper's preprocessing steps.

## Pretrained models

- [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h): image encoder checkpoint.
- [Decima](https://github.com/Genentech/decima): package, checkpoint, gene metadata
  and reference genome for sequence preparation.
- [scGPT](https://github.com/bowang-lab/scGPT): whole-human checkpoint, vocabulary
  and model arguments.
- For GeneQuery, DeepSpot-M and their image encoders, use the
  [external-model instructions](../experiments/external_genequery_component_audit/README.md).

Check downloaded files against the revisions and hashes in
[`external_models.yaml`](../configs/manuscript/external_models.yaml).
The weights remain subject to their providers' licenses.

## Results

Numerical Source Data and Supplementary Tables are separate manuscript files.
Running the [experiments](reproduction.md) produces the downstream checkpoints,
prediction matrices and summary tables.
