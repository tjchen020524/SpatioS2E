"""Align prepared cohort inputs to the archived primary-workflow directory contract.

Run once per cohort in a new data workspace. Input expression is genes-by-spots
CSR, already normalized log(CPM+1). No normalization or split reselection occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unique_index(values, label):
    values = [str(x) for x in values]
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate identifiers: {label}')
    return {value: index for index, value in enumerate(values)}


def link_input(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(destination)
    else:
        destination.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, choices=['hippocampus', 'dlpfc', 'nac', 'her2st'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expression-root', type=Path, required=True)
    parser.add_argument('--embedding-root', type=Path, required=True)
    parser.add_argument('--gene-vectors', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    name = 'hippocampus_donor_disjoint' if args.cohort == 'hippocampus' else args.cohort
    records = ROOT / 'configs/manuscript'
    biological = records / 'biological_splits' / f'{args.cohort}.json'
    split = json.loads(biological.read_text())
    gene_dir = records / 'gene_splits' / name
    gene_lists = {key: (gene_dir / f'{key}_genes.txt').read_text().splitlines()
                  for key in ('train', 'heldout')}
    genes = sorted(gene_lists['train'] + gene_lists['heldout'])
    unique_index(genes, 'frozen gene panel')
    manifest = pd.read_csv(args.manifest, sep='\t', dtype=str)
    if not {'sample', 'barcode', 'split'} <= set(manifest):
        raise ValueError('Manifest requires sample, barcode and split columns')
    if manifest.duplicated(['sample', 'barcode']).any():
        raise ValueError('Manifest has duplicate sample/barcode pairs')
    sample_split = {s: key for key in ('train', 'val', 'test') for s in split[key]}
    if set(manifest['sample']) != set(sample_split):
        raise ValueError('Manifest samples must exactly match the frozen biological split')
    if (manifest['sample'].map(sample_split) != manifest['split']).any():
        raise ValueError('Manifest violates the frozen biological partition')
    metadata = args.data_root / 'experiments/multicohort_geneheldout_decima/data' / name
    cohort_root = args.data_root / f'experiments/{args.cohort}_current_full_ablation/data'
    if (metadata / 'input_provenance.json').exists():
        raise FileExistsError('Prepared cohort already exists; use a new data workspace')
    with np.load(args.gene_vectors, allow_pickle=False) as archive:
        vector_index = unique_index(archive['gene_ids'], 'gene vectors')
        if any(gene not in vector_index for gene in genes):
            raise ValueError('Decima vectors do not cover the frozen cohort panel')
        if archive['embeddings'].shape[1] != 1920:
            raise ValueError('Primary Decima vectors must have 1920 columns')
    provenance = [{'role': 'manifest', 'sha256': digest(args.manifest)},
                  {'role': 'Decima vectors', 'sha256': digest(args.gene_vectors)},
                  {'role': 'biological split', 'sha256': digest(biological)}]
    for sample, rows in manifest.groupby('sample', sort=False):
        assignment = sample_split[sample]
        expression = args.expression_root / sample / f'{assignment}.npz'
        features = args.embedding_root / sample / 'embeddings.npz'
        with np.load(expression, allow_pickle=True) as archive:
            if archive['gene_ids'].astype(str).tolist() != genes:
                raise ValueError(f'{sample}: expression rows must match sorted frozen gene panel')
            barcodes = unique_index(archive['barcodes'], f'{sample} expression')
            shape = tuple(archive['shape'])
            matrix = sparse.csr_matrix((archive['data'], archive['indices'], archive['indptr']), shape=shape)
            if shape != (len(genes), len(barcodes)) or not np.isfinite(matrix.data).all():
                raise ValueError(f'{sample}: invalid expression matrix')
            if any(b not in barcodes for b in rows['barcode']):
                raise ValueError(f'{sample}: expression lacks manifest barcodes')
        with np.load(features, allow_pickle=True) as archive:
            feature_barcodes = archive['barcodes'].astype(str)
            feature_index = unique_index(feature_barcodes, f'{sample} features')
            embeddings = archive['embeddings'].astype(np.float32)
            if embeddings.shape != (len(feature_index), 1536) or not np.isfinite(embeddings).all():
                raise ValueError(f'{sample}: expected finite 1536-dimensional UNI2-h features')
            if any(b not in feature_index for b in rows['barcode']):
                raise ValueError(f'{sample}: features lack manifest barcodes')
        out = cohort_root / 'multimodal' / sample / 'multimodal_features.npz'
        if out.exists():
            raise FileExistsError(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, barcodes=feature_barcodes, histology_embeddings=embeddings)
        link_input(expression, cohort_root / 'expression' / sample / f'{assignment}.npz')
        provenance.extend({'sample': sample, 'role': role, 'sha256': digest(path)}
                          for role, path in [('expression', expression), ('UNI2-h', features)])
    metadata.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(metadata / 'manifest.tsv', sep='\t', index=False)
    (metadata / 'gene_ids.txt').write_text('\n'.join(genes) + '\n')
    split['embedding_dim'] = 1536
    (metadata / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
    # Keep both archival gene-partition locations populated for secondary drivers.
    for target in (metadata / 'gene_splits', args.data_root /
                   'experiments/multicohort_geneheldout_decima_clean_split/data' / name / 'gene_splits'):
        for key in gene_lists:
            link_input(gene_dir / f'{key}_genes.txt', target / f'{key}_genes.txt')
    link_input(args.gene_vectors, args.data_root /
               'experiments/sample_split_fullgenes_v12_uni2h_direct_seq_residual_no_celltype/artifacts/'
               'decima_gene_embeddings_fullgenes.npz')
    (metadata / 'input_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Prepared {name}: {len(manifest)} spots, {len(genes)} genes at {metadata}')


if __name__ == '__main__':
    main()
