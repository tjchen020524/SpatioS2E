"""Exercise input staging and the original six-epoch primary driver on synthetic data.

The synthetic panel/sections are deliberately small and are not the manuscript
partitions. Decoder dimensions, fitting loop, selection and component exporter
are the archived scientific implementations. No weights or raw data are used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix='spatios2e-archived-primary-') as temporary:
        work = Path(temporary)
        records = work / 'fixtures/configs/manuscript'
        gene_dir = records / 'gene_splits/hippocampus_donor_disjoint'
        gene_dir.mkdir(parents=True)
        genes = [f'GENE{i}' for i in range(6)]
        (gene_dir / 'train_genes.txt').write_text('\n'.join(genes[:4]) + '\n')
        (gene_dir / 'heldout_genes.txt').write_text('\n'.join(genes[4:]) + '\n')
        biological = records / 'biological_splits/hippocampus.json'
        biological.parent.mkdir()
        split = {key: [key + '_section'] for key in ('train', 'val', 'test')}
        biological.write_text(json.dumps(split))
        rows = []
        for key, samples in split.items():
            sample = samples[0]
            count = 256 if key == 'train' else 32
            barcodes = np.asarray([f'spot{i}' for i in range(count)])
            rows.extend({'sample': sample, 'barcode': b, 'split': key} for b in barcodes)
            expression = sparse.csr_matrix(np.log1p(rng.poisson(3, (6, count))).astype(np.float32))
            path = work / 'expression' / sample / f'{key}.npz'
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, data=expression.data, indices=expression.indices,
                                indptr=expression.indptr, shape=expression.shape,
                                gene_ids=genes, barcodes=barcodes)
            features = work / 'features' / sample / 'embeddings.npz'
            features.parent.mkdir(parents=True)
            np.savez_compressed(features, barcodes=barcodes[::-1],
                                embeddings=rng.normal(size=(count, 1536)).astype(np.float32))
        manifest = work / 'manifest.tsv'
        pd.DataFrame(rows).to_csv(manifest, sep='\t', index=False)
        vectors = work / 'vectors.npz'
        np.savez_compressed(vectors, gene_ids=genes, embeddings=rng.normal(size=(6, 1920)).astype(np.float32))
        spec = importlib.util.spec_from_file_location('prepare_primary', ROOT / 'scripts/prepare_manuscript_inputs.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = work / 'fixtures'
        previous = sys.argv
        data_root = work / 'staged'
        try:
            sys.argv = ['prepare', '--cohort', 'hippocampus', '--manifest', str(manifest),
                        '--expression-root', str(work / 'expression'), '--embedding-root', str(work / 'features'),
                        '--gene-vectors', str(vectors), '--data-root', str(data_root)]
            module.main()
        finally:
            sys.argv = previous
        command = [sys.executable, str(ROOT / 'manuscript_workflows/launch.py'), '--data-root', str(data_root),
                   'experiments.multicohort_geneheldout_decima_clean_split.run_clean_geneheldout',
                   '--cohort', 'hippocampus_donor_disjoint', '--seed', '42', '--variant', 'decima']
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        completed = subprocess.run(command, cwd=work, env=environment, capture_output=True, text=True)
        tables = list(data_root.glob('experiments/multicohort_geneheldout_decima_clean_split/components/runs/'
                                     'hippocampus_donor_disjoint/seed_42/decima/*'))
        report = {'scope': __doc__, 'return_code': completed.returncode,
                  'status': 'PASS' if completed.returncode == 0 and tables else 'FAIL',
                  'stdout': completed.stdout.replace(str(work), '<synthetic-workspace>'),
                  'stderr': completed.stderr.replace(str(work), '<synthetic-workspace>'),
                  'component_exports': {p.name: p.read_text() for p in tables if p.suffix in {'.json', '.tsv'}},
                  'counts': {'training_spots': 256, 'validation_spots': 32, 'test_spots': 32,
                             'training_genes': 4, 'heldout_genes': 2, 'epochs': 6}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        if report['status'] != 'PASS':
            print(completed.stderr)
            raise SystemExit(1)
        print(f'PASS: original primary training and component exports; report {args.output}')


if __name__ == '__main__':
    main()
