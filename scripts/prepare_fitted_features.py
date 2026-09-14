"""Assemble barcode-aligned fitted-gene inputs with zero composition channels.

Spatial CSVs are emitted by spatios2e-prepare-spatial-features; statistics use
only sections marked train in the supplied frozen biological split.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spatios2e.data.dataset import ModalityNormalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--biological-split', type=Path, required=True)
    parser.add_argument('--spatial-root', type=Path, required=True)
    parser.add_argument('--embedding-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--celltype-width', type=int, required=True,
                        help='Reserved zero-channel width; 18 for the hippocampus fitted-gene inputs')
    args = parser.parse_args()
    if args.celltype_width < 1:
        raise ValueError('Reserved channel width must be positive')
    manifest = pd.read_csv(args.manifest, sep='\t', dtype=str)
    split = json.loads(args.biological_split.read_text())
    expected = {s: key for key in ('train', 'val', 'test') for s in split[key]}
    if len(expected) != sum(len(split[key]) for key in ('train', 'val', 'test')):
        raise ValueError('Biological split contains repeated sections')
    if not {'sample', 'barcode', 'split'} <= set(manifest) or manifest.duplicated(['sample', 'barcode']).any():
        raise ValueError('Invalid spot manifest')
    if set(manifest['sample']) != set(expected) or (manifest['sample'].map(expected) != manifest['split']).any():
        raise ValueError('Manifest violates the frozen biological split')
    for sample, rows in manifest.groupby('sample', sort=False):
        target = args.output_root / sample / 'multimodal_features.npz'
        if target.exists():
            raise FileExistsError(target)
        spatial = pd.read_csv(args.spatial_root / sample / 'spatial_features.csv.gz', dtype={'barcode': str})
        if spatial['barcode'].duplicated().any() or len(spatial.columns) != 32:
            raise ValueError(f'{sample}: expected unique barcodes and 31 coordinate descriptors')
        spatial = spatial.set_index('barcode')
        with np.load(args.embedding_root / sample / 'embeddings.npz', allow_pickle=False) as archive:
            barcodes = archive['barcodes'].astype(str)
            if len(set(barcodes)) != len(barcodes):
                raise ValueError(f'{sample}: duplicated feature barcodes')
            index = {b: i for i, b in enumerate(barcodes)}
            order = rows['barcode'].to_numpy(dtype=str)
            image = archive['embeddings'][[index[b] for b in order]].astype(np.float32)
        coords = spatial.loc[order].to_numpy(np.float32)
        if image.shape != (len(order), 1536) or not np.isfinite(image).all() or not np.isfinite(coords).all():
            raise ValueError(f'{sample}: invalid image/coordinate features')
        zeros = np.zeros((len(order), args.celltype_width), dtype=np.float32)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, barcodes=order, spatial_features=coords,
                            spatial_feature_names=spatial.columns.to_numpy(dtype=str),
                            histology_embeddings=image, celltype_weights=zeros,
                            celltype_names=np.asarray([f'reserved_{i}' for i in range(args.celltype_width)]),
                            features=np.concatenate([coords, image, zeros], axis=1))
    stats = args.output_root / 'modality_stats.json'
    if stats.exists():
        raise FileExistsError(stats)
    ModalityNormalizer.compute(split['train'], args.output_root).save(stats)
    print(f'Wrote aligned features and training-only statistics to {args.output_root}')


if __name__ == '__main__':
    main()
