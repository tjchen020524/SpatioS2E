"""Protect the input contracts and selected historical code archive without weights."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

from spatios2e.preprocessing.extract_uni2h import white_crop

ROOT = Path(__file__).resolve().parents[1]


def test_archive_is_hash_traced_and_python_syntax_is_valid():
    archive = ROOT / 'manuscript_workflows'
    records = json.loads((archive / 'source_manifest.json').read_text())['files']
    assert len(records) > 100
    for record in records:
        path = archive / record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['archived_sha256']
        assert path.suffix in {'.py', '.yaml', '.json'}
        if path.suffix == '.py':
            ast.parse(path.read_text())


def test_historical_decoder_computes_outside_development_root(tmp_path):
    archive = ROOT / 'manuscript_workflows'
    program = f'''
import os, sys
os.environ['SPATIOS2E_DATA_ROOT'] = {str(tmp_path)!r}
sys.path.insert(0, {str(archive)!r})
import torch
from experiments.geneheldout_uni2h_decima_decoder_noseq_controls import run_geneheldout_noseq_controls as base
assert str(base.ROOT) == {str(tmp_path)!r}
assert base.__file__.startswith({str(archive)!r})
model = base.GeneConditionedProgramDecoder(4, 3, 8, 2, 0)
pred = model(torch.ones(5, 4), torch.ones(6, 3))
assert pred.shape == (5, 6)
assert torch.isfinite(pred).all()
pred.square().mean().backward()
assert all(p.grad is not None for p in model.parameters())
'''
    subprocess.run([sys.executable, '-I', '-c', program], cwd=tmp_path, check=True, capture_output=True)


def test_her2_white_padding_matches_historical_crop():
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    patch = white_crop(image, 0, 0, 4)
    assert patch.shape == (4, 4, 3)
    assert np.all(patch[:2] == 255)
    assert np.all(patch[2:, 2:] == 0)


def test_genequery_records_cover_all_runs_and_freeze_actual_batch_sizes():
    records = json.loads((ROOT / 'configs/manuscript/genequery_runs.json').read_text())['runs']
    assert len(records) == 24
    keys = {(r['setting'], r['arguments']['seed'], r['arguments']['variant']) for r in records}
    assert len(keys) == 24
    for record in records:
        args = record['arguments']
        assert (args['batch_size'], args['eval_batch_size'], args['epochs'], args['patience']) == (4, 8, 100, 12)
        assert not args['no_amp']


def test_preparation_and_training_entrypoints_have_help(tmp_path):
    for relative in ('scripts/prepare_manuscript_inputs.py', 'scripts/run_genequery_record.py',
                     'manuscript_workflows/launch.py'):
        subprocess.run([sys.executable, str(ROOT / relative), '--help'], cwd=tmp_path,
                       check=True, capture_output=True)


def test_fitted_feature_statistics_exclude_validation_and_test(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('fitted_prepare', ROOT / 'scripts/prepare_fitted_features.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    split = {key: [key + '_section'] for key in ('train', 'val', 'test')}
    split_file = tmp_path / 'split.json'
    split_file.write_text(json.dumps(split))
    rows = []
    for key, samples in split.items():
        sample = samples[0]
        rows.extend({'sample': sample, 'barcode': b, 'split': key} for b in ('a', 'b'))
        spatial = tmp_path / 'spatial' / sample / 'spatial_features.csv.gz'
        spatial.parent.mkdir(parents=True)
        table = pd.DataFrame(np.zeros((2, 31)), columns=[f'c{i}' for i in range(31)])
        table.insert(0, 'barcode', ['b', 'a'])
        table.to_csv(spatial, index=False)
        image = tmp_path / 'images' / sample / 'embeddings.npz'
        image.parent.mkdir(parents=True)
        values = np.ones((2, 1536), dtype=np.float32) * (1 if key == 'train' else 100)
        np.savez_compressed(image, barcodes=['b', 'a'], embeddings=values)
    manifest = tmp_path / 'manifest.tsv'
    pd.DataFrame(rows).to_csv(manifest, sep='\t', index=False)
    output = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--manifest', str(manifest), '--biological-split', str(split_file),
                                    '--spatial-root', str(tmp_path / 'spatial'), '--embedding-root',
                                    str(tmp_path / 'images'), '--output-root', str(output), '--celltype-width', '18'])
    module.main()
    stats = json.loads((output / 'modality_stats.json').read_text())
    assert np.allclose(stats['histology']['mean'], 1)
    with np.load(output / 'train_section/multimodal_features.npz') as archive:
        assert archive['barcodes'].tolist() == ['a', 'b']
        assert not np.any(archive['celltype_weights'])
