"""Prepare reference-sequence inputs and export frozen Decima gene vectors.

Requires separately installed Decima 0.5.1 and operator-supplied metadata,
reference FASTA and checkpoint. No third-party artifact is downloaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sequence = commands.add_parser('sequences', help='Construct strand-aware sequence plus gene-body mask')
    sequence.add_argument('--genes', type=Path, required=True)
    sequence.add_argument('--metadata', type=Path, required=True, help='Decima metadata AnnData file')
    sequence.add_argument('--fasta', type=Path, required=True)
    sequence.add_argument('--output-dir', type=Path, required=True)
    vectors = commands.add_parser('vectors', help='Mean-pool the frozen 1920-channel final feature map')
    vectors.add_argument('--genes', type=Path, required=True)
    vectors.add_argument('--input-dir', type=Path, required=True)
    vectors.add_argument('--checkpoint', type=Path, required=True)
    vectors.add_argument('--output', type=Path, required=True)
    vectors.add_argument('--batch-size', type=int, default=1)
    vectors.add_argument('--device', default='cuda')
    args = parser.parse_args()
    genes = [g.strip() for g in args.genes.read_text().splitlines() if g.strip()]
    if not genes or len(set(genes)) != len(genes):
        raise ValueError('Gene list must be nonempty with unique exact Ensembl identifiers')
    record = {'gene_list_sha256': sha256(args.genes), 'n_genes': len(genes), 'inputs': []}
    if args.command == 'sequences':
        from decima import DecimaResult
        result = DecimaResult.load(str(args.metadata))
        metadata = result.gene_metadata.dropna(subset=['gene_id'])
        mapping = metadata.reset_index().set_index('gene_id')['index'].to_dict()
        missing = [g for g in genes if g not in mapping]
        if missing:
            raise ValueError(f'{len(missing)} genes absent from Decima metadata: {missing[:5]}')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for gene in genes:
            target = args.output_dir / f'{gene}.npz'
            if target.exists():
                raise FileExistsError(target)
            sequence, mask = result.prepare_one_hot(mapping[gene], genome=str(args.fasta.resolve()))
            array = np.vstack([sequence, mask]).astype(np.float32)
            if array.shape != (5, 524288):
                raise ValueError(f'{gene}: unexpected sequence/mask shape {array.shape}')
            np.savez_compressed(target, ensembl_id=gene, gene_name=mapping[gene], seq_mask=array)
            record['inputs'].append({'gene': gene, 'sha256': sha256(target)})
        record.update(metadata_sha256=sha256(args.metadata), fasta_sha256=sha256(args.fasta))
        report = args.output_dir / 'sequence_provenance.json'
    else:
        from spatios2e.models.decima_wrapper import DecimaSequenceWrapper
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.batch_size < 1:
            raise ValueError('--batch-size must be positive')
        # The wrapper has a zero-fill fallback; reference export must reject missing inputs.
        for gene in genes:
            path = args.input_dir / f'{gene}.npz'
            if not path.is_file():
                raise FileNotFoundError(path)
            record['inputs'].append({'gene': gene, 'sha256': sha256(path)})
        model = DecimaSequenceWrapper(ckpt_path=args.checkpoint, h5_path=None, npz_dir=args.input_dir,
                                      freeze_backbone=True, freeze_head=True).to(args.device).eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(genes), args.batch_size):
                chunks.append(model.encode_genes(genes[start:start + args.batch_size],
                                                device=torch.device(args.device)).cpu().float().numpy())
        embeddings = np.concatenate(chunks)
        if embeddings.shape != (len(genes), 1920) or not np.isfinite(embeddings).all():
            raise ValueError(f'Invalid Decima export: {embeddings.shape}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, gene_ids=np.asarray(genes), embeddings=embeddings)
        record.update(checkpoint_sha256=sha256(args.checkpoint), output_sha256=sha256(args.output))
        report = args.output.with_suffix('.provenance.json')
    report.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Wrote {report}')


if __name__ == '__main__':
    main()
