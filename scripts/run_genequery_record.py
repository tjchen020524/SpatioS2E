"""Render or execute one of the 24 archived GeneQuery training configurations."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=['frozen_features', 'end_to_end'], required=True)
    parser.add_argument('--seed', type=int, choices=[42, 123, 456], required=True)
    parser.add_argument('--variant', choices=['semantic', 'identity_shuffle', 'random', 'constant'], required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--execute', action='store_true', help='Otherwise print the exact command without training')
    args = parser.parse_args()
    records = json.loads((ROOT / 'configs/manuscript/genequery_runs.json').read_text())['runs']
    record = next(r for r in records if r['setting'] == args.setting and r['arguments']['seed'] == args.seed
                  and r['arguments']['variant'] == args.variant)
    command = [sys.executable, '-m', record['module']]
    for name, value in record['arguments'].items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    command.extend(['--feature-root' if args.setting == 'frozen_features' else '--patch-root',
                    str(args.input_root.resolve()), '--output-root', str(args.output_root.resolve())])
    if args.setting == 'end_to_end':
        if args.checkpoint is None:
            parser.error('End-to-end reconstruction requires an explicit ResNet --checkpoint')
        command.extend(['--checkpoint', str(args.checkpoint.resolve())])
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
