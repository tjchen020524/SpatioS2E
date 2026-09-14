"""Run one archived scientific module; no data are downloaded automatically."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('module', help='Dotted archived module name')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    target = root.joinpath(*args.module.split('.')).with_suffix('.py')
    if not target.is_relative_to(root) or not target.is_file():
        parser.error('Module must identify a Python file inside the workflow archive')
    if not args.data_root.is_dir():
        parser.error('--data-root must be an existing prepared-data directory')
    os.environ['SPATIOS2E_DATA_ROOT'] = str(args.data_root.resolve())
    sys.path[:0] = [str(root), str(target.parent), str(root / 'scripts')]
    os.chdir(args.data_root)
    sys.argv = [str(target), *args.arguments]
    runpy.run_module(args.module, run_name='__main__')


if __name__ == '__main__':
    main()
