#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BASE_PYTHON NEW_VENV REPORT_JSON" >&2
  exit 2
fi

base_python=$1
new_venv=$2
report_json=$3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -e "$new_venv" ]]; then
  echo "refusing to reuse an existing clean-room path: $new_venv" >&2
  exit 2
fi

"$base_python" -m venv --copies "$new_venv"
"$new_venv/bin/python" -m pip install \
  --requirement "$repo_root/requirements-lock-linux-x86_64-py310.txt"
"$new_venv/bin/python" -m pip install --no-deps --no-build-isolation "$repo_root"
"$new_venv/bin/python" "$repo_root/scripts/validate_cleanroom.py" --output "$report_json"
