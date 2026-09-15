"""Check validation-report portability and archived file integrity."""
import hashlib
import importlib.util
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("user_root", ["/home/researcher", "/users/analyst", "/Users/researcher"])
def test_validation_logs_redact_home_directories(user_root):
    spec = importlib.util.spec_from_file_location("validate_cleanroom", ROOT / "scripts/validate_cleanroom.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    message = f"WARNING: The directory '{user_root}/.cache/pip' is not writable."
    assert module._portable_text(message) == "WARNING: The directory '<user-home>/.cache/pip' is not writable."


def test_archived_reports_exclude_machine_local_home_paths():
    home_path = re.compile(r"/(?:home|users|Users)/[^/\s\"'<>]+/")
    for path in (ROOT / "validation").glob("*.json"):
        assert not home_path.search(path.read_text()), path


def test_validation_checksums_match_archived_files():
    for row in (ROOT / "validation/checksums.sha256").read_text().splitlines():
        digest, relative = row.split(maxsplit=1)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
