from pathlib import Path
import re

import yaml

import spatios2e


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.1.2"


def test_release_version_is_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())

    assert match.group(1) == EXPECTED_VERSION
    assert spatios2e.__version__ == EXPECTED_VERSION
    assert str(citation["version"]) == EXPECTED_VERSION


def test_release_license_uses_collective_holder_and_no_email():
    license_text = (ROOT / "LICENSE").read_text()
    citation_text = (ROOT / "CITATION.cff").read_text()

    assert "Copyright (c) 2026 SpatioS2E authors" in license_text
    assert "@" not in citation_text


def test_readme_overview_is_included_in_source_distribution():
    readme = (ROOT / "README.md").read_text()
    images = re.findall(r'(?:src="|\]\()(docs/assets/[^"\)]+\.png)', readme)
    manifest = (ROOT / "MANIFEST.in").read_text()

    assert images
    assert all((ROOT / path).is_file() for path in images)
    assert "recursive-include docs *.md *.png" in manifest
    assert "recursive-include examples *.sh *.txt *.md" in manifest
