"""Keep the public documentation and weight-free entry point usable."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_quickstart_runs_without_external_inputs(tmp_path):
    run = subprocess.run(
        [sys.executable, str(ROOT / "examples/synthetic_components.py")],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    metrics = json.loads(run.stdout)
    assert metrics["mean_only"]["full_matrix_pcc"] > 0.9
    assert abs(metrics["mean_only"]["mean_gene_pcc"]) < 1e-10
    assert metrics["mean_and_spatial"]["mean_gene_pcc"] > 0.99


def test_public_markdown_relative_links_resolve():
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *ROOT.glob("docs/**/*.md")]
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            assert (document.parent / relative).exists(), (document, target)


def test_readme_asset_checksum_is_current():
    asset = ROOT / "docs/assets/figure_1_cde.png"
    assert hashlib.sha256(asset.read_bytes()).hexdigest() in (asset.parent / "README.md").read_text()


def test_archived_python_has_no_author_specific_absolute_roots():
    for path in (ROOT / "manuscript_workflows").rglob("*.py"):
        text = path.read_text()
        assert "/users/tchen2/" not in text, path
        assert "/dcs04/hicks/data/tchen2/" not in text, path
