# Release and Zenodo archival procedure

This document records the release procedure for repository maintainers. The
order matters: Zenodo must be connected before the corresponding GitHub
release is created.

## 1. Verify the release candidate

From a clean `main` branch:

```bash
python -m pytest -q
python -m build
git status --short
```

Check that the package version, `CITATION.cff`, changelog and release notes all
name the same release. Do not include private data, model weights, checkpoints,
predictions or cluster logs in the tag.

## 2. Enable GitHub integration in Zenodo

1. Sign in to Zenodo with the GitHub account that can access the repository.
2. Open the profile menu, choose **GitHub**, and select **Sync now**.
3. Find `tjchen020524/SpatioS2E` and enable its integration toggle.
4. Refresh the page and confirm that the repository remains enabled.

The repository intentionally uses `CITATION.cff` as its single metadata source.
Do not add `.zenodo.json` unless confirmed funding, community or other
Zenodo-specific metadata is required: when both files are present, Zenodo
ignores `CITATION.cff` during GitHub archiving.

## 3. Create the immutable release

After integration is confirmed, create an annotated `v0.2.0` tag at the audited
commit and publish a GitHub release titled `SpatioS2E v0.2.0`. Use
[`docs/releases/v0.2.0.md`](releases/v0.2.0.md) as the release notes. Do not mark
the release as a prerelease.

Zenodo should ingest the new GitHub release automatically. Processing can take
time; inspect the repository entry on Zenodo if it reports a metadata error.

## 4. Verify and cite the archive

On the Zenodo record:

- confirm the title, version, authors, open access and MIT license;
- confirm that the archived source resolves to the intended Git commit;
- check **External resources > Archived in** for the preservation status;
- record both the version DOI and the concept DOI.

Use the version DOI in the manuscript's Code availability statement to identify
the exact analyzed release. The concept DOI is appropriate for references that
should resolve to the newest release. Once assigned, add the DOI links and
badge to `README.md` on `main`; do not move or replace the published tag.
