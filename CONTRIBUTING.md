# Contributing

Bug reports, documentation improvements and focused pull requests are welcome.

For a reproducibility question, include the Git commit, Python/package versions,
command, expected and observed behaviour, and a minimal example with synthetic
inputs. Do not attach access tokens, restricted data, model weights or large
prediction matrices. Report suspected credential exposure privately to the
maintainers rather than quoting it in a public issue.

For code changes, work on a branch and run:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check spatios2e tests scripts experiments
python -m build
```

The exact tested Linux/CPU dependencies are in
`requirements-lock-linux-x86_64-py310.txt`. Optional preprocessing and upstream
model dependencies are documented separately in `docs/reproduction.md`.

Keep frozen partitions and historical validation records intact. Changes to
metric definitions, target eligibility, control assignments or training
selection require tests and an explicit explanation of their scientific effect.
Contributions to repository code are made under the MIT license; third-party
artifacts retain their own access and license terms.
