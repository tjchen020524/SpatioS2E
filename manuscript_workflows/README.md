# Custom manuscript workflows

Start with [the reproduction guide](../docs/reproduction.md). This directory
contains selected custom scripts, their local import dependencies and final
fitted-gene configurations, not the full development repository or third-party
code/weights. Original and archived hashes are in `source_manifest.json`.

Use `launch.py` with an explicit separate data workspace. It puts archived code
first on the import path and resolves historical data/output roots through
`workflow_paths.py`. No inputs are downloaded by the launcher. Scientific
scripts can train models or generate outputs when invoked; inspect their
argument parsers before launching large workflows.
