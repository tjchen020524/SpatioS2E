# Training and analysis scripts

This directory holds the cohort-specific training and analysis scripts.
Follow the [reproduction guide](../docs/reproduction.md) to prepare the inputs
and run a module with `launch.py --data-root /your/data/workspace MODULE`.

The launcher imports code from this directory and reads and writes data under
`--data-root`. The original and portable script hashes are recorded in
`source_manifest.json`.
