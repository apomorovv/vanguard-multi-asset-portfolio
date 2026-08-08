# Provenance publication note

This directory publishes the resolved JSON configurations for the full-hybrid
and stretch-scaling campaigns. The corresponding source archives also contain
machine-specific environment inventories. Those host records are intentionally
not copied into the Git branch; `../archive_manifest.csv` records the SHA-256
fingerprint of each complete source archive.

The stretch configuration's output directory is normalized from an absolute
workstation path to the equivalent repository-relative path. Its original
configuration hash is retained as `source_case_config_sha256`; no numerical,
solver, seed, or stopping setting is changed.

For a new run, use the published configurations together with the repository's
installation guidance, record the new machine and package environment locally,
and report any timeout or fallback exactly as observed.
