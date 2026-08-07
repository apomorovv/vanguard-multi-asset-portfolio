# Complete Benchmark Evidence Archive

This directory publishes the complete set of scientifically relevant files
from the three result archives used in the final report. The files are retained
with their original experiment names so that figures can be traced to the CSV,
JSON, configuration, and report files that generated them.

## Archive coverage

| Source directory | Main contents |
|---|---|
| `presentation_benchmark_suite/` | Exact certification, the 100-asset case, all-constraint tests, scenario and preference sweeps, repeated scaling, fixed-window comparisons, and IBM QPU evidence. |
| `large_example/` | The independent 250-asset continuous and equal-lot classical benchmark, complete weights, checks, diagnostics, and plots. |
| `hybrid_scaling/` | The repeated full-hybrid scaling study from 250 to 20,000 assets. |

The unpacked evidence set contains 224 files:

- 73 CSV tables;
- 24 JSON records;
- 64 PNG figures;
- 52 PDF figures or documents;
- six Markdown reports;
- four text timing records; and
- one YAML configuration.

The original ZIP byte counts and SHA-256 digests are preserved in
[`source_archive_manifest.csv`](source_archive_manifest.csv). The ZIP
containers themselves are not republished because they include unredacted
machine-environment snapshots. All scientific result files are unpacked in
this directory.

## Integrity and the large diagnostic file

[`archive_file_manifest.csv`](archive_file_manifest.csv) records the size,
SHA-256 digest, repository path, and storage form of every one of the 224
original files. One diagnostics file,
`presentation_benchmark_suite/03_all_constraints_10000_scenario_hybrid/hybrid_diagnostics.json`,
is 73.5 MB. To remain comfortably within GitHub's file-transfer limits, its
browser-tree copy is stored losslessly as `hybrid_diagnostics.json.gz`; the
decompressed bytes match the original JSON digest recorded in the manifest.

To recover the direct JSON copy:

```bash
gzip -dk hybrid_diagnostics.json.gz
```

## Deliberate exclusions and publication redactions

The original ZIPs contain macOS resource-fork metadata and automatically
created Jupyter checkpoint duplicates. Those items are not experimental
results and are omitted from the unpacked tree.

Three environment snapshots and the environment block in
`large_example/benchmark_metadata.json` originally recorded local host,
CPU/GPU, platform, executable, and installed-package metadata. Their
repository copies retain timestamps, experiment commit information where
present, an explicit publication note, and the original SHA-256 digest; the
machine metadata is removed. Three absolute local paths in two configuration
files are normalized to repository-relative paths. These redactions do not
change portfolio inputs, solver outputs, timing tables, constraint checks,
QPU records, or figures.

## Suggested reading order

1. Start with the
   [final technical report](../../docs/portfolio_optimization_report.md) for
   the evidence narrative.
2. Inspect the
   [original evidence-backed claims](presentation_benchmark_suite/winning_claims.md)
   and their interpretation limits.
3. Use the CSV tables for numerical claims and the row-level constraint files
   for feasibility checks.
4. Use JSON and YAML files for configurations and solver provenance; the
   publication notes identify the redacted environment snapshots.
5. Treat PNG and PDF files as visual summaries of the machine-readable tables,
   not as substitutes for them.
