# Final submission evidence

This directory is the audit trail for the final report and presentation. Start
with [`evidence_summary.csv`](evidence_summary.csv) for the result overview and
[`claim_evidence_map.csv`](claim_evidence_map.csv) to trace every headline claim
to its source table.

## Reading order

1. [`evidence_summary.csv`](evidence_summary.csv) — strongest findings and their
   evidence level.
2. [`claim_evidence_map.csv`](claim_evidence_map.csv) — claim-to-file mapping and
   the qualification that must accompany each claim.
3. [`classical/`](classical/) — continuous cross-check, exact sparse cases,
   warm starts, robustness, and the 250-asset equal-lot benchmark.
4. [`constraints/`](constraints/) — row-level and family-level validation for
   the certified and 10,000-scenario gauntlets.
5. [`preferences/`](preferences/) — scenario-tail and investor-preference
   trade-offs.
6. [`scaling/`](scaling/) — repeated full-hybrid and stretch runs, including
   time, memory, fallback, and bound fields.
7. [`quantum/`](quantum/) — controlled local methods, width-depth evidence,
   IBM provenance, hardware survival, fairness, and QUBO alignment.
8. [`figures/`](figures/) — presentation views backed by the preceding tables.
9. [`provenance/`](provenance/) and
   [`archive_manifest.csv`](archive_manifest.csv) — resolved configurations,
   a publication note, archive fingerprints, and integrity status.

## Evidence labels

- **Globally certified:** exhaustive enumeration or a mixed-integer incumbent
  with a matching solver bound.
- **Bounded:** a valid sparse result compared with a solved continuous lower
  bound, without a matching mixed-integer certificate.
- **Validated heuristic:** the portfolio passes every financial rule, but no
  global quality bound is available.
- **Hardware observation:** measured circuit behavior on a named QPU. This does
  not establish quantum advantage.

## Headline results

| Result | Evidence | Interpretation |
|---|---|---|
| Four continuous backends agree within \(1.41\times10^{-9}\) | `classical/continuous_backend_crosscheck.csv` | Independent numerical triangulation |
| 100-asset sparse optimum \(-0.0384147146\), 0.0% gap | `classical/main_100_asset_hybrid_summary.csv` | Global certificate for the stated model |
| 17 families and 244 checks pass | `constraints/all_constraints_certified_*` | Full guardrail certificate |
| 17 families and 858 checks pass at 10,000 scenarios | `constraints/all_constraints_10000_scenario_*` | Scenario-rich validated heuristic |
| 21/21 full-hybrid runs through 20,000 assets | `scaling/full_hybrid_to_20000_*` | Repeated zero-breach end-to-end scale |
| 27/27 stretch runs through 300,000 assets | `scaling/stretch_to_300000_*` | Repeated safe engineering scale; not a global optimum |
| 29.76 MiB factor arrays vs 670.55 GiB dense covariance at 300,000 assets | `scaling/global_scaling_summary.csv` | Measured factor storage and analytical dense storage |
| IBM test through 28 qubits; 30/30 valid after allocation | `quantum/ibm_qpu_*` | Hardware integration does not bypass safety |
| QPU beats matched random in only 6/30 strict comparisons | `quantum/candidate_pool_fairness_summary.csv` | No quantum-advantage claim |

## What is intentionally not duplicated

The supplied archives include large raw diagnostic JSON, raw IBM count payloads,
duplicate PDF/PNG render pairs, and notebook-checkpoint copies. The curated Git
package keeps the tables, certificates, provenance, and figures required to
audit the paper without adding more than 70 MiB of duplicated diagnostics. The
SHA-256 values in `archive_manifest.csv` identify the complete source archives.
Machine-specific environment inventories are also left in those fingerprinted
source archives instead of being published; the resolved run configurations
remain in `provenance/`.

## Interpretation rules

- Lower objective values are better; the objective is not a return percentage.
- A continuous relaxation may have a lower objective because it is not the
  exact-cardinality sparse problem.
- Blank relaxation gaps above 20,000 assets are deliberate: the guide timed out
  or fell back and did not supply a solved bound.
- A valid QPU sample is a support proposal, not a final allocation.
- Synthetic backtests and generated universes are reproducibility tests, not
  forecasts or financial advice.
