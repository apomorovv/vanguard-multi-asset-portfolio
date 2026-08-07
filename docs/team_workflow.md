# Team Workflow and Contribution Record

This document assigns ownership for our  team and defines how model, quantum, and engineering changes are reviewed.

## Team and primary roles

| Team member | Primary role | Accountable deliverables |
|---|---|---|
| **Andrei Pomorov** | Project, mathematical-modeling, and optimization lead | Financial formulation; classical mean-variance, exact-cardinality, and LNS methods; hybrid decomposition; scenario-risk design; classical validation and optimality interpretation; integration of classical and quantum results. |
| **Ilia Fazeli** | Quantum algorithms and hardware lead | QUBO/Ising mapping review; XY-QAOA circuit and fixed-weight subspace methods; simulator and IBM QPU protocols; transpilation and noise-management studies; quantum diagnostics and fair quantum-versus-classical comparisons. |
| **Kirankumar Dhanireddy** | Software engineering, reproducibility, and prototype lead | Repository architecture; canonical schemas and interfaces; packaging, configurations, scripts, tests, and release checks; result/artifact pipeline; Streamlit portfolio co-pilot and reproducible execution workflow. |

Andrei is accountable for overall scientific consistency, Ilia for the quantum
evidence, and Kirankumar for codebase and release quality. Accountability does
not replace review: no team member approves their own cross-cutting change.

## Challenge-requirement ownership

`A` means accountable for final sign-off, `R` means responsible for the work,
and `C` means required reviewer or contributor.

| Challenge requirement | Andrei | Ilia | Kirankumar | Required evidence |
|---|:---:|:---:|:---:|---|
| 1. Binary variables, linear constraints, quadratic objective | A/R | C | C | Mathematical model, schema, and objective-equivalence tests. |
| 2. Quantum-compatible cost function | A | R | C | QUBO/Ising derivation, energy checks, and decoding tests. |
| 3. Synthetic return, volatility, correlation, and cost data | A | C | R | Deterministic generators, seeds, PSD checks, and privacy-safe identifiers. |
| 4. Mean-variance baseline, realistic constraints, scenario penalties | A/R | C | R | Classical baselines, constraint tests, and the held-out CVaR-penalty sweep. |
| 5. Growth, income, drawdown, and cost controls | A/R | C | R | Preference presets and co-pilot controls using the canonical model. |
| 6. Risk, return, turnover, breaches, and explainability comparison | A | R for quantum metrics | R for tables/figures | Common result schema, independent checks, and auditable plots. |
| 7. Classical validation | A | C | R | Fixed-support QP, optional Gurobi reference, validator, and regression tests. |
| 8. Presentation and working demonstration | A | R for quantum section | R for prototype/demo | Evidence-backed claims, section sign-offs, and a clean live run. |
| 9. Portfolio co-pilot | C | C | A/R | Recommended allocation, baseline trade-offs, guardrail explanations, and infeasibility messages. |
| 10. Best risk-adjusted result with zero hard breaches | A | C | R | Final validator certificate and no post-solve clipping or renormalization. |

## Workstream responsibilities

### Andrei Pomorov - model, classical optimization, and integration

- Own the canonical financial notation, units, assumptions, and objective.
- Own the continuous relaxation, fixed-support allocation logic, classical LNS,
  exact/MIQP references, and correct optimality language.
- Define realistic hard guardrails and the scenario-based CVaR preference
  experiment; verify the risk-return interpretation of every plot.
- Integrate quantum support proposals with the same allocation oracle and
  independent validator used for classical proposals.
- Approve final claims about risk, return, feasibility, optimality, and
  scalability.

### Ilia Fazeli - quantum algorithms, simulation, and IBM hardware

- Own the QUBO-to-Ising and bitstring conventions and review energy-equivalence
  tests against the classical surrogate.
- Own XY-QAOA ansatz choices, fixed-Hamming-weight initialization/mixers, angle
  optimization protocol, and simulator comparisons.
- Own IBM backend selection, transpilation records, job manifests, raw-count
  provenance, mitigation ablations, and hardware-cost reporting.
- Compare QPU candidates with exact/best-known QUBO states, matched random
  screening, and classical LNS at the same allocation-oracle budget.
- Approve all quantum claims and ensure that hardware validation is not
  described as quantum advantage.

### Kirankumar Dhanireddy - codebase, testing, artifacts, and co-pilot

- Own package layout, dependency specifications, configuration parsing, command
  line scripts, notebook contracts, and portable installation instructions.
- Maintain canonical interfaces so classical and quantum modules do not create
  competing problem schemas or objective definitions.
- Own automated tests, reproducibility checks, artifact checksums, compact
  CSV/JSON outputs, and presentation-figure generation.
- Own the Streamlit co-pilot workflow and ensure it uses validated solver
  results rather than duplicating optimization logic.
- Approve release readiness: clean install, passing portable tests, safe
  optional-dependency behavior, and a reproducible notebook run.

## Code ownership and required reviewers

| Area | Primary owner | Required reviewer |
|---|---|---|
| `schemas.py`, `portfolio_model.py`, `classical*.py`, `allocation.py`, `hybrid.py` | Andrei | Kirankumar; Ilia when a QUBO interface changes |
| `qubo_builder.py`, `quantum_solver.py`, quantum experiment cells, IBM protocol | Ilia | Andrei for mathematical meaning; Kirankumar for interfaces and persistence |
| `data_generation.py`, configuration and execution scripts | Kirankumar | Andrei for financial assumptions |
| `validation.py`, metrics, result schemas, and artifact generation | Kirankumar | Andrei for tolerances and financial definitions |
| `copilot_app.py` | Kirankumar | Andrei for explanations and controls; Ilia for quantum-status wording |
| Benchmark notebook | Andrei for scientific design | Ilia signs quantum sections; Kirankumar signs execution/reproducibility sections |
| Final report and presentation | Andrei coordinates | All three review and sign their attributed sections |

## Branch policy

`main` must remain runnable and reviewed. Completed units should be merged in
small pull requests rather than accumulated in one final challenge-day merge.

Recommended branch prefixes are:

- `model/` or `classical/` for Andrei's formulation and solver work;
- `quantum/` for Ilia's QUBO, simulation, and QPU work;
- `engineering/` or `copilot/` for Kirankumar's repository and prototype work;
- `research/` for an agreed integration experiment such as the benchmark
  notebook.

Before starting a new branch:

```bash
git switch main
git pull --ff-only origin main
git switch -c model/short-description
python -m pip install -e ".[full]"
```

The prefix changes with the workstream. A shared integration branch must name
one owner and one reviewer before edits begin.

## Pull-request and review workflow

1. The author states the mathematical or engineering contract being changed.
2. The author stages only related source, tests, documentation, and selected
   evidence artifacts.
3. The primary reviewer checks the workstream-specific behavior.
4. A second reviewer is required when a change affects both the canonical
   objective/constraints and the quantum encoding, validator, or co-pilot.
5. The author resolves review comments and reruns the relevant checks.
6. A reviewer merges only after the branch is current with `main` and checks
   pass.

Minimum checks before a merge are:

```bash
python -m pytest -q
python scripts/install_environment.py --verify-only
git diff --check
git status --short
```

For model or QUBO changes, also run the tiny exact-certification case. For QPU
changes, perform an offline manifest/resume test before any paid submission. A
failed optional Gurobi, Qiskit, or IBM dependency must be reported as skipped;
it must not break the portable classical path.

## Experiment and evidence rules

- Record seed, commit, branch, package versions, solver options, hardware, and
  output path for every result used in the submission.
- Use only synthetic, anonymized, or challenge-approved data. Never commit IBM
  credentials or restricted Vanguard information.
- Preserve raw solver/QPU outputs and report postselection, reallocation, and
  other post-processing explicitly.
- Every reported portfolio must pass the independent validator with zero hard
  breaches. Do not repair a failed result by clipping or renormalizing it after
  the solve.
- Report time-limited MIQP incumbents with status, best bound, and MIP gap.
- Report QPU width separately from global asset-universe size and distinguish
  QPU usage from service wall time.
- Record unsuccessful experiments and limitations; do not select only favorable
  seeds, calibrations, or plots.

## Final contribution and tooling disclosure

The final submission must include this named contribution record and links to
the corresponding commits, notebook sections, or experiment manifests. Each
member reviews the description of their own contribution before submission.

AI-assisted coding or writing must be disclosed. AI output is not experimental
evidence: the team is responsible for reviewing the equations, code, tests,
citations, numerical results, and claims, and every member must be able to
explain the part attributed to them.
