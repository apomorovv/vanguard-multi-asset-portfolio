# Results directory guide

## Final hybrid result folders

`scripts/run_hybrid.py` writes each final run to a named subdirectory such as
`results/tiny_hybrid/`, `results/final_hybrid/`, or `results/large_hybrid/`.
Keep the complete folder together. Its standard evidence package is:

- `hybrid_summary.csv`: one comparable row per relaxation/classical/quantum/exact method;
- `allocation_weights.csv`: exact weights, trades, groups, and support decisions;
- `constraint_checks.csv`: every independently recomputed hard guardrail;
- `change_windows.csv`: which weak holdings and promising replacements were compared;
- `objective_timeline.csv`: best-valid objective versus end-to-end time;
- `backtest_summary.csv`: synthetic out-of-sample return, volatility, CVaR, drawdown, and wealth;
- `hybrid_diagnostics.json`: solver, oracle, quantum, circuit, and skipped-component details;
- `hybrid_report.md`: nontechnical interpretation;
- `artifact_manifest.json`: SHA-256 checksums and sizes;
- `plots/*.png` and matched vector `plots/*.pdf` graphics.

The required acceptance condition remains `feasible = True` and
`breaches = 0`. XY-QAOA results in these folders are supports that have already
passed through the exact continuous allocation oracle and validator—not raw
bitstrings presented as portfolios.

The sections below document the retained classical equal-lot benchmark output.

This directory contains the evidence produced by the classical portfolio benchmark: raw solver runs, summary tables, validation information, and plots. This guide assumes no previous knowledge of portfolio optimization.

## The short version

The program chooses how much of a portfolio to place in each asset. It rewards expected return and income, and penalizes risk, trading cost, and violations of investment rules. Several solvers receive exactly the same data, objective, and constraints. Their answers are independently checked and then compared.

There are two versions of the decision:

- **Continuous:** an asset may receive almost any permitted percentage, such as 17.36%. This is a convex quadratic program and is the scalable classical baseline.
- **Discrete:** the portfolio is split into a fixed number of equal lots. With 20 lots, every weight changes in steps of 5%. This is an integer quadratic problem and is the direct classical reference for later QUBO or quantum comparisons.

Lower objective values are better, but an answer is acceptable only when `feasible` is `True` and `breaches` is zero.

## Do these files need to be kept?

`results/README.md` should always be kept. It documents the directory.

All other files in this directory are reproducible outputs and may be deleted during development. Running the benchmark again recreates them and normally overwrites files with the same names.

For a final submission, paper, presentation, or published comparison, keep the following files together:

- the exact YAML configuration used;
- `benchmark_metadata.json`;
- `benchmark_runs.csv`;
- `benchmark_summary.csv`;
- `allocation_weights.csv`;
- `constraint_checks.csv`;
- `solver_diagnostics.json`;
- `problem.json`;
- `resolved_config.yaml`;
- `artifact_manifest.json`;
- `classical_baseline_report.md`;
- `test_report.txt`;
- every plot cited in the submission;
- the complete `results/scaling/` snapshot, if scaling claims are made.

Do not keep a plot without its tables and metadata: the picture alone does not record the objective weights, lot resolution, skipped solvers, or random seeds. Because runs overwrite this directory, copy an important final snapshot to a uniquely named directory before starting another experiment. An older file such as `allocation_balanced.png` can be removed if nothing cites it; the current standard allocation figure is `allocation_comparison.png`.

## What problem is being solved?

Imagine that one dollar represents the whole portfolio. A **portfolio weight** is the fraction assigned to an asset:

- `0.40` means 40%;
- `0.07` means 7%;
- all weights normally add to `1.0`, or 100%.

For each asset, the input data include:

| Symbol | Code field | Plain-language meaning |
|---|---|---|
| \(w_i\) | solver output | New portfolio weight chosen for asset \(i\) |
| \(w_i^0\) | `w0` | Current weight before rebalancing |
| \(\mu_i\) | `mu` | Estimated annual return |
| \(y_i\) | `y` | Estimated annual income yield |
| \(c_i\) | `c` | Trading-cost coefficient |
| \(\ell_i,u_i\) | `lower`, `upper` | Minimum and maximum allowed weights |
| \(\Sigma\) | `cov` | Covariance matrix describing individual risk and co-movement |

Rates use decimals. For example, an expected return of `0.07` means 7% per year, not 0.07%.

### Objective function

Every solver minimizes the same score:

\[
F(w)=
\lambda_{risk}w^T\Sigma w
-\lambda_{return}\mu^T w
-\lambda_{income}y^T w
+\lambda_{cost}\sum_i c_i|w_i-w_i^0|.
\]

The four parts mean:

1. **Risk:** \(w^T\Sigma w\) is portfolio variance. It increases the score, so the optimizer tries to reduce it.
2. **Expected return:** \(\mu^T w\) is subtracted. A higher expected return therefore improves, or lowers, the score.
3. **Income:** \(y^T w\) is also subtracted when income is valued.
4. **Transaction cost:** moving away from the current portfolio increases the score.

The objective can be negative. That is normal: return and income are reward terms with minus signs. It is a ranking score, not a dollar profit or a percentage return.

### Hard constraints

The optimizer must obey all enabled rules:

\[
\sum_i w_i=1,
\qquad
\ell_i\leq w_i\leq u_i,
\qquad
L_g\leq\sum_{i\in g}w_i\leq U_g.
\]

In plain language:

- exactly 100% of the budget is invested;
- every asset stays inside its minimum and maximum allocation;
- every asset group stays inside its group limits;
- holdings are long-only in this baseline, so weights cannot be negative;
- an optional minimum expected return may be imposed;
- an optional maximum turnover may be imposed.

Preferences in the objective are **soft trade-offs**. Constraints are **hard rules**. Raising the risk preference encourages lower risk, while a risk constraint would prohibit any answer beyond a fixed limit. The code never accepts an objective improvement as compensation for breaking a hard constraint.

### Continuous representation

For \(n\) assets, the continuous model uses:

- \(n\) real-valued allocation variables \(w_i\);
- \(n\) nonnegative auxiliary variables \(t_i\) representing \(|w_i-w_i^0|\).

Thus the solver-facing quadratic program has \(2n\) continuous variables. The auxiliary variables are bookkeeping for turnover and transaction cost; they are not additional assets.

With a positive-semidefinite covariance matrix and linear constraints, this is a **convex quadratic program**. It can be solved to global optimality in polynomial time to a requested numerical accuracy using standard convex-optimization methods. In practice, runtime also depends on matrix size, sparsity, conditioning, solver tolerances, and hardware.

### Discrete representation

The discrete model divides the budget into `units = M` equal lots. It uses integer variables \(q_i\):

\[
w_i=\frac{q_i}{M},
\qquad
q_i\in\{0,1,\ldots,M\},
\qquad
\sum_i q_i=M
\]

when the budget is 1. With `M = 20`, one lot is 5%; with `M = 100`, one lot is 1%.

This is a mixed-integer quadratic program. Integer choices make the general problem NP-hard. Before bounds and group rules prune candidates, the number of ways to distribute \(M\) identical lots across \(n\) assets is

\[
\binom{M+n-1}{n-1}.
\]

Examples:

| Assets \(n\) | Lots \(M\) | Unconstrained allocations |
|---:|---:|---:|
| 6 | 10 | 3,003 |
| 6 | 20 | 53,130 |
| 10 | 20 | 10,015,005 |
| 50 | 100 | approximately \(6.7\times10^{39}\) |

This explains why exact enumeration is excellent for small correctness tests but unsuitable for large experiments. Increasing both assets and lots can make the search space explode even though each individual objective calculation is quick.

## Is this the full model?

It is the complete **classical single-period baseline implemented in this repository**. All classical backends use the same canonical objective, asset bounds, group bounds, budget, optional target-return rule, optional turnover cap, validation, and metrics.

It is not every possible real-world portfolio feature. In particular, the current model is not:

- a historical backtest or a guarantee of future performance;
- a multi-period trading model;
- a tax, liquidity, market-impact, or scenario-based risk engine;
- a complete end-to-end quantum solver;
- financial advice.

The discrete model is especially important because it provides an exact or certifiable classical target against which a QUBO or quantum result can later be judged. A credible quantum experiment should use the same data and constraints and report its gap to this classical reference.

## Which solver does what?

### Continuous solvers

| Configuration name | Method | Role |
|---|---|---|
| `scipy` | SciPy SLSQP with analytic gradients | Guaranteed, easy-to-install baseline |
| `osqp` | Specialized open-source convex QP solver | Recommended open-source production baseline for large sparse QPs |
| `cvxpy:CLARABEL` | CVXPY model with Clarabel | Independent open-source formulation and cross-check |
| `gurobi` | Direct Gurobi convex QP | Strong commercial comparison, subject to a valid license |

For this model, OSQP is usually the best open-source continuous benchmark, while Gurobi is the strongest commercial comparison. SciPy remains valuable because it is always available and provides an independent implementation. Agreement among independent solvers is more convincing than relying on one solver alone.

### Discrete solvers

| Configuration name | Method | Exact? | Best use |
|---|---|---:|---|
| `enumeration` | Checks every feasible lot allocation | Yes | Tiny instances and ground-truth tests |
| `local_search` | Repeatedly moves one lot between two assets | No | Fast deterministic baseline |
| `annealing` | Seeded simulated annealing followed by swap polishing | No | Broader heuristic search |
| `gurobi` | Direct MIQP with integer lot variables | Yes if status is optimal | Best general exact/certifiable discrete benchmark |
| `cvxpy:SCIP` | MIQP expressed through CVXPY and solved by SCIP | Yes if status is optimal | Open-source exact/certifiable comparison when SCIP is installed |

For a tiny problem, enumeration is the clearest truth source. For a larger discrete problem, Gurobi is generally the strongest exact solver here; SCIP is the main open-source alternative. Local search and annealing are useful scalable heuristics, but they cannot prove global optimality even when they happen to find the same answer as an exact solver.

## How the code flows

The benchmark follows this sequence:

1. `scripts/run_classical.py` reads a YAML configuration.
2. `data_generation.py` creates or loads a `PortfolioProblem`.
3. `schemas.py` checks dimensions, bounds, covariance consistency, and basic validity.
4. `portfolio_model.py` defines the one canonical objective and converts it to continuous-QP or discrete-lot form.
5. `classical_continuous.py` and `classical_discrete.py` call the selected solver backends.
6. `_result.py`, `validation.py`, and `metrics.py` independently recompute feasibility and financial metrics.
7. `classical.py` constructs fair comparisons and writes tables and metadata.
8. `plotting.py` creates the graphics in this directory.

Important source files:

| File | Responsibility |
|---|---|
| `src/vanguard_portfolio/schemas.py` | The only definitions of the problem, preferences, and normalized solver result |
| `src/vanguard_portfolio/data_generation.py` | Reproducible synthetic and factor-based universes |
| `src/vanguard_portfolio/portfolio_model.py` | Objective, QP matrices, lot conversion, and direct calculations |
| `src/vanguard_portfolio/classical_continuous.py` | SciPy, OSQP, CVXPY, and Gurobi continuous solvers |
| `src/vanguard_portfolio/classical_discrete.py` | Enumeration, local search, annealing, Gurobi, and CVXPY/SCIP discrete solvers |
| `src/vanguard_portfolio/validation.py` | Solver-independent checks of every hard constraint |
| `src/vanguard_portfolio/metrics.py` | Return, risk, turnover, cost, concentration, and objective-gap metrics |
| `src/vanguard_portfolio/classical.py` | Solver orchestration, presets, aggregation, and report writing |
| `src/vanguard_portfolio/plotting.py` | Plot generation |
| `scripts/run_classical.py` | Complete config-driven benchmark entry point |
| `scripts/run_experiment.py` | Continuous scaling experiment across increasing asset counts |

## Installation

Run commands from the repository root, the directory containing `pyproject.toml`.

Python 3.10 or newer is required. Install the complete configured solver stack and test tools with:

```bash
python -m pip install -e ".[all-solvers,test]"
```

Gurobi still requires a working license. Verify it with:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version()); print(gp.Model().Status)"
```

The formal baseline and large configurations set `missing_optional: error`; a requested solver cannot silently disappear. The tiny portable configuration may still record an unavailable backend as skipped.

## How to run the benchmark

### 1. Verify the code

```bash
python -m pytest -q
```

To save the test log in this directory:

```bash
python -m pytest -q > results/test_report.txt 2>&1
```

### 2. Run a quick learning example

```bash
python scripts/run_classical.py --config configs/tiny_example.yaml
```

This uses 10 lots, one annealing seed, and only the guaranteed solvers. It is the right first run because exact enumeration is small enough to check the heuristics.

### 3. Run the full configured baseline

```bash
python scripts/run_classical.py --config configs/baseline.yaml
```

This requests all configured continuous and discrete solvers, uses 20 lots, runs annealing with five seeds, writes complete audit files, and creates plots. Its configuration already treats missing backends as errors.

To make missing optional solvers stop the run instead of being skipped:

```bash
python scripts/run_classical.py \
  --config configs/baseline.yaml \
  --strict-optional
```

To write a separate experiment without overwriting this directory:

```bash
python scripts/run_classical.py \
  --config configs/baseline.yaml \
  --output results/experiment_01
```

The console table is a quick summary. The files are the durable record.

### 4. Run the large 250-asset example

```bash
python scripts/run_classical.py \
  --config configs/large_example.yaml \
  --strict-optional
```

This uses a factor-generated universe, 1,000 lots, scalable candidate-pool heuristics, repeated continuous timings, ten annealing seeds, and time-limited Gurobi/SCIP MIQP runs. It intentionally omits enumeration and writes to `results/large_example/`.

## Configuration parameters

The main controls are in `configs/baseline.yaml`.

### Problem source

```yaml
problem:
  source: synthetic
```

- `synthetic` creates the fixed reproducible demonstration universe.
- A relative or absolute JSON path loads a saved `PortfolioProblem` instead.

Changing this field changes the investment data and constraints. It is the data source, not a solver setting.

### Preference weights

```yaml
preferences:
  preset: balanced
  lambda_return: 1.0
  lambda_risk: 5.0
  lambda_income: 0.0
  lambda_cost: 1.0
```

| Parameter | Increasing it usually does this |
|---|---|
| `lambda_return` | Favors assets with higher estimated return, often accepting more risk |
| `lambda_risk` | Favors lower variance and often a more diversified allocation |
| `lambda_income` | Favors assets with higher income yield |
| `lambda_cost` | Keeps the new portfolio closer to the current portfolio |

Available presets are `balanced`, `growth`, `income`, `drawdown_control`, and `cost_sensitive`. Explicit lambda values override the selected preset.

There is no universally correct lambda value. The terms have different numeric scales, so a value of 5 for risk is not intrinsically five times more important than a value of 1 for return. Compare several values, inspect the objective components and financial metrics, and document the chosen preference. The `drawdown_control` preset raises variance aversion; it is not a true maximum-drawdown model.

Do not compare raw objective values from different lambda settings as though they were solver gaps. Changing lambdas changes the question being asked.

### Discrete controls

```yaml
discrete:
  units: 20
  seeds: [0, 1, 2, 3, 4]
  annealing_iterations: 20000
```

| Parameter | Meaning | Cost of increasing it |
|---|---|---|
| `units` | Number of equal budget lots | Finer weights but a much larger integer search space |
| `seeds` | Independent random annealing runs | More reliable statistics and proportionally more runtime |
| `annealing_iterations` | Trial moves per annealing run | More search effort and roughly proportional runtime |

`units` does **not** change the number of assets. It changes weight precision. Asset count comes from the problem data.

### Solver lists

```yaml
solvers:
  continuous: [scipy, osqp, cvxpy:CLARABEL, gurobi]
  discrete: [enumeration, local_search, annealing, gurobi, cvxpy:SCIP]
  missing_optional: skip
```

Only listed solvers are attempted. Use `skip` for portable runs. Use `error`, or the command-line `--strict-optional` flag, when a formal benchmark must prove that every requested backend actually ran.

### Output controls

```yaml
outputs:
  directory: results
  make_plots: true
  make_risk_aversion_sweep: true
```

- `directory` selects the destination.
- `make_plots` enables all standard plots.
- `make_risk_aversion_sweep` solves the continuous SciPy model at risk weights `0.5, 1, 2, 5, 10, 20, 50` and creates `risk_aversion_sweep.png`.

## How to increase the problem size correctly

There are four separate ways to make an experiment harder. Use them deliberately.

### A. Increase weight precision

Change `discrete.units`, for example:

```yaml
discrete:
  units: 40
  seeds: [0, 1, 2, 3, 4]
  annealing_iterations: 50000
```

At 40 lots, the step size is 2.5%. At 100 lots, it is 1%. Finer resolution normally lets the discrete answer approach the continuous answer, but exact enumeration becomes much slower.

### B. Increase the number of assets

The supplied scaling runner generates factor-based universes of different sizes and tests continuous solvers:

```bash
python scripts/run_experiment.py \
  --sizes 10 25 50 100 250 \
  --backends scipy osqp cvxpy:CLARABEL gurobi \
  --instance-seeds 0 1 2 \
  --repetitions 3 \
  --strict-optional
```

This is the safest way to study continuous scaling. It does not run the discrete solvers.

For a larger discrete experiment, use the package directly in a script or notebook:

```python
from vanguard_portfolio.classical import benchmark_solvers
from vanguard_portfolio.data_generation import generate_factor_universe
from vanguard_portfolio.schemas import Preferences

problem = generate_factor_universe(
    n_assets=50,
    n_groups=5,
    n_factors=4,
    seed=123,
)

report = benchmark_solvers(
    problem,
    Preferences(lambda_return=1.0, lambda_risk=5.0),
    units=100,
    continuous_backends=[
        {"name": "osqp", "repetitions": 3},
        {"name": "gurobi", "repetitions": 3},
    ],
    discrete_backends=[
        {"name": "local_search", "options": {"candidate_pool_size": 64}},
        {"name": "annealing", "options": {"polish_candidate_pool_size": 64}},
        {"name": "gurobi", "options": {"time_limit": 600, "mip_gap": 1e-3}},
    ],
    seeds=range(10),
    annealing_iterations=100_000,
)
```

Do not include `enumeration` for 50 assets and 100 lots. The enumeration safety guard will reject an excessive asset-bound-feasible search space before recursion begins.

### C. Tighten the hard constraints

Tighter limits test feasibility handling and solver robustness. For example:

```python
from dataclasses import replace

harder_problem = replace(
    problem,
    target_return=0.06,
    max_turnover=0.30,
)
```

This requires at least 6% expected return and limits aggregate absolute reallocation to 30%. These numbers are examples, not universally feasible choices. Confirm feasibility with a continuous solver before attempting a discrete grid. A continuous model can be feasible while a coarse lot grid is infeasible.

Asset and group bounds can also be tightened, but their lower limits, upper limits, and budget must remain mutually compatible. Do not make a test “hard” by supplying a non-symmetric or non-positive-semidefinite covariance matrix; that creates an invalid model rather than a meaningful benchmark.

### D. Increase stochastic search effort

Use at least 10 seeds and more annealing iterations for serious heuristic reporting. Report the distribution, not only the best run. A good result table includes the median, interquartile range, best objective, feasible rate, and exact-reference gap.

## Recommended hard-test ladder

Scale one dimension at a time so that a failure has a clear cause.

| Stage | Suggested experiment | Purpose |
|---|---|---|
| 1. Smoke | 6 assets, 10 lots, 1 seed | Verify installation and file generation |
| 2. Exact correctness | Small assets, 10–20 lots, enumeration plus MIQP | Confirm exact solvers and objective calculations agree |
| 3. Resolution | Same data with 5, 10, 20, 40, 100 lots | Measure convergence toward the continuous relaxation |
| 4. Continuous scale | 10, 25, 50, 100, 250+ assets | Measure solver runtime and feasibility as \(n\) grows |
| 5. Heuristic robustness | 10+ seeds and 50,000–100,000 iterations | Measure stochastic reliability |
| 6. Constraint stress | Add return and turnover limits gradually | Test feasibility and boundary behavior |
| 7. Deliberate infeasibility | Construct one known-impossible case | Confirm the program reports infeasibility instead of returning a bad portfolio |

For fair timing:

- use the same computer and software environment;
- keep the same problem and preferences across compared solvers;
- run multiple repetitions after a warm-up;
- report median and interquartile range;
- include model-construction and post-processing time consistently;
- record skipped packages and license failures;
- never compare a heuristic’s best-of-many time with an exact solver’s single-run time without disclosing the difference.

## Generated files

### `benchmark_runs.csv`

The main flat machine-readable run table. There is one row per solver run. A stochastic solver therefore has one row per random seed, and a repeated deterministic solver has one row per repetition. Exact weights and nested native diagnostics are stored in the dedicated files below.

Important columns:

| Column | Meaning |
|---|---|
| `run_id` | Unique key joining runs, allocations, constraints, and diagnostics |
| `method` | Normalized solver name |
| `model_type` | `continuous` or `discrete` |
| `repetition` | Repeated timing index |
| `objective` | Canonical score; lower is better under the same model and preferences |
| `runtime_seconds` | Wall-clock time for model construction and solve |
| `status` | Solver’s textual completion status |
| `success` | The backend returned a usable answer |
| `optimal` | The backend reported or proved global optimality |
| `feasible` | Independent validation found no unacceptable hard-constraint violation |
| `breaches` | Number of violated hard constraints; must be zero |
| `max_violation` | Largest violation magnitude; should be zero within numerical tolerance |
| `units` | Lot count for a discrete run; blank for continuous runs |
| `seed` | Random seed for a stochastic run; blank otherwise |
| `risk_term` | Signed risk contribution to the objective |
| `return_term` | Signed expected-return contribution to the objective |
| `income_term` | Signed income contribution to the objective |
| `cost_term` | Signed transaction-cost contribution to the objective |
| `expected_return` | Weighted annual expected return |
| `variance` | Portfolio variance \(w^T\Sigma w\) |
| `volatility` | Square root of variance, in annual decimal units |
| `income` | Weighted annual income yield |
| `turnover` | \(\sum_i|w_i-w_i^0|\), the total absolute reallocation |
| `transaction_cost` | Cost coefficient dot absolute trades |
| `concentration_hhi` | Sum of squared weights; larger means more concentrated |
| `effective_holdings` | Reciprocal of concentration; an intuitive diversification count |
| `absolute_objective_gap` | Objective minus the best exact/optimal reference in the same model class |
| `relative_objective_gap` | Absolute gap divided by a scale-safe reference magnitude |
| `absolute_gap_to_certified_bound` | Objective minus the tightest available global lower bound |
| `relative_gap_to_certified_bound` | Certified-bound gap divided by a scale-safe bound magnitude |

`success` and `feasible` are different. A solver can finish successfully but still return a numerically invalid allocation. `optimal` is also different: heuristics remain marked non-optimal even if they happen to match the exact objective.

### `benchmark_summary.csv`

One aggregated row per model type and method. It reports:

- number of runs;
- success, feasibility, and optimal-status rates;
- best and median objectives;
- best gap to the relevant reference;
- gap to the tightest certified lower bound when a time-limited MIQP has no proven optimum;
- median runtime and its first and third quartiles;
- median return, volatility, and turnover.

Use this file for comparison tables. Use `benchmark_runs.csv` when investigating variability or an unexpected aggregate.

### `allocation_weights.csv`

One row per `(run, asset)`. It preserves exact optimized weights, current weights, changes, group membership, asset bounds, and integer lots for discrete runs. Use `run_id` to join it to every other result table.

### `constraint_checks.csv`

One row per independently recomputed hard constraint. It records the left-hand side, sense, right-hand side, signed slack, and violation. This is the detailed evidence behind `feasible`, `breaches`, and `max_violation`.

### `solver_diagnostics.json`

Preserves backend-specific information that does not fit naturally into one CSV schema, including iterations, residuals, feasible-start time, model-build time, solve time, integer lots, objective evaluations, accepted moves, Gurobi nodes, best bounds, and reported MIP gaps.

### `benchmark_metadata.json`

The minimum context needed to interpret the numbers:

- problem label;
- discrete lot count;
- four objective preference weights;
- continuous reference objective;
- discrete reference objective;
- optional solvers that were skipped and the reason for each skip.
- requested solver specifications and repetitions;
- asset/group counts and problem fingerprint;
- Python, operating-system, and package versions.

This file is why a missing Gurobi license cannot silently look like a lost comparison. It also preserves separate optimal-reference and certified-lower-bound values.

### `problem.json`

The exact serialized `PortfolioProblem` used in the run. It includes all returns, volatilities, correlations, covariance values, costs, current weights, bounds, and optional hard guardrails.

### `resolved_config.yaml`

The exact YAML configuration accepted by the runner. Keep it with `problem.json`: the problem records the data, while the configuration records preferences, solvers, limits, repetitions, seeds, and output controls.

### `artifact_manifest.json`

Lists each generated table/report/plot with its byte size and SHA-256 checksum. A changed checksum proves that a file no longer matches the recorded snapshot.

### `classical_baseline_report.md`

A short human-readable benchmark table derived from the CSV files. It is convenient for review, but the CSV and JSON files remain the authoritative detailed outputs.

### `test_report.txt`

The captured unit-test log. It shows whether model construction, solvers, metrics, validation, and expected optional-solver behavior passed at the time of the snapshot. It is produced by redirecting the test command; `run_classical.py` does not create it automatically.

### `results/scaling/scaling_benchmark.csv`

Created only by `scripts/run_experiment.py`. Each row identifies asset count, factor/group count, instance seed, repetition, backend, feasibility, objective, timing phases, and financial metrics. The same directory also contains `scaling_benchmark_summary.csv`, `scaling_benchmark_runtime.png`, and a checksum manifest. Use these files to study runtime and feasibility growth; do not compare objectives across different problem sizes.

## Generated plots

### `allocation_comparison.png`

Shows the current portfolio and the best feasible allocation from each method. The horizontal axis lists assets; bar height is portfolio weight. Similar bars mean methods found similar allocations. A method is not necessarily missing when bars overlap exactly. Above 30 assets, the plot shows the 30 largest displayed allocations and aggregates the remainder as `Other assets`.

Questions this plot answers:

- Which assets gained or lost weight?
- Do continuous and discrete solutions look similar?
- Did multiple solvers find the same allocation?

### `risk_return.png`

Places every successful feasible run on a risk-return chart:

- horizontal axis: annualized volatility, farther right means more risk;
- vertical axis: annualized expected return, higher means more expected return;
- circles identify continuous results;
- squares identify discrete results.

Several labels may occupy the same point because different solvers found the same portfolio. This chart is based on model estimates, not realized future returns.

### `runtime_comparison.png`

Compares median wall-clock runtime. The vertical axis is logarithmic, so equal visual distances can represent large multiplicative differences. Error bars show the interquartile range when repeated runs exist. Runtime includes Python model construction and solver execution and is specific to the machine and installed software.

### `objective_gap.png`

Shows

\[
\text{best method objective}-\text{reference objective}
\]

within each model class. When a proven optimum exists, it is the reference. Otherwise, the plot uses the tightest available certified lower bound from a bounded solver such as Gurobi. Zero means the method matched the certificate; a positive value is worse. Continuous methods are compared only within the continuous model and discrete methods only within the discrete model. Do not interpret the gap between these two model classes as a solver error—the discrete grid is a different feasible set.

### `correlation_heatmap.png`

Shows how asset returns are assumed to move together:

- `+1`: perfectly aligned movement;
- `0`: no linear relationship in the model;
- `-1`: perfectly opposite movement;
- diagonal cells are always `1` because an asset is perfectly correlated with itself.

Correlation is converted to covariance using each asset’s volatility. The covariance matrix, not correlation alone, drives portfolio variance.

For more than 60 assets, individual tick labels are intentionally hidden so the heatmap remains readable.

### `constraint_slacks.png`

Shows the selected solution’s distance from each hard-constraint boundary:

- positive bar: safely inside the permitted region;
- zero or nearly zero: the constraint is binding;
- negative bar: a violation.

Budget is an equality and is omitted from this plot because its signed slack is always zero. The CSV fields `breaches` and `max_violation` remain the definitive validation summary.

For large problems, only the 50 violated or most binding constraints are displayed. The complete set remains in `constraint_checks.csv`.

### `risk_aversion_sweep.png`

Shows continuous solutions obtained while only `lambda_risk` changes. Moving toward higher risk aversion usually lowers volatility and may lower expected return. The labels show the risk preference used. This is a preference-sensitivity plot, not a historical efficient frontier or a backtest.

## How to read a result safely

Use this order:

1. Check `benchmark_metadata.json` to learn the preferences, lot resolution, and skipped solvers.
2. Require `success = True`, `feasible = True`, and `breaches = 0`.
3. Compare objectives only when the problem data, preferences, and model type are the same.
4. Check the objective gap. Zero is best; a small positive gap is near the reference.
5. Compare return, volatility, turnover, transaction cost, and concentration to understand why objectives differ.
6. Inspect allocation and constraint-slack plots.
7. Compare runtime only after checking run counts, seeds, and machine conditions.

Useful interpretation details:

- `0.05` expected return or volatility means 5%.
- In a long-only fully invested portfolio, turnover can be as large as 2. A turnover of `0.50` means 50% aggregate absolute reallocation; it is not necessarily 50% of assets sold because purchases and sales both contribute.
- A lower continuous objective than discrete objective is expected: the continuous solver can choose weights between grid points.
- A skipped optional solver is not a failed optimization.
- An `optimal = False` heuristic may still have a zero measured gap when an independent exact reference exists. It found the same score but did not prove it by itself.
- Objective values from different presets or data sets are not directly comparable.

## Expected cross-checks

A healthy small benchmark should satisfy all of these:

- every reported solution has zero constraint breaches;
- independent continuous solvers agree within numerical tolerance;
- exact enumeration and an optimal Gurobi or SCIP MIQP agree at the same lot resolution;
- the continuous optimum is no worse than the discrete optimum for this minimization problem;
- changing only an annealing seed may change runtime or the heuristic result, but never the hard constraints;
- increasing lots usually reduces discretization loss, although it need not improve every heuristic run;
- unavailable optional solvers appear in `benchmark_metadata.json` rather than disappearing silently.

## Common problems

### “A solver was skipped”

Install the corresponding optional package and, for Gurobi, confirm that a valid license is available. Run with `--strict-optional` when skipped comparisons should be treated as errors.

### Enumeration runs for too long

Reduce the number of assets or `units`, or remove `enumeration` from the discrete solver list. Use Gurobi/SCIP for a certifiable larger problem or local search/annealing for a heuristic result.

### “No feasible allocation exists”

The budget, asset bounds, group bounds, return target, turnover cap, and lot grid may conflict. First test the continuous model. Then check whether the requested bounds are multiples of the lot size or allow any nearby grid point.

### `ModuleNotFoundError`

Run the installation command from the repository root:

```bash
python -m pip install -e ".[qp,test]"
```

Then run scripts from that same root.

### The objective is negative

This is normal. Return and income are subtracted because the program minimizes the score.

### Results changed after another run

The runner overwrites standard filenames. Different configs, packages, random seeds, machines, or solver versions can also change outputs. Use a separate `--output` directory for every result snapshot that must remain reproducible.

## Reproducibility checklist for a final result

Before presenting a comparison, record:

- repository commit or source snapshot;
- Python and solver versions;
- computer and operating system;
- exact YAML configuration;
- exact input data or generation seed;
- asset count, group count, and discrete lot count;
- all random seeds and annealing iterations;
- solver statuses, licenses, time limits, and optimality tolerances;
- raw and summarized result files;
- unit-test output;
- whether any local polishing or post-processing was used.

The central rule is simple: **a result is trustworthy only when its model, feasibility, reference, and experimental conditions can all be reconstructed.**
