# Evidence-backed presentation claims

- Repository commit: `571d450be694789be4d57bfa5887fec0b9f594ed` on branch `research/benchmark-notebook-cleanup`.
- 4 independent continuous-QP backends agreed within an objective spread of 1.406e-09.
- Tiny exact-certification case completed with 0 hard-constraint breaches.
- Tiny enumeration and Gurobi objectives differ by 0.000e+00.
- The 100-asset canonical hybrid case returned 20 holdings with 0 hard-constraint breaches.
- Gurobi reported a MIP gap of 0.0000% for the 100-asset reference.
- In the controlled Gurobi start study, `hybrid_incumbent_start` was fastest at 0.027 s; cold and warm starts are reported separately.
- Out-of-sample behavior was evaluated across 30 independent synthetic paths, with medians and 10th–90th percentile ranges reported.
- One constraint-gauntlet case simultaneously enforced exact cardinality, minimum/maximum positions, eligibility, mandatory assets, group limits, turnover, target return, minimum income, factor bands, stress floors, and empirical CVaR.
- The all-constraints portfolio had 0 independently recomputed breaches.
- The scenario-based CVaR preference sweep evaluated 30 independently validated allocations on a fixed exact-K support; every penalty level had a zero-breach rate of 100%.
- Relative to lambda_scenario=0, the lowest median held-out CVaR occurred at lambda_scenario=1: tail loss fell by 54.31 bp with an expected-return sacrifice of 41.55 bp.
- On the frozen 16-asset window, the best tested method was `random_fixed_weight` with validated improvement 0.00229682 over the warm start.
- Aer execution metadata verified GPU sampling for the frozen-window circuit.
- The largest successful measured factor-native universe contained 300,000 assets.
- At 300,000 assets, median time to the first valid portfolio was 34.693 s and median complete hybrid search time was 112.600 s.
- Across recorded scaling trials, 27/27 successful runs had zero breaches.
- IBM Runtime evidence contains 1 successful job IDs across 10 width/depth cases in the primary frontier campaign.
- All 30 primary-frontier hardware observations completed allocation and validation with 0 total hard-constraint breaches.
- Primary-frontier fixed-cardinality survival ranged from 29.14% at the median to 67.44% at the best observation.
- Exact QUBO-to-final-objective alignment was weak over the tractable windows (Spearman rho 0.24 to 0.28), so QUBO quality and final allocated portfolio quality are reported separately.

## Required interpretation limits

- Global asset count is not QPU width; the QPU receives only the adaptive window.
- A continuous-relaxation gap is not a certified mixed-integer optimality gap.
- Cardinality preservation and mitigation gains characterize hardware, not quantum advantage.
- IBM-reported QPU usage and created-to-finished job wall time are distinct, non-additive platform metrics.
- Final feasibility comes from the classical allocation oracle and independent validator, not from raw QPU samples.
- A mitigation gain over the unmitigated QPU is not a win over random screening or classical LNS.
- Synthetic backtests are demonstrations, not investment forecasts.
