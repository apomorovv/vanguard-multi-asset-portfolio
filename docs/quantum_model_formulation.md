# Quantum Model Formulation

## 1. Purpose

This document defines the quantum optimization component used by the final
hybrid portfolio pipeline.

The implemented method is fixed-cardinality XY-QAOA over a small adaptive
change window. It does not optimize the full asset universe and does not assign
final portfolio percentages.

The division of responsibility is:

- the classical model builds and validates the financial problem;
- the quantum circuit proposes combinations of assets inside one change window;
- the classical allocation oracle assigns exact continuous weights;
- the independent validator checks every configured hard constraint;
- the optional Gurobi model provides the exact-cardinality classical reference.

A quantum output is therefore a candidate support, not a complete portfolio.

## 2. Support and Allocation Decisions

The final sparse portfolio contains:

$$
z_i\in\{0,1\},
$$

where $z_i=1$ means asset $i$ is selected, and

$$
w_i\ge0,
$$

where $w_i$ is its exact portfolio weight.

The full sparse model links them through

$$
m_i z_i\le w_i\le u_i z_i,
$$

and enforces exact cardinality:

$$
\sum_{i=1}^{n}z_i=K.
$$

A direct quantum encoding of all continuous weights would require additional
discretization and many more binary variables. The final architecture therefore
uses quantum variables only for local support changes. Exact weights remain
continuous and are solved classically.

## 3. Adaptive Change Window

Suppose the current valid portfolio contains exactly $K$ assets.

At hybrid iteration $j$, the classical algorithm constructs a window

$$
W_j\subseteq\{1,\ldots,n\}
$$

containing:

- weak selected assets that may be removed;
- promising unselected assets that may be added.

Selected assets outside the window remain frozen.

Let:

- $F=|W_j|$ be the number of window assets;
- $r$ be the number currently selected inside the window;
- $x_i\in\{0,1\}$ indicate whether window asset $i$ is selected in the proposal.

The quantum search is restricted to

$$
\sum_{i=1}^{F}x_i=r.
$$

The frozen support contains $K-r$ assets. Every valid window proposal therefore
has

$$
(K-r)+r=K
$$

total holdings.

The global asset universe may contain thousands of assets while the quantum
circuit uses only $F$ qubits.

## 4. Window Support Surrogate

The quantum circuit searches a quadratic surrogate rather than the complete
continuous financial model.

Let:

- $w^{\mathrm{current}}$ be the current valid portfolio;
- $w_F$ be the current portfolio with all window entries set to zero;
- $C_W$ be the capital currently assigned to window assets;
- $a=C_W/r$ be the equal proxy weight assigned to each selected window asset;
- $S_W$ map a window bit vector into the full asset universe.

The proxy allocation is

$$
\widetilde w(x)=w_F+aS_Wx.
$$

Because every valid bitstring contains exactly $r$ ones,

$$
\sum_{i\in W}\widetilde w_i(x)=ra=C_W.
$$

The proxy preserves window capital but is not the final allocation.

## 5. Window QUBO Derivation

The canonical objective is

$$
F(w) = \lambda_{\mathrm{risk}}w^\top\Sigma w -
\lambda_{\mathrm{return}}\mu^\top w -
\lambda_{\mathrm{income}}y^\top w +
\lambda_{\mathrm{cost}}c^\top|w-w^0|.
$$

Substitution of $\widetilde w(x)$ gives

$$
F(\widetilde w(x)) = x^\top Qx+h^\top x+\kappa.
$$

This is the window QUBO.

### 5.1 Quadratic Risk Term

Let $\Sigma_{WW}$ be the covariance block for window assets. Then

$$
Q = \lambda_{\mathrm{risk}}a^2\Sigma_{WW}.
$$

### 5.2 Risk Interaction With Frozen Holdings

The covariance between a candidate window asset and frozen holdings contributes

$$
2\lambda_{\mathrm{risk}}a(\Sigma w_F)_W
$$

to the linear vector.

### 5.3 Return and Income

The return and income contributions are

$$
-\lambda_{\mathrm{return}}a\mu_W - \lambda_{\mathrm{income}}ay_W.
$$

### 5.4 Proxy Transaction Cost

For window asset $i$, the proxy cost is:

- $c_i|0-w_i^0|$ when $x_i=0$;
- $c_i|a-w_i^0|$ when $x_i=1$.

The corresponding linear coefficient is

$$
h_i^{\mathrm{cost}} = \lambda_{\mathrm{cost}}c_i \left( |a-w_i^0|-|w_i^0| \right).
$$

### 5.5 Group Pressure

The window constructor may add a linear signal that favors assets from groups
near lower limits and discourages assets from groups near upper limits.

This affects proposal ranking only. Exact group bounds remain enforced by the
allocation oracle.

## 6. QUBO Definition

The implementation stores

$$
E(x) = \kappa+h^\top x+x^\top Qx, \qquad x\in\{0,1\}^{F}.
$$

The QUBO also records:

- window asset indices;
- asset names;
- required number of ones $r$;
- proxy weight $a$;
- window capital $C_W$;
- active interaction count;
- removed interaction count;
- cardinality mode.

A low QUBO energy does not by itself imply a feasible or superior final
portfolio.

## 7. Optional Interaction Sparsification

For hardware experiments, the implementation may:

- remove interactions below a magnitude threshold;
- retain only a configured number of strongest edges.

This replaces $Q$ with a sparse approximation $\widehat Q$.

Sparsification changes proposal quality but not final feasibility because every
support is reallocated and evaluated with the complete risk model.

## 8. QUBO-to-Ising Conversion

Binary and spin variables are related by

$$
x_{i}=\frac{1-Z_{i}}{2}
$$

The cost Hamiltonian has the form

$$
H_{C} = c_{0}I + \sum_{i=1}^{F}h_{i}^{(Z)}Z_{i} + \sum_{i \lt j} J_{ij}Z_{i}Z_{j}
$$


The implementation produces:

- a constant shift;
- one-qubit $Z$ fields;
- two-qubit $ZZ$ couplings.

The constant changes the reported energy but not which bitstring minimizes it.

## 9. Exact-Cardinality XY Mixer

The production mixer is

$$
H_M = \frac12 \sum_{(i,j)\in E} \left( X_iX_j+Y_iY_j \right).
$$

For one edge, the mixer exchanges

$$
|10\rangle
\leftrightarrow
|01\rangle.
$$

It does not create or remove an excitation. Therefore,

$$
[H_M,\widehat N]=0,
$$

where

$$
\widehat N = \sum_{i=1}^{F}\frac{I-Z_i}{2}
$$

is the Hamming-weight operator.

Starting from a state with $r$ ones keeps ideal evolution inside

$$
\mathcal{H}_{r} = \mathrm{span} \left[ |x\rangle : \sum_{i} x_{i} = r \right]
$$

## 10. Mixer Topologies

### 10.1 Ring Mixer

The ring uses

$$
(0,1), (1,2), \ldots, (F-2, F-1), (F-1, 0)
$$

It requires $O(F)$ mixer edges and is the practical default.

### 10.2 Complete Mixer

The complete mixer uses every pair:

$$
E = \{ (i,j) : 0 \le i \lt j \lt F \}
$$

It improves direct exchange connectivity but requires $O(F^2)$ interactions and a deeper physical circuit.

## 11. Initial States

### 11.1 Warm Start

The default state is the current window support:

$$
|\psi_{0}\rangle = |x^{\mathrm{current}}\rangle
$$

It already has the required Hamming weight and is easy to prepare.

### 11.2 Dicke State

The optional Dicke state is

$$
|D_{F}^{r}\rangle = \frac{1}{\sqrt{\binom{F}{r}}} \sum_{\sum_{i} x_{i} = r} |x\rangle
$$

It is an ablation that gives equal initial amplitude to all fixed-weight states but requires more expensive preparation.


## 12. QAOA State and Objective

For depth $p$,

$$
|\psi(\gamma,\beta)\rangle = \prod_{\ell=1}^{p} e^{-i\beta_\ell H_M} e^{-i\gamma_\ell H_C} |\psi_0\rangle.
$$

The variational objective is

$$
\mathcal E(\gamma,\beta) = \langle\psi(\gamma,\beta)| H_C |\psi(\gamma,\beta)\rangle.
$$

Cost coefficients are normalized before angle optimization. Positive
normalization changes scale but not the ordering of QUBO states.

## 13. Fixed-Weight Subspace Simulator

The reference simulator stores only basis states with exactly $r$ ones.

The subspace dimension is

$$
N_{\mathrm{subspace}}=\binom Fr.
$$

| $F$ | $r$ | Subspace states | Full states |
|---:|---:|---:|---:|
| 8 | 4 | 70 | 256 |
| 12 | 6 | 924 | 4,096 |
| 16 | 7 | 11,440 | 65,536 |
| 20 | 10 | 184,756 | 1,048,576 |

The configured `maximum_subspace_states` guard prevents an unsafe exact
subspace simulation.

The cost layer applies

$$
\psi_x\leftarrow e^{-i\gamma E(x)}\psi_x.
$$

The XY layer rotates amplitudes between fixed-weight partner states that differ
by `10` and `01` on one mixer edge.

## 14. Classical Angle Optimization

The implementation uses COBYLA with multiple seeded starts.

There are $2p$ variational parameters. The best expected normalized energy
across starts determines the final angles.

This is a nonconvex heuristic optimization. It does not prove the globally best
QAOA angles were found.

## 15. Sampling Backends

| Backend | Role |
|---|---|
| `subspace` | Exact fixed-weight CPU reference and angle optimization. |
| `aer_cpu` | Physical-circuit simulation on CPU. |
| `aer_gpu` | Physical-circuit simulation on NVIDIA GPU. |
| `ibm_runtime` | Sampling on a selected IBM QPU. |

The optimized logical circuit is reused across the optional physical backends.

## 16. Physical Circuit

For a warm start:

1. apply $X$ gates to currently selected window qubits;
2. apply `RZ` and `RZZ` gates for the cost Hamiltonian;
3. apply `RXX` and `RYY` gates for the XY mixer;
4. repeat for depth $p$;
5. measure all qubits.

The run records setup time, transpilation time, sampling time, decoding time,
circuit depth, operation counts, requested device, and actual device.

## 17. Cardinality Feasibility Rate

For required weight $r$,

$$
\text{cardinality rate} = \frac{ \sum_x \mathbf 1\left[\sum_i x_i=r\right]N_x }{\sum_xN_x},
$$

where $N_x$ is the count of bitstring $x$.

The exact subspace simulator should have rate one. Physical circuits may produce
invalid weights because of gate noise, routing, decoherence, and readout error.

Raw and postselected rates must be reported separately.

## 18. Full-Support Reconstruction

Let $\mathcal F_{\mathrm{frozen}}$ be the frozen support and
$\mathcal H_W(x)$ the selected window assets.

The proposed full support is

$$
\mathcal H(x) = \mathcal F_{\mathrm{frozen}} \cup \mathcal H_W(x).
$$

For a fixed-weight bitstring,

$$
|\mathcal H(x)|=K.
$$

The bitstring still does not define the final portfolio weights.

## 19. Allocation Oracle and Validation

Every support is sent to the classical allocation oracle.

The oracle:

1. fixes outside-support weights to zero;
2. enforces selected-asset lower bounds;
3. applies effective upper bounds;
4. solves the reduced continuous model;
5. reconstructs the full-universe vector;
6. computes the canonical objective;
7. validates every hard constraint;
8. caches duplicate supports.

A support is rejected if it has incorrect cardinality, violates eligibility or
mandatory holdings, cannot satisfy position bounds, violates turnover through
forced liquidation, produces an infeasible allocation, or fails independent
validation.

## 20. Why QUBO Energy and Final Objective Differ

The QUBO uses equal proxy weights. The allocation oracle uses optimized
continuous weights.

Therefore,

$$
E(x_1)<E(x_2)
$$

does not guarantee

$$
F(w^{*}(x_{1})) \lt F(w^{*}(x_{2}))
$$

The QUBO ranks proposals. The allocation oracle determines actual financial
quality and feasibility.

## 21. X-Mixer Penalty-QAOA Ablation

The comparison method adds

$$
P\left(\sum_i x_i-r\right)^2
$$

and uses an ordinary X mixer.

The X mixer does not preserve Hamming weight. The penalty must be tuned and
invalid-cardinality samples may remain.

This ablation measures the benefit of encoding cardinality in the mixer.

## 22. Classical Baselines

The quantum method is compared with:

- exact fixed-weight enumeration when $\binom Fr$ is small;
- tabu and large-neighborhood search for larger windows.

A fair comparison uses the same window, QUBO, starting support, required weight,
allocation oracle, validation rules, and time or candidate budget.

## 23. Complexity and Practical Limits

The quantum width is $F$, not the global universe size $n$.

The exact subspace simulator scales with

$$
\binom Fr,
$$

while a full statevector scales with

$$
2^F.
$$

A global portfolio may contain thousands of assets while the quantum circuit
uses 8, 12, or 16 qubits.

This is a decomposition strategy. It does not mean the small circuit directly
solves the full global combinatorial problem.

## 24. Required Diagnostics

Preserve:

- requested backend and actual device;
- bitstring counts;
- optimized angles;
- expected surrogate energy;
- best sampled QUBO energy;
- cardinality rate;
- optimizer history;
- setup, transpilation, sampling, and decoding times;
- circuit depth and operation counts;
- allocation-oracle time;
- evaluated, feasible, and duplicate support counts;
- final validated portfolio objective.

Primary files are:

- `quantum_execution.csv`;
- `hybrid_diagnostics.json`;
- `change_windows.csv`;
- `hybrid_summary.csv`;
- `constraint_checks.csv`.

## 25. Correct Interpretation

The final quantum method supports this statement:

> Fixed-cardinality XY-QAOA proposes local asset-support changes inside an
> adaptive window, while classical optimization assigns exact weights and
> enforces the complete financial model.

It does not establish by itself:

- full-universe QPU optimization;
- quantum calculation of portfolio percentages;
- global optimality;
- financial feasibility from cardinality alone;
- quantum speedup;
- quantum advantage.

## 26. Source Files

The final quantum path is implemented in:

- `src/vanguard_portfolio/qubo_builder.py`;
- `src/vanguard_portfolio/quantum_solver.py`;
- `src/vanguard_portfolio/window_search.py`;
- `src/vanguard_portfolio/hybrid.py`;
- `src/vanguard_portfolio/allocation.py`;
- `src/vanguard_portfolio/validation.py`.

The command-line entry point is `scripts/run_hybrid.py`.

The older PCE and VQE documents describe separate experimental paths and should
not be used as the mathematical specification of the final branch.
