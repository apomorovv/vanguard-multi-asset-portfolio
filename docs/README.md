# Documentation guide

The documentation moves from the investment problem to the implementation and
then to the evidence. New readers should follow the first column in order.

| Order | Document | What it answers |
|---:|---|---|
| 1 | [`portfolio_optimization_report.md`](portfolio_optimization_report.md) | What was studied, what the literature says, what was found, and what is not claimed? |
| 2 | [`presentation/portfolio_optimization_challenge_deck.pdf`](presentation/portfolio_optimization_challenge_deck.pdf) | What is the concise challenge story? |
| 3 | [`presentation/README.md`](presentation/README.md) | How should the deck be delivered and which evidence backs each slide? |
| 4 | [`mathematical_model.md`](mathematical_model.md) | What are the variables, objective terms, and financial guardrails? |
| 5 | [`final_hybrid_model.md`](final_hybrid_model.md) | How does the end-to-end solver produce and validate a portfolio? |
| 6 | [`classical_model_formulation.md`](classical_model_formulation.md) | Which continuous, mixed-integer, enumeration, annealing, and LNS baselines are implemented? |
| 7 | [`quantum_model_formulation.md`](quantum_model_formulation.md) | How are the window QUBO, XY mixer, candidate decoding, and allocation handoff defined? |
| 8 | [`validation_protocol.md`](validation_protocol.md) | What must pass before a result can be accepted? |
| 9 | [`ibm_qpu_experiment.md`](ibm_qpu_experiment.md) | How should IBM hardware be run and compared fairly? |
| 10 | [`installation.md`](installation.md) | How are CPU, GPU, Gurobi, and IBM environments installed? |
| 11 | [`team_workflow.md`](team_workflow.md) | How the team collaborates and manages the project? |

The machine-readable audit trail is in
[`../results/final_submission/`](../results/final_submission/). Use its claim
map rather than reading numerical values from figures by eye.

## Core distinction

- **Support selection** chooses the assets.
- **Allocation** assigns continuous portfolio percentages.
- **Validation** recomputes every financial rule.
- **Certification** supplies a bound on global solution quality when an exact
  mixed-integer solver completes.

The quantum component participates only in support selection inside a small
adaptive window. It never assigns the final percentages or bypasses validation.
