# Presentation plan

## Communication job

By the end, challenge judges should believe that this solver is the strongest
practical submission because it combines exact financial safety, compelling
classical scale, transparent investor controls, and a fair real-hardware
quantum experiment without overstating the evidence.

The recommended delivery is 11-13 minutes for the 14 main slides, followed by
appendix questions. Lead with the invariant—**no candidate becomes a portfolio
until allocation and independent validation pass**—and return to it when
discussing hardware noise.

## Narrative

| Slide | Takeaway title | Purpose | Suggested time |
|---:|---|---|---:|
| 1 | Constraint-safe quantum-guided portfolio optimization | Name the solver and promise: invest better without breaking guardrails | 0:25 |
| 2 | The winning problem is allocation under rules—not a low circuit energy | Translate the challenge into an investable output | 0:45 |
| 3 | One safety boundary lets every search method compete fairly | Explain support selection, allocation oracle, and validator | 0:55 |
| 4 | The model turns investor intent into measurable trade-offs | Explain risk, growth, income, cost, and hard constraints in plain language | 0:55 |
| 5 | Correctness is triangulated before scale is claimed | Show four-backend agreement, exact enumeration, and Gurobi certificate | 0:55 |
| 6 | All 17 guardrail families pass a global certificate | Make zero breaches tangible with the certified gauntlet | 0:50 |
| 7 | Investor preferences move the portfolio in the expected direction | Show growth, income, drawdown, and cost control | 0:50 |
| 8 | Tail risk falls by 54 basis points along a transparent frontier | Show scenario penalty benefit and expected-return cost | 0:50 |
| 9 | Every repeated run stays valid through 300,000 assets | Present 21/21 full-hybrid and 27/27 stretch evidence | 1:00 |
| 10 | Factor risk removes a 670 GiB memory wall | Explain linear factor storage versus quadratic dense covariance | 0:45 |
| 11 | The quantum computer proposes only a 16-asset local change | Correct the “one qubit per global asset” misconception | 0:55 |
| 12 | IBM hardware reaches 28 qubits without compromising the portfolio | Present circuit survival, depth, and 30/30 post-allocation validity | 1:00 |
| 13 | The honest audit tells us exactly what to improve | State no advantage, weak QUBO/allocation alignment, and classical LNS lead | 0:55 |
| 14 | Production-safe today; quantum-ready by design | Close with three reasons to select the solver | 0:40 |

## Appendix

| Slide | Use when judges ask… |
|---:|---|
| A1 | What exactly did you run, repeat, and certify? |
| A2 | How close did the heuristic come to the 100-asset global optimum? |
| A3 | Did the candidate methods receive comparable oracle budgets? |
| A4 | What do “certified,” “bounded,” “heuristic,” and “hardware observation” mean? |

## Delivery notes

- Open with the operational consequence: a noisy or bad candidate is rejected,
  not repaired into an unsafe recommendation.
- When showing a negative objective, say once: “lower is better; this is a
  composite score, not a return percentage.”
- Distinguish the three scaling statements verbally:
  1. exact certificates on tractable cases;
  2. repeated full-hybrid evidence through 20,000 assets; and
  3. heuristic stretch engineering through 300,000 assets.
- On the IBM slide, emphasize that raw cardinality survival falls with width,
  yet the portfolio breach count remains zero because hardware cannot bypass
  the allocation and validation layers.
- Do not defend a quantum-advantage claim. Use the fair comparison as evidence
  of scientific discipline and as a precise roadmap: better allocation-aware
  surrogate first, more depth second.
- If a live demo is possible, use the Copilot to move from “balanced” to
  “drawdown control,” show lower volatility/CVaR, then open the validation view.
  Avoid a long re-run of the 300,000-asset benchmark during judging.

## Evidence map

Every visible number is sourced in slide notes and in
[`../../results/final_submission/claim_evidence_map.csv`](../../results/final_submission/claim_evidence_map.csv).
The main visual sources are the certified guardrail, preference, scenario,
scaling, memory, and IBM frontier figures in
[`../../results/final_submission/figures/`](../../results/final_submission/figures/).
