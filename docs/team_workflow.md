# Team Workflow

## Branch policy

`main` should remain runnable and reviewed, but it does not need to remain frozen
until the entire project ends. Merge small completed features after tests and
review; do not accumulate the whole challenge in one final merge.

Suggested branches:

- `classical-baseline` - canonical model, solvers, benchmark, and plots;
- `qubo-validation` - encoding and energy-equivalence tests;
- `hybrid-support-selection` - quantum/classical support experiments;
- `copilot-ui` - interface that consumes validated result objects.

## Before editing

```bash
git switch main
git pull --ff-only origin main
git switch -c classical-baseline
python -m pip install -e ".[qp,test]"
```

## Before committing

```bash
python -m pytest -q
python scripts/run_classical.py --config configs/tiny_example.yaml
git status
git diff --check
```

Commit only relevant source/docs/tests. Generated runtime tables should usually
remain uncommitted; commit selected figures only when cited.

## Review checklist

- Does the change import the canonical schema/objective?
- Are new constraints enforced by every applicable backend and validator?
- Is a direct unit test added?
- Are exact/heuristic/continuous claims labeled correctly?
- Is optional software handled without breaking the open-source path?
- Does the comparison include model construction and post-processing time?
- Do all figures have readable axes, units, and titles?

## Merge

```bash
git add <specific-files>
git commit -m "Finalize classical portfolio baseline"
git push -u origin classical-baseline
```

Open a pull request. Merge after review and passing checks; then teammates should
update their branches from the new `main` before continuing.

