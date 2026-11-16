# Architecture

```text
scripts/
  thin command wrappers
        ↓
src/rice_fertilizer/pipelines/
  orchestration
        ↓
data → validation → modeling → surrogate → optimization → evaluation → reporting
```

## Dependency direction

- `data` does not import modeling or optimization.
- `modeling` does not import optimization.
- `surrogate` wraps the trained pipeline and reconstructs every candidate feature row.
- `optimization` only calls the surrogate interface.
- `evaluation` consumes results and may rerun optimization for perturbation analyses.
- `reporting` does not modify scientific results.

## Critical contract

The optimizer never calls a raw estimator directly. It calls `SurrogatePredictor`, ensuring all NPK-derived features are recalculated whenever N, P, or K changes.
