# Decision Checklist

| Decision | Configuration | Evidence required |
|---|---|---|
| Ratio pseudocount | `decisions.feature_engineering.ratio_pseudocount` | Distribution and zero-denominator behavior |
| Target | `decisions.target_column` | Coverage and wet-basis limitation |
| Feature profile | `decisions.feature_profile` | Missingness, performance, leakage, relevance |
| Group key | `decisions.group_column` | Experiment-ID uniqueness |
| Validation | `decisions.validation_strategy` | Intended generalization claim |
| Baseline | `decisions.baseline_policy` | Fair same-state comparison |
| NPK bounds | `decisions.optimization.bounds` | Historical domain and expert/literature |
| Total thresholds | `total_soft`, `total_max` | Data/literature/expert |
| Proportion ranges | `proportion_bounds` | Agronomic rationale |
| Penalty weights | `lambda_*` | Scale and ablation analysis |
| Perturbation | `decisions.perturbations` | Measurement uncertainty and model resolution |
| Agronomic warning | `agronomic_validation` | Expert/literature threshold |
