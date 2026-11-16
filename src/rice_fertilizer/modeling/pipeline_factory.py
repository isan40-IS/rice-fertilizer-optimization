from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .imputation import (
    ResolvedImputationPlan,
    build_mice_imputer,
    build_simple_imputer,
)

import warnings
from typing import Literal
ImputationMode = Literal["plan", "none"]
def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    decision_features: list[str],
    scale_numeric: bool,
    imputation_plan: ResolvedImputationPlan | None = None,
    imputation_mode: ImputationMode | None = None,
) -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    imputation_mode:
    - "plan": imputasi per fitur mengikuti `imputation_plan`
              (median / most_frequent / MICE).
    - "none": tidak ada imputasi. NaN dipertahankan apa adanya pada fitur
              numerik agar ditangani native oleh model (mis. XGBoost).
              Fitur kategorikal tetap di-one-hot; NaN menjadi kategori
              tersendiri (bukan default-direction native).

    Struktur:
    - numeric median      -> SimpleImputer(median)          [mode "plan"]
    - numeric most freq   -> SimpleImputer(most_frequent)   [mode "plan"]
    - numeric MICE        -> IterativeImputer               [mode "plan"]
    - numeric lainnya     -> passthrough (semua numerik pada mode "none")
    - decision features   -> passthrough
    - categorical         -> (most_frequent jika "plan") + OneHotEncoder
    """

    # =========================================================
    # 0. Validasi mode
    # =========================================================

    if imputation_mode is None:
        imputation_mode = imputation_plan.mode if imputation_plan else "plan"
    if imputation_mode not in ("plan", "none"):
        raise ValueError(
            f"imputation_mode harus 'plan' atau 'none', bukan {imputation_mode!r}."
        )

    use_imputation = imputation_mode == "plan"

    if use_imputation and imputation_plan is None:
        raise ValueError(
            "imputation_mode='plan' membutuhkan imputation_plan."
        )

    if not use_imputation and imputation_plan is not None and any((
        imputation_plan.median_columns,
        imputation_plan.most_frequent_columns,
        imputation_plan.mice_columns,
    )):
        warnings.warn(
            "imputation_plan diberikan tetapi imputation_mode='none'; "
            "plan diabaikan.",
            stacklevel=2,
        )

    transformers = []

    # Some broader profiles also list N/P/K as numeric: transform them once.
    numeric_features = [c for c in numeric_features if c not in decision_features]

    numeric_set = set(numeric_features)

    # =========================================================
    # 1. Pisahkan imputation plan berdasarkan tipe fitur
    # =========================================================

    if use_imputation:
        median_numeric = [
            col
            for col in imputation_plan.median_columns
            if col in numeric_set
        ]

        most_frequent_numeric = [
            col
            for col in imputation_plan.most_frequent_columns
            if col in numeric_set
        ]

        mice_numeric = [
            col
            for col in imputation_plan.mice_columns
            if col in numeric_set
        ]
    else:
        # Mode "none": tidak ada kolom yang diimputasi.
        median_numeric = []
        most_frequent_numeric = []
        mice_numeric = []

    # =========================================================
    # 2. Numeric features yang tidak membutuhkan imputasi
    #    (pada mode "none": semua fitur numerik)
    # =========================================================

    imputed_numeric = set(
        median_numeric
        + most_frequent_numeric
        + mice_numeric
    )

    passthrough_numeric = [
        col
        for col in numeric_features
        if col not in imputed_numeric
    ]

    # =========================================================
    # 3. Numeric — median
    # =========================================================

    if median_numeric:
        steps = [
            ("imputer", build_simple_imputer("median"))
        ]

        if scale_numeric:
            steps.append(("scaler", StandardScaler()))

        transformers.append((
            "numeric_median",
            Pipeline(steps),
            median_numeric,
        ))

    # =========================================================
    # 4. Numeric — most frequent
    # =========================================================

    if most_frequent_numeric:
        steps = [
            ("imputer", build_simple_imputer("most_frequent"))
        ]

        if scale_numeric:
            steps.append(("scaler", StandardScaler()))

        transformers.append((
            "numeric_most_frequent",
            Pipeline(steps),
            most_frequent_numeric,
        ))

    # =========================================================
    # 5. Numeric — MICE
    # =========================================================

    if mice_numeric:
        steps = [
            ("imputer", build_mice_imputer(
                imputation_plan.mice_params
            ))
        ]

        if scale_numeric:
            steps.append(("scaler", StandardScaler()))

        transformers.append((
            "numeric_mice",
            Pipeline(steps),
            mice_numeric,
        ))

    # =========================================================
    # 6. Numeric lengkap / tidak diimputasi
    #    Pada mode "none", NaN dibiarkan untuk native handling model.
    #    (StandardScaler mengabaikan NaN, tetapi scaling tidak
    #    diperlukan untuk tree-based model.)
    # =========================================================

    if passthrough_numeric:
        if scale_numeric:
            transformers.append((
                "numeric_complete",
                StandardScaler(),
                passthrough_numeric,
            ))
        else:
            transformers.append((
                "numeric_complete",
                "passthrough",
                passthrough_numeric,
            ))

    # =========================================================
    # 7. Decision variables N, P, K
    # =========================================================
    # Decision features sengaja tidak masuk state imputation.
    # Nilainya sudah ditangani sebelumnya melalui structural-zero
    # reconstruction.

    if decision_features:
        if scale_numeric:
            transformers.append((
                "decision_features",
                StandardScaler(),
                decision_features,
            ))
        else:
            transformers.append((
                "decision_features",
                "passthrough",
                decision_features,
            ))

    # =========================================================
    # 8. Categorical features
    #    Mode "none": tanpa imputer; OneHotEncoder memperlakukan
    #    NaN sebagai kategori tersendiri.
    # =========================================================

    categorical_imputed = [
        col for col in categorical_features
        if use_imputation and col in imputation_plan.most_frequent_columns
    ]
    categorical_complete = [
        col for col in categorical_features if col not in categorical_imputed
    ]
    for name, columns, impute in [
        ("categorical", categorical_imputed, True),
        ("categorical_complete", categorical_complete, False),
    ]:
        if not columns:
            continue
        categorical_steps = []

        if impute:
            categorical_steps.append((
                "imputer",
                build_simple_imputer("most_frequent"),
            ))

        categorical_steps.append((
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ))

        transformers.append((
            name,
            Pipeline(categorical_steps),
            columns,
        ))

    # =========================================================
    # 9. Final preprocessor
    # =========================================================

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(
    estimator,
    numeric_features: list[str],
    categorical_features: list[str],
    decision_features: list[str],
    model_class: str,
    imputation_plan: ResolvedImputationPlan | None = None,
    imputation_mode: ImputationMode | None = None,
) -> Pipeline:

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        decision_features=decision_features,
        scale_numeric=model_class == "linear_regression",
        imputation_plan=imputation_plan,
        imputation_mode=imputation_mode,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
