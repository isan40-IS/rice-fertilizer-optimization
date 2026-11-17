from __future__ import annotations
from pathlib import Path
from typing import Any
import logging
import pandas as pd
from ..config import require_value, resolve_path
from ..exceptions import ConfigurationError
from ..utils import hash_file, save_dataframe, save_json
from .loader import add_record_id, build_composite_column, load_csv
from .validator import (
    coerce_numeric_columns, duplicate_report, find_numeric_anomalies,
    profile_columns, validate_required_columns, validate_target,
)
from .crop_filter import filter_crop
from .missing_values import reconstruct_structural_zeroes
from .cleaner import filter_valid_target_rows, standardize_string_columns, filter_valid_decision_rows
from .feature_engineering import add_npk_features, select_feature_frame
from .cohort import align_reference_cohort, validate_reference_preserved
from ..exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)

def _selected_profile(configs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile_name = require_value(configs["decisions"], "feature_profile")
    profiles = configs["features"]["profiles"]
    if profile_name not in profiles:
        raise ConfigurationError(
            f"Unknown feature profile '{profile_name}'. Available: {sorted(profiles)}"
        )
    return profile_name, profiles[profile_name]

def run_preprocessing(configs: dict[str, Any], run_dir: Path) -> dict[str, Path]:
    data_cfg, paths_cfg = configs["data"], configs["paths"]
    input_path = resolve_path(paths_cfg["input_csv"])
    processed_path = resolve_path(paths_cfg["processed_dataset"])
    if paths_cfg.get("processed_dataset_read_only", False):
        raise ConfigurationError("This configuration reuses a read-only processed dataset; do not run preprocessing")
    if processed_path.resolve() == input_path.resolve():
        raise ConfigurationError("Processed output must not overwrite raw input")
    audit_dir = run_dir / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    df = standardize_string_columns(
        add_record_id(
            load_csv(input_path, data_cfg["read_csv"]),
            data_cfg["identifier_columns"]["record_id"],
        )
    )
    target = configs["decisions"].get("target_column") or data_cfg["target_column"]
    reference = None
    cohort_metadata = {"mode": "all_eligible"}
    cohort_cfg = data_cfg.get("cohort", {})
    cohort_mode = cohort_cfg.get("mode", "all_eligible")
    if cohort_mode not in {"all_eligible", "reference"}:
        raise ConfigurationError("data.cohort.mode must be all_eligible or reference")
    if cohort_mode == "reference":
        reference_path = resolve_path(require_value(paths_cfg, "cohort_reference"))
        if reference_path.resolve() in {
            processed_path.resolve(), processed_path.with_suffix(".parquet").resolve(),
        }:
            raise ConfigurationError("Processed output must not overwrite the reference cohort")
        for path, key in [
            (input_path, "input_sha256"), (reference_path, "reference_sha256"),
        ]:
            if hash_file(path) != require_value(cohort_cfg, key):
                raise DataValidationError(f"Cohort provenance hash mismatch: {path}")
        reference = pd.read_csv(reference_path)
        df = align_reference_cohort(
            df, reference, data_cfg["identifier_columns"]["record_id"],
            int(require_value(cohort_cfg, "expected_rows")),
        )
        cohort_metadata = {
            "mode": "reference", "reference_path": str(reference_path),
            "reference_sha256": hash_file(reference_path),
            "expected_rows": len(reference),
        }
    decision_features = configs["features"]["decision_features"]
    profile_name, profile = _selected_profile(configs)
    engineered_features = set(configs["features"].get("engineered_features", []))
    source_numeric_features = [
        column for column in profile["numeric"] if column not in engineered_features
    ]
    required = list(dict.fromkeys([
        data_cfg["crop"]["column"], target,
        data_cfg["structural_zero"]["indicator_column"],
        *data_cfg["nutrient_rates"].values(),
        *source_numeric_features, *profile["categorical"],
    ]))
    validate_required_columns(df, required)
    
    numeric_columns = list(dict.fromkeys([
        *data_cfg.get("numeric_columns", []), *source_numeric_features,
        *data_cfg["nutrient_rates"].values(), target,
    ]))
    df, coercion = coerce_numeric_columns(df, numeric_columns)
    for column, limits in data_cfg.get("numeric_ranges", {}).items():
        if column in df:
            observed = df[column].dropna()
            if not observed.between(limits["min"], limits["max"]).all():
                raise DataValidationError(f"Values outside configured range: {column}")
    raw_profile = profile_columns(df)
    raw_anomalies = find_numeric_anomalies(df, numeric_columns)
    duplicates = duplicate_report(df)

    df = filter_crop(df, **data_cfg["crop"])
    validate_target(df, target)
    zero_summary, label_audit = pd.DataFrame(), pd.DataFrame()
    structural = data_cfg["structural_zero"]
    if structural.get("enabled", True):
        df, zero_summary, label_audit = reconstruct_structural_zeroes(
            df,
            structural["indicator_column"],
            data_cfg["nutrient_rates"],
            structural.get("control_labels", []),
            structural.get("explicit_label_map", {}),
        )

    # Remove rows whose fertilizer rates are still unknown
    decision_columns = list(data_cfg["nutrient_rates"].values())

    df, removed_decision_missing = filter_valid_decision_rows(
        df,
        decision_columns,
    )

    df, removed_target = filter_valid_target_rows(df, target)

    ratio_pseudocount = float(
        require_value(
            configs["decisions"],
            "feature_engineering.ratio_pseudocount",
        )
    )

    df = add_npk_features(
        df,
        data_cfg["nutrient_rates"]["N"],
        data_cfg["nutrient_rates"]["P"],
        data_cfg["nutrient_rates"]["K"],
        ratio_pseudocount=ratio_pseudocount,
    )

    for group_cfg in configs["validation"].get("composite_groups", {}).values():
        df = build_composite_column(
            df, group_cfg["output_column"], group_cfg["source_columns"]
        )

    passthrough = [
        data_cfg["identifier_columns"][key]
        for key in ["record_id", "article_id", "experiment_id", "year", "location"]
        if data_cfg["identifier_columns"][key] in df.columns
    ]
    passthrough += [
        column for column in [
            structural["indicator_column"],
            "group_article_experiment",
            "group_article_experiment_year",
        ] if column in df.columns
    ]
    model_df = select_feature_frame(
        df, profile["numeric"], profile["categorical"], target,
        decision_features=decision_features, passthrough_columns=passthrough,
    )
    if reference is not None:
        validate_reference_preserved(model_df, reference)
        cohort_metadata["reference_values_preserved"] = True
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(processed_path, index=False)
    if data_cfg["runtime_output"].get("save_parquet", True):
        try:
            model_df.to_parquet(processed_path.with_suffix(".parquet"), index=False)
        except (ImportError, ModuleNotFoundError):
            LOGGER.warning("Parquet output skipped")

    for name, frame in {
        "raw_column_profile": raw_profile,
        "raw_numeric_anomalies": raw_anomalies,
        "numeric_coercion": coercion,
        "processed_column_profile": profile_columns(model_df),
        "structural_zero_summary": zero_summary,
        "structural_zero_label_audit": label_audit,
    }.items():
        if not frame.empty:
            save_dataframe(frame, audit_dir / name, save_parquet=False)

    save_json({
        "input_path": str(input_path),
        "input_sha256": hash_file(input_path),
        "processed_path": str(processed_path),
        "processed_sha256": hash_file(processed_path),
        "cohort": cohort_metadata,
        "feature_profile": profile_name,
        "target": target,
        "rows_processed": int(len(model_df)),
        "removed_missing_target": removed_target,
        "removed_missing_decision_features": removed_decision_missing,
        "duplicates": duplicates,
        "numeric_features": profile["numeric"],
        "categorical_features": profile["categorical"],
        "ratio_pseudocount": ratio_pseudocount,
    }, audit_dir / "preprocessing_metadata.json")
    LOGGER.info("Processed dataset written to %s", processed_path)
    return {"processed_dataset": processed_path, "audit_dir": audit_dir}
