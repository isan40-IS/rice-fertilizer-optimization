from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rice_fertilizer.config import load_all_configs
from rice_fertilizer.data.feature_engineering import add_npk_features
from rice_fertilizer.data.preprocessing_pipeline import run_preprocessing
from rice_fertilizer.exceptions import ConfigurationError, DataValidationError
from rice_fertilizer.utils import hash_file


@pytest.fixture
def cohort_setup(synthetic_data, tmp_path):
    cfg = load_all_configs()
    source = synthetic_data.copy()
    source.loc[2, "avprec.mm"] = np.nan
    source["rep_lat"] = "12,5"
    # Reference order is deliberately different from raw input order.
    reference = add_npk_features(source).iloc[[10, 2, 8]][[
        "record_id", "N.rate.kg_ha", "P.rate.kg_ha", "K.rate.kg_ha",
        "avprec.mm", "fertilizer_type", "yield.wb.kg_ha", "total_npk",
    ]].copy()
    raw = tmp_path / "raw.csv"
    ref = tmp_path / "reference.csv"
    source.to_csv(raw, index=False)
    reference.to_csv(ref, index=False)
    cfg["paths"].update({
        "input_csv": str(raw), "cohort_reference": str(ref),
        "processed_dataset": str(tmp_path / "enriched.csv"),
    })
    cfg["features"]["profiles"]["cohort_test"] = {
        "numeric": ["avprec.mm", "rep_lat", "total_npk"],
        "categorical": ["fertilizer_type"],
    }
    cfg["decisions"]["feature_profile"] = "cohort_test"
    cfg["data"]["cohort"] = {
        "mode": "reference", "expected_rows": 3,
        "input_sha256": hash_file(raw), "reference_sha256": hash_file(ref),
    }
    cfg["data"]["runtime_output"]["save_parquet"] = False
    return cfg, source, reference


def test_reference_cohort_enrichment_preserves_ids_values_and_parses_profile(cohort_setup, tmp_path):
    cfg, _, reference = cohort_setup
    # Parsing must follow the selected profile even if numeric_columns is incomplete.
    cfg["data"]["numeric_columns"] = ["yield.wb.kg_ha"]
    original_hash = hash_file(cfg["paths"]["cohort_reference"])
    result = run_preprocessing(cfg, tmp_path / "run")
    enriched = pd.read_csv(result["processed_dataset"])
    pd.testing.assert_frame_equal(enriched[reference.columns], reference.reset_index(drop=True))
    assert enriched["rep_lat"].tolist() == [12.5, 12.5, 12.5]
    assert hash_file(cfg["paths"]["cohort_reference"]) == original_hash


def test_preprocessing_refuses_reference_overwrite(cohort_setup, tmp_path):
    cfg, _, _ = cohort_setup
    cfg["paths"]["processed_dataset"] = cfg["paths"]["cohort_reference"]
    with pytest.raises(ConfigurationError, match="overwrite the reference"):
        run_preprocessing(cfg, tmp_path / "run")


def test_preprocessing_rejects_changed_source_provenance(cohort_setup, tmp_path):
    cfg, _, _ = cohort_setup
    cfg["data"]["cohort"]["input_sha256"] = "not-the-source-hash"
    with pytest.raises(DataValidationError, match="hash mismatch"):
        run_preprocessing(cfg, tmp_path / "run")
    assert not Path(cfg["paths"]["processed_dataset"]).exists()


@pytest.mark.parametrize("failure", ["duplicate_id", "missing_id", "changed_label", "dropped_row", "coordinate"])
def test_preprocessing_rejects_reference_changes_instead_of_silently_dropping_rows(
    cohort_setup, tmp_path, failure,
):
    cfg, source, _ = cohort_setup
    if failure == "duplicate_id":
        source.loc[0, "record_id"] = source.loc[1, "record_id"]
    elif failure == "missing_id":
        source.loc[10, "record_id"] = 1000
    elif failure == "changed_label":
        source.loc[10, "yield.wb.kg_ha"] += 1
    elif failure == "dropped_row":
        source.loc[10, "yield.wb.kg_ha"] = np.nan
    else:
        source.loc[10, "rep_lat"] = "100,0"
    source.to_csv(cfg["paths"]["input_csv"], index=False)
    cfg["data"]["cohort"]["input_sha256"] = hash_file(cfg["paths"]["input_csv"])
    with pytest.raises(DataValidationError):
        run_preprocessing(cfg, tmp_path / "run")
    assert not Path(cfg["paths"]["processed_dataset"]).exists()


def test_reduced_profile_preserves_unselected_reference_columns(cohort_setup, tmp_path):
    cfg, _, reference = cohort_setup
    cfg["features"]["profiles"]["cohort_test"]["numeric"].remove("total_npk")
    result = run_preprocessing(cfg, tmp_path / "run")
    processed = pd.read_csv(result["processed_dataset"])
    pd.testing.assert_frame_equal(processed[reference.columns], reference.reset_index(drop=True))
    profile = cfg["features"]["profiles"]["cohort_test"]
    model_inputs = profile["numeric"] + profile["categorical"] + cfg["features"]["decision_features"]
    assert "total_npk" in processed
    assert "total_npk" not in model_inputs
