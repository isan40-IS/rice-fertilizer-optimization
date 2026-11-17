import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rice_fertilizer.config import load_all_configs
from rice_fertilizer.data.feature_engineering import add_npk_features
from rice_fertilizer.pipelines.run_experiment_c import run_experiment, evaluate_frozen_candidate
from rice_fertilizer.modeling.serializer import load_model_bundle
from rice_fertilizer.exceptions import ConfigurationError


def test_repeated_studies_and_explicit_holdout(synthetic_data, tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from rice_fertilizer.modeling import trainer
    configs = load_all_configs(Path(__file__).resolve().parents[2]/"configs/experiment_C")
    data = add_npk_features(synthetic_data)
    data["initial.ph.soil"] = 6.5
    data["soc.percentage"] = 1.2
    data.loc[::3, "avprec.mm"] = np.nan
    data["group_article_experiment"] = data.id.astype(str)+"::"+data["exp.id"].astype(str)
    dataset = tmp_path/"synthetic.csv"
    data.to_csv(dataset, index=False)
    configs["paths"].update(processed_dataset=str(dataset), artifacts_root=str(tmp_path/"artifacts"))
    configs["validation"]["n_splits"] = 2
    configs["experiment"] = {"methods": ["random", "optuna"], "search_seeds": [11, 29]}
    configs["models"]["selection"]["minimum_r2"] = None
    configs["models"]["tuning"]["random"]["n_iter"] = 2
    configs["models"]["tuning"]["optuna"]["n_trials"] = 2
    configs["models"]["models"] = {
        "xgboost": {"class": "xgboost", "params": {"n_estimators": 3, "n_jobs": 1},
                    "search_space": {"model__max_depth": [1, 2]}},
        "catboost": {"class": "catboost", "params": {"iterations": 3, "thread_count": 1, "verbose": False, "allow_writing_files": False},
                     "search_space": {"model__depth": [1, 2]}},
    }
    config_dir = tmp_path/"configs"
    config_dir.mkdir()
    for name, cfg in configs.items():
        (config_dir/f"{name}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # Any accidental holdout evaluation while searching must fail the test.
    monkeypatch.setattr(trainer, "evaluate_holdout", lambda *a, **k: pytest.fail("holdout used during search"))
    group = run_experiment(config_dir)
    comparison = pd.read_csv(group/"study_comparison.csv")
    assert len(comparison) == 8
    assert not any(c.startswith("holdout_") for c in comparison.columns)
    manifests = []
    for run in sorted(group.glob("*_seed_*")):
        manifest = json.loads((run/"training_manifest.json").read_text())
        manifests.append(manifest["cv_manifest_sha256"])
        snapshot = load_all_configs(run/"config_snapshot")
        assert snapshot["models"]["tuning"]["method"] == run.name.split("_seed_")[0]
        assert not snapshot["models"]["report_holdout"]
        for model in ["xgboost", "catboost"]:
            bundle = load_model_bundle(run/"models"/f"{model}.joblib")
            assert bundle["holdout_metrics"] == {}
            assert bundle["tuning"]["n_complete"] == 2
            assert bundle["tuning"]["method"] == snapshot["models"]["tuning"]["method"]
            assert (run/"tuning"/f"{model}_search_results.csv").exists()
    assert len(set(manifests)) == 1
    frozen = json.loads((group/"frozen_candidate.json").read_text())
    assert frozen["candidate"]["rmse_mean"] == comparison.rmse_mean.min()
    assert not (group/"selected_holdout").exists()
    metrics = evaluate_frozen_candidate(group)
    assert metrics["rmse"] > 0
    assert (group/"selected_holdout/predictions.csv").exists()
    # Refuse evaluation if data have changed after parameter selection.
    data.loc[0, "yield.wb.kg_ha"] += 1
    data.to_csv(dataset, index=False)
    with pytest.raises(ConfigurationError, match="Dataset changed"):
        evaluate_frozen_candidate(group)


def test_c_preprocessing_cannot_overwrite_reused_b_dataset(tmp_path):
    from rice_fertilizer.data.preprocessing_pipeline import run_preprocessing
    cfg = load_all_configs(Path(__file__).resolve().parents[2]/"configs/experiment_C")
    with pytest.raises(ConfigurationError, match="read-only"):
        run_preprocessing(cfg, tmp_path/"run")
