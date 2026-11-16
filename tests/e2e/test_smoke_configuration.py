from pathlib import Path
from rice_fertilizer.config import load_yaml

def test_active_decisions_disable_imputation_and_leave_penalties_open():
    root = Path(__file__).resolve().parents[2]
    decisions = load_yaml(root / "configs" / "decisions.yaml")
    features = load_yaml(root / "configs" / "features.yaml")
    assert decisions["feature_profile"] in features["profiles"]
    assert decisions["state_imputation"]["mode"] == "none"
    assert decisions["state_imputation"]["default_method"] == "none"
    assert decisions["state_imputation"]["columns"] == {}
    assert decisions["optimization"]["lambda_total"] is None
    assert decisions["optimization"]["lambda_composition"] is None
