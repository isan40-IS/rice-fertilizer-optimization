from pathlib import Path
from rice_fertilizer.config import load_yaml

def test_decision_template_is_intentionally_incomplete():
    root = Path(__file__).resolve().parents[2]
    decisions = load_yaml(root / "configs" / "decisions.yaml")
    assert decisions["feature_profile"] is None
    assert decisions["optimization"]["bounds"]["N"]["max"] is None
