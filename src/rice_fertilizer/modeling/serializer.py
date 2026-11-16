from pathlib import Path
from typing import Any
import joblib

def save_model_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path

def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(path))
