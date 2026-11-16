from __future__ import annotations
from pathlib import Path
import shutil
from ..config import ensure_runtime_directories, load_all_configs
from ..utils import make_run_id, save_json

def initialize_run(config_dir: str | Path | None = None, run_id: str | None = None):
    configs = load_all_configs(config_dir)
    directories = ensure_runtime_directories(configs)
    run_id = run_id or make_run_id("ta")
    run_dir = directories["artifacts_root"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config_snapshot"
    snapshot.mkdir(exist_ok=True)
    source = (
        Path(config_dir)
        if config_dir else Path(__file__).resolve().parents[3] / "configs"
    )
    for path in source.glob("*.yaml"):
        shutil.copy2(path, snapshot / path.name)
    save_json({"run_id": run_id}, run_dir / "run_metadata.json")
    return configs, run_dir
