from __future__ import annotations
from pathlib import Path
import shutil
import yaml
from ..config import ensure_runtime_directories, load_all_configs, deep_merge
from ..utils import make_run_id, save_json

def initialize_run(config_dir: str | Path | None = None, run_id: str | None = None, overrides: dict | None = None):
    configs = load_all_configs(config_dir)
    if overrides:
        configs = deep_merge(configs, overrides)
    directories = ensure_runtime_directories(configs)
    run_id = run_id or make_run_id("ta")
    run_dir = directories["artifacts_root"] / run_id
    if overrides and run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite study run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config_snapshot"
    snapshot.mkdir(exist_ok=True)
    source = (
        Path(config_dir)
        if config_dir else Path(__file__).resolve().parents[3] / "configs"
    )
    for path in source.glob("*.yaml"):
        if overrides:
            (snapshot / path.name).write_text(yaml.safe_dump(configs[path.stem], sort_keys=False), encoding="utf-8")
        else:
            shutil.copy2(path, snapshot / path.name)
    save_json({"run_id": run_id}, run_dir / "run_metadata.json")
    return configs, run_dir
