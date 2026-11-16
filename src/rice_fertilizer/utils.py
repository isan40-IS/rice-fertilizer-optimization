from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, logging, random
from typing import Any
import numpy as np
import pandas as pd

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"

def hash_file(path: str | Path, chunk_size: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def save_dataframe(df: pd.DataFrame, path_without_suffix: str | Path, save_parquet: bool = True) -> None:
    base = Path(path_without_suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    if save_parquet:
        try:
            df.to_parquet(base.with_suffix(".parquet"), index=False)
        except (ImportError, ModuleNotFoundError):
            logging.getLogger(__name__).warning("Parquet dependency unavailable; skipped")
