from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TABLE_DIR, ensure_artifact_dirs


def write_table(df: pd.DataFrame, name: str) -> Path:
    ensure_artifact_dirs()
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)
