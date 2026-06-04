from __future__ import annotations

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = CODE_DIR.parent
ARTIFACT_DIR = CODE_DIR / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"
WORKSPACE_DIR = ARTIFACT_DIR / "workspace"


def find_repo_root() -> Path:
    """Find the shared repository root by walking upward until data/ exists."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").exists():
            return parent
    return CODE_DIR


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data"


def ensure_artifact_dirs() -> None:
    for path in (FIGURE_DIR, TABLE_DIR, WORKSPACE_DIR):
        path.mkdir(parents=True, exist_ok=True)
