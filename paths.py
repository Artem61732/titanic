"""Project paths (resolved from repository root)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
ML_OUTPUTS_DIR = OUTPUTS_DIR / "ml"
DL_OUTPUTS_DIR = OUTPUTS_DIR / "dl"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def ensure_output_dirs() -> None:
    for directory in (DATA_DIR, ML_OUTPUTS_DIR, DL_OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
