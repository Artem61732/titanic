"""
Titanic — единая точка входа.

Быстрый старт:
    pip install -r requirements.txt
    python main.py

По умолчанию запускается ML-пайплайн:
  загрузка данных -> препроцессинг -> CV -> таблица результатов -> submission.csv
"""

from __future__ import annotations

import argparse
import sys

import bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Titanic pipeline: CV, results table, Kaggle submission",
    )
    parser.add_argument(
        "--pipeline",
        choices=("ml", "dl", "all"),
        default="ml",
        help="ml (default): full ML pipeline; dl: neural net submission; all: both",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Faster run with smaller model grids and fewer Optuna trials",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a custom YAML config",
    )
    return parser.parse_args()


def run_ml_pipeline(cfg, *, quick: bool) -> None:
    from ml.main import run_pipeline

    if quick:
        cfg.experiment.quick = True
        cfg.tune.n_trials = 5
        if "validation" in cfg:
            cfg.validation.n_repeats = 2

    run_pipeline(cfg, stage="all")


def run_dl_pipeline(cfg) -> None:
    from dl.create_submission import create_submission

    create_submission(
        experiment=cfg,
        output_path=cfg.paths.get("dl_submission_csv", cfg.paths.submission_csv),
    )


def main() -> int:
    from config import load_config

    args = parse_args()
    project_cfg = load_config(args.config)

    if args.pipeline in ("ml", "all"):
        print("=== ML pipeline ===")
        run_ml_pipeline(project_cfg, quick=args.quick)

    if args.pipeline in ("dl", "all"):
        print("\n=== DL submission ===")
        run_dl_pipeline(project_cfg)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
