"""DL K-fold CV — точка входа (аналог house_prices/dl/main.py)."""

from __future__ import annotations

import argparse

import bootstrap  # noqa: F401

from config import FeatureMode, load_config, resolve_feature_mode, with_train_overrides
from dl.train import (
    _MODE_LABELS,
    hyperparameter_grid_search,
    stratified_kfold_cv,
)


def evaluate_dnn_experiments(
    modes: list[str] | None = None,
    *,
    run_grid: bool = False,
    run_cosine: bool = False,
) -> list[dict]:
    cfg = load_config()
    if modes:
        mode_list = [FeatureMode(m) for m in modes]
    else:
        mode_list = [resolve_feature_mode(None, cfg)]

    results: list[dict] = []
    for mode in mode_list:
        label = _MODE_LABELS.get(mode, mode.value)
        print(f"\n=== CV: {label} (plateau) ===")
        results.append(stratified_kfold_cv(cfg, feature_mode=mode))
        if run_cosine:
            print(f"\n=== CV: {label} (cosine) ===")
            cfg_cos = with_train_overrides(cfg, scheduler="cosine")
            results.append(stratified_kfold_cv(cfg_cos, feature_mode=mode))

    if run_grid:
        for mode in mode_list:
            hyperparameter_grid_search(cfg, feature_mode=mode)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DL stratified K-fold CV")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "onehot", "embedding"],
        default=None,
        help="Режимы признаков (по умолчанию — feature_mode из config)",
    )
    parser.add_argument("--grid", action="store_true", help="Запустить grid search")
    parser.add_argument(
        "--cosine",
        action="store_true",
        help="Дополнительно CV с cosine scheduler",
    )
    args = parser.parse_args()
    evaluate_dnn_experiments(args.modes, run_grid=args.grid, run_cosine=args.cosine)


if __name__ == "__main__":
    main()
