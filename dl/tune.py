"""Optuna-тюнинг гиперпараметров DNN (Titanic MLP / EmbeddingMLP)."""

from __future__ import annotations

import argparse
import json
from typing import Any

import bootstrap  # noqa: F401
import numpy as np
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold

from config import FeatureMode, load_config, resolve_feature_mode, with_train_overrides
from dl.feature_engineering import FeatureBuilder
from dl.train import prepare_fold, train_config_label, train_fold
from paths import DL_BEST_PARAMS_PATH, ensure_output_dirs

REFINED_BATCH_SIZES = [32, 64, 128]
WIDE_BATCH_SIZES = [16, 32, 64, 128, 256]


def _sample_train_cfg(
    trial: optuna.Trial,
    base: DictConfig,
    search_space: str,
    feature_mode: FeatureMode,
) -> DictConfig:
    search_space = search_space.lower()
    if search_space == "refined":
        overrides: dict[str, Any] = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", REFINED_BATCH_SIZES),
            "scheduler": trial.suggest_categorical("scheduler", ["plateau", "cosine"]),
            "mlp_dropout": trial.suggest_float("mlp_dropout", 0.1, 0.4),
            "use_class_weights": trial.suggest_categorical(
                "use_class_weights", [False, True]
            ),
        }
    else:
        overrides = {
            "lr": trial.suggest_float("lr", 5e-5, 5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", WIDE_BATCH_SIZES),
            "scheduler": trial.suggest_categorical(
                "scheduler", ["plateau", "cosine", "none"]
            ),
            "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.5),
            "use_class_weights": trial.suggest_categorical(
                "use_class_weights", [False, True]
            ),
        }
    if feature_mode == FeatureMode.EMBEDDING:
        emb_choices = [4, 8, 16] if search_space == "refined" else [2, 4, 8, 16, 32]
        overrides["embedding_dim"] = trial.suggest_categorical(
            "embedding_dim", emb_choices
        )
    return with_train_overrides(base, **overrides).train


def cv_mean_accuracy(
    exp: DictConfig,
    train_cfg: DictConfig,
    feature_mode: FeatureMode,
    n_splits: int,
    random_state: int,
    *,
    trial: optuna.Trial | None = None,
) -> float:
    builder = FeatureBuilder()
    df = builder.read_raw(str(exp.paths.train_csv))
    y = df[builder.cfg.target_col]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        train_loader, val_loader, model, y_train = prepare_fold(
            df,
            train_idx,
            val_idx,
            feature_mode,
            train_cfg.batch_size,
            device,
            random_state,
            fold,
            builder,
            train_cfg,
        )
        _, val_acc, _ = train_fold(
            model,
            train_loader,
            val_loader,
            y_train,
            device,
            train_cfg,
            verbose=False,
        )
        fold_scores.append(val_acc)
        if trial is not None:
            trial.report(float(np.mean(fold_scores)), fold - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


def make_objective(
    exp: DictConfig,
    feature_mode: FeatureMode,
    n_splits: int,
    random_state: int,
    search_space: str,
    n_epochs: int,
    patience: int,
):
    tune_base = with_train_overrides(
        exp,
        max_epochs=n_epochs,
        early_stopping_patience=patience,
    )

    def objective(trial: optuna.Trial) -> float:
        train_cfg = _sample_train_cfg(
            trial, tune_base, search_space, feature_mode
        )
        return cv_mean_accuracy(
            exp,
            train_cfg,
            feature_mode,
            n_splits,
            random_state,
            trial=trial,
        )

    return objective


def tune_dnn(
    *,
    n_trials: int | None = None,
    n_splits: int | None = None,
    n_epochs: int | None = None,
    patience: int | None = None,
    search_space: str | None = None,
    feature_mode: str | None = None,
) -> dict[str, Any]:
    ensure_output_dirs()
    exp = load_config()
    mode = resolve_feature_mode(feature_mode, exp)

    tune_cfg = exp.tune
    n_trials = n_trials if n_trials is not None else int(tune_cfg.n_trials)
    n_splits = n_splits if n_splits is not None else int(exp.n_splits)
    n_epochs = n_epochs if n_epochs is not None else int(tune_cfg.n_epochs)
    patience = patience if patience is not None else int(tune_cfg.patience)
    search_space = search_space or str(tune_cfg.search_space)

    random_state = int(exp.random_state)
    print(
        f"DL Optuna | mode={mode.value} | {n_trials} trials | "
        f"{n_splits}-fold | epochs≤{n_epochs} | space={search_space}"
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"titanic_dl_{mode.value}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        make_objective(
            exp, mode, n_splits, random_state, search_space, n_epochs, patience
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_train = with_train_overrides(
        exp,
        **study.best_params,
        max_epochs=n_epochs,
        early_stopping_patience=patience,
    ).train

    result = {
        "feature_mode": mode.value,
        "best_cv_accuracy": study.best_value,
        "best_params": dict(study.best_params),
        "train": OmegaConf.to_container(best_train, resolve=True),
        "n_trials": len(study.trials),
        "n_splits": n_splits,
        "search_space": search_space,
    }
    print(
        f"\nBest CV acc: {study.best_value:.4f} | "
        f"{train_config_label(best_train)}"
    )

    DL_BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DL_BEST_PARAMS_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {DL_BEST_PARAMS_PATH}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DL Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--search-space",
        choices=["refined", "wide"],
        default=None,
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "onehot", "embedding"],
        default=None,
        help="Режим признаков (по умолчанию — feature_mode из config)",
    )
    args = parser.parse_args()
    tune_dnn(
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        n_epochs=args.n_epochs,
        patience=args.patience,
        search_space=args.search_space,
        feature_mode=args.mode,
    )


if __name__ == "__main__":
    main()
