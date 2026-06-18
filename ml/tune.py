"""Optuna-тюнинг ключевых моделей (используется из main.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from ml.feature_engineering import FeatureBuilder


def _study_name(model_name: str) -> str:
    return f"titanic_{model_name}"


def _summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.tune_dir) / "tune_summary.json"


def load_saved_tune_results(
    cfg: DictConfig,
    model_names: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    """Загрузить сохранённые результаты тюнинга из tune_summary.json."""
    summary_path = _summary_path(cfg)
    if not summary_path.exists():
        return None

    with summary_path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        return None

    requested = list(model_names) if model_names else list(cfg.tune.models)
    by_model = {str(r.get("model")): r for r in rows if isinstance(r, dict)}
    missing = [m for m in requested if m not in by_model]
    if missing:
        return None

    return [by_model[m] for m in requested]


def build_objective(
    model_name: str,
    df: pd.DataFrame,
    cfg: DictConfig,
) -> Callable[[optuna.Trial], float]:
    builder = FeatureBuilder()
    y = df[builder.cfg.target_col]
    feature_cfg = cfg.features
    tune_scheme = str(
        cfg.validation.get("tune_scheme", "kfold")
        if "validation" in cfg
        else "kfold"
    )
    cfg_tune = OmegaConf.merge(
        cfg,
        {"validation": {"n_splits": int(cfg.cv.n_splits)}},
    )

    def objective(trial: optuna.Trial) -> float:
        from ml.main import _feature_kwargs, iter_cv_splits

        model = _sample_model(trial, model_name, cfg)
        fkw = _feature_kwargs(cfg)
        fold_scores: list[float] = []
        for split in iter_cv_splits(y, cfg_tune, tune_scheme):
            fold = builder.build_fold(
                df,
                split.train_idx,
                split.val_idx,
                feature_cfg.mode,
                **fkw,
            )
            model.fit(fold.X_train, fold.y_train)
            pred = model.predict(fold.X_val)
            fold_scores.append(accuracy_score(fold.y_val, pred))
        return float(np.mean(fold_scores))

    return objective


def _sample_model(trial: optuna.Trial, model_name: str, cfg: DictConfig):
    rs = int(cfg.experiment.random_state)

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=100),
            max_depth=trial.suggest_categorical("max_depth", [4, 6, 8, 10, 12, None]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
            max_features=cast(
                Any,
                trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            ),
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            random_state=rs,
            n_jobs=-1,
        )

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 200, 1000, step=100),
            depth=trial.suggest_int("depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count=trial.suggest_int("border_count", 32, 255),
            verbose=0,
            random_seed=rs,
            allow_writing_files=False,
        )

    if model_name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            num_leaves=trial.suggest_int("num_leaves", 8, 128),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=rs,
            verbosity=-1,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=rs,
            verbosity=0,
            n_jobs=-1,
            eval_metric="logloss",
        )

    raise ValueError(f"Unsupported tune model: {model_name!r}")


def build_model_from_params(
    model_name: str, params: dict[str, Any], cfg: DictConfig
) -> Any:
    """Собрать модель по best_params из Optuna (без trial)."""
    rs = int(cfg.experiment.random_state)
    p = params

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=p["max_depth"],
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=p["max_features"],
            class_weight=p.get("class_weight"),
            random_state=rs,
            n_jobs=-1,
        )

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=int(p["iterations"]),
            depth=int(p["depth"]),
            learning_rate=float(p["learning_rate"]),
            l2_leaf_reg=float(p["l2_leaf_reg"]),
            random_strength=float(p.get("random_strength", 1.0)),
            bagging_temperature=float(p.get("bagging_temperature", 1.0)),
            border_count=int(p.get("border_count", 128)),
            verbose=0,
            random_seed=rs,
            allow_writing_files=False,
        )

    if model_name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=int(p["n_estimators"]),
            num_leaves=int(p["num_leaves"]),
            learning_rate=float(p["learning_rate"]),
            min_child_samples=int(p["min_child_samples"]),
            subsample=float(p.get("subsample", 1.0)),
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            reg_lambda=float(p.get("reg_lambda", 0.0)),
            random_state=rs,
            verbosity=-1,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            learning_rate=float(p["learning_rate"]),
            subsample=float(p.get("subsample", 1.0)),
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            min_child_weight=int(p.get("min_child_weight", 1)),
            gamma=float(p.get("gamma", 0.0)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            reg_lambda=float(p.get("reg_lambda", 0.0)),
            random_state=rs,
            verbosity=0,
            n_jobs=-1,
            eval_metric="logloss",
        )

    raise ValueError(f"Unsupported tune model: {model_name!r}")


def run_optuna_studies(
    cfg: DictConfig,
    df: pd.DataFrame | None = None,
    *,
    model_names: list[str] | None = None,
    n_trials: int | None = None,
    use_saved: bool | None = None,
) -> list[dict[str, Any]]:
    """Запуск Optuna для моделей из cfg.tune.models (или model_names)."""
    builder = FeatureBuilder()
    if df is None:
        df = builder.read_raw(str(cfg.paths.train_csv))

    out_dir = Path(cfg.paths.tune_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = list(model_names) if model_names else list(cfg.tune.models)
    allowed = {"catboost", "lightgbm", "xgboost", "random_forest"}
    unknown = set(models) - allowed
    if unknown:
        raise ValueError(f"Unsupported tune models: {sorted(unknown)}")

    use_saved_flag = (
        bool(cfg.tune.get("use_saved", True))
        if use_saved is None
        else bool(use_saved)
    )
    if use_saved_flag:
        cached = load_saved_tune_results(cfg, model_names=models)
        if cached is not None:
            print(f"\nUsing cached tune results: {_summary_path(cfg)}")
            return cached

    results: list[dict[str, Any]] = []
    n_trials_val = n_trials if n_trials is not None else int(cfg.tune.n_trials)
    timeout = cfg.tune.get("timeout_sec")
    timeout = None if timeout is None else int(timeout)

    for model_name in models:
        print(f"\n=== Optuna: {model_name} ({n_trials_val} trials) ===")
        study = optuna.create_study(
            direction="maximize",
            study_name=_study_name(model_name),
        )
        study.optimize(
            build_objective(model_name, df, cfg),
            n_trials=n_trials_val,
            timeout=timeout,
            show_progress_bar=True,
        )
        row = {
            "model": model_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }
        results.append(row)
        path = out_dir / f"{model_name}_best.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print(f"  best CV acc: {study.best_value:.4f} -> {path}")

    summary_path = out_dir / "tune_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nTune summary: {summary_path}")
    return results


if __name__ == "__main__":
    import argparse

    import bootstrap  # noqa: F401

    from config import load_config

    parser = argparse.ArgumentParser(description="ML Optuna hyperparameter tuning")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["catboost", "lightgbm", "xgboost", "random_forest"],
        help="Модели для тюнинга (по умолчанию — tune.models из config)",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Игнорировать сохранённые best params и запустить Optuna заново",
    )
    args = parser.parse_args()

    run_optuna_studies(
        load_config(),
        model_names=args.models,
        n_trials=args.n_trials,
        use_saved=not args.force,
    )
