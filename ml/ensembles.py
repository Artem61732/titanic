"""Ensemble models and rule-based blending."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from ml.cv import cross_validate_model, evaluate_cv, iter_cv_splits
from ml.feature_engineering import FeatureBuilder
from ml.results import ModelResult, ResultsTracker
from ml.utils import feature_kwargs, random_seed


def build_shallow_rf(cfg: DictConfig) -> RandomForestClassifier:
    rf = cfg.ensemble.diverse.random_forest
    return RandomForestClassifier(
        n_estimators=int(rf.get("n_estimators", 200)),
        max_depth=int(rf.get("max_depth", 5)),
        min_samples_leaf=int(rf.get("min_samples_leaf", 5)),
        max_features=rf.get("max_features", "sqrt"),
        random_state=random_seed(cfg),
        n_jobs=-1,
    )


def build_shallow_catboost(cfg: DictConfig):
    from catboost import CatBoostClassifier

    cb = cfg.ensemble.diverse.catboost
    return CatBoostClassifier(
        iterations=int(cb.get("iterations", 300)),
        depth=int(cb.get("depth", 4)),
        learning_rate=float(cb.get("learning_rate", 0.05)),
        l2_leaf_reg=float(cb.get("l2_leaf_reg", 3.0)),
        verbose=0,
        random_seed=random_seed(cfg),
        allow_writing_files=False,
    )


def build_diverse_estimators(cfg: DictConfig) -> list[tuple[str, Any]]:
    from ml.create_submission import build_logistic_from_submission_cfg

    return [
        ("logistic", build_logistic_from_submission_cfg(cfg)),
        ("shallow_rf", build_shallow_rf(cfg)),
        ("shallow_catboost", build_shallow_catboost(cfg)),
    ]


def rule_based_predictions(df_raw: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    builder = FeatureBuilder()
    rules = set(cfg.ensemble.rule_blend.get("use_rules", ["woman", "pclass1", "child"]))
    feat = builder.featurize(df_raw)
    pred = np.zeros(len(df_raw), dtype=int)
    if "woman" in rules:
        pred |= (df_raw["Sex"].astype(str).str.lower() == "female").values
    if "pclass1" in rules:
        pred |= (df_raw["Pclass"] == 1).values
    if "child" in rules:
        pred |= feat.apply(builder._is_child_row, axis=1).values.astype(int)
    return pred


def rule_based_proba(df_raw: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    return rule_based_predictions(df_raw, cfg).astype(float)


def _predict_proba_fold(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    smin, smax = scores.min(), scores.max()
    if smax - smin < 1e-9:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - smin) / (smax - smin)


def oof_predict_proba(
    model: Any,
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    feature_mode: str | None = None,
    feature_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    builder = FeatureBuilder()
    y = df[builder.cfg.target_col]
    mode = feature_mode or str(cfg.features.mode)
    fkw = feature_kwargs or {}
    oof = np.zeros(len(df))
    skf = StratifiedKFold(
        n_splits=int(cfg.experiment.n_splits),
        shuffle=True,
        random_state=random_seed(cfg),
    )
    for train_idx, val_idx in skf.split(df, y):
        fold = builder.build_fold(df, train_idx, val_idx, mode, **fkw)
        m = clone(model)
        m.fit(fold.X_train, fold.y_train)
        oof[val_idx] = _predict_proba_fold(m, fold.X_val)
    return oof


def oof_diverse_mean_proba(
    df: pd.DataFrame,
    cfg: DictConfig,
    feature_kwargs: dict[str, Any],
) -> np.ndarray:
    mode = str(cfg.ensemble.get("feature_mode") or cfg.features.mode)
    probas = [
        oof_predict_proba(est, df, cfg, feature_mode=mode, feature_kwargs=feature_kwargs)
        for _, est in build_diverse_estimators(cfg)
    ]
    return np.mean(np.column_stack(probas), axis=1)


def _cv_score_from_proba(
    y: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> float:
    return float(accuracy_score(y, (proba >= threshold).astype(int)))


def evaluate_diverse_voting(
    df: pd.DataFrame,
    cfg: DictConfig,
    feature_kwargs: dict[str, Any],
) -> tuple[float, list[float]]:
    estimators = build_diverse_estimators(cfg)
    vote = VotingClassifier(estimators=estimators, voting="soft")
    builder = FeatureBuilder()
    y = df[builder.cfg.target_col]
    mode = str(cfg.ensemble.get("feature_mode") or cfg.features.mode)
    metrics = evaluate_cv(
        vote,
        df,
        y,
        cfg,
        "kfold",
        build_fold_fn=builder.build_fold,
        feature_mode=mode,
        feature_kwargs=feature_kwargs,
        clone_model=clone,
    )
    return metrics.accuracy_mean, metrics.fold_scores


def evaluate_rule_only(df: pd.DataFrame, cfg: DictConfig) -> float:
    y = df[FeatureBuilder().cfg.target_col].values
    return float(accuracy_score(y, rule_based_predictions(df, cfg)))


def evaluate_rule_blend_oof(
    df: pd.DataFrame,
    cfg: DictConfig,
    feature_kwargs: dict[str, Any],
) -> tuple[float, float, float]:
    y = df[FeatureBuilder().cfg.target_col].values
    ml_w = float(cfg.ensemble.rule_blend.get("ml_weight", 0.65))
    rule_w = 1.0 - ml_w
    ml_oof = oof_diverse_mean_proba(df, cfg, feature_kwargs)
    rule = rule_based_proba(df, cfg)
    blended = ml_w * ml_oof + rule_w * rule
    return (
        _cv_score_from_proba(y, blended),
        _cv_score_from_proba(y, ml_oof),
        float(accuracy_score(y, rule.astype(int))),
    )


def cv_rule_blend_by_splits(
    df: pd.DataFrame,
    cfg: DictConfig,
    feature_kwargs: dict[str, Any],
) -> tuple[float, list[float]]:
    builder = FeatureBuilder()
    y = df[builder.cfg.target_col]
    mode = str(cfg.ensemble.get("feature_mode") or cfg.features.mode)
    ml_w = float(cfg.ensemble.rule_blend.get("ml_weight", 0.65))
    rule_w = 1.0 - ml_w
    fold_scores: list[float] = []

    for split in iter_cv_splits(y, cfg, "kfold"):
        fold = builder.build_fold(
            df, split.train_idx, split.val_idx, mode, **feature_kwargs
        )
        val_raw = df.iloc[split.val_idx]
        rule_val = rule_based_proba(val_raw, cfg)
        probas = []
        for _, est in build_diverse_estimators(cfg):
            m = clone(est)
            m.fit(fold.X_train, fold.y_train)
            probas.append(_predict_proba_fold(m, fold.X_val))
        ml_val = np.mean(np.column_stack(probas), axis=1)
        blended = ml_w * ml_val + rule_w * rule_val
        y_val = y.iloc[split.val_idx].values
        fold_scores.append(_cv_score_from_proba(y_val, blended))

    return float(np.mean(fold_scores)), fold_scores


def fit_predict_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: DictConfig,
    feature_kwargs: dict[str, Any],
    *,
    method: str,
) -> np.ndarray:
    builder = FeatureBuilder()
    mode = str(
        cfg.submission.get("feature_mode")
        or cfg.ensemble.get("feature_mode")
        or cfg.features.mode
    )
    matrices = builder.build_train_test(train_df, test_df, mode, **feature_kwargs)

    if method == "rule_only":
        return rule_based_predictions(test_df, cfg)

    if method == "diverse_voting":
        vote = VotingClassifier(
            estimators=build_diverse_estimators(cfg), voting="soft"
        )
        vote.fit(matrices.X_train, matrices.y_train)
        return vote.predict(matrices.X_test).astype(int)

    probas = []
    for _, est in build_diverse_estimators(cfg):
        m = clone(est)
        m.fit(matrices.X_train, matrices.y_train)
        probas.append(_predict_proba_fold(m, matrices.X_test))
    ml_proba = np.mean(np.column_stack(probas), axis=1)

    if method == "diverse_mean":
        return (ml_proba >= 0.5).astype(int)

    if method in ("rule_blend", "diverse_rule_blend"):
        rule_test = rule_based_proba(test_df, cfg)
        ml_w = float(cfg.ensemble.rule_blend.get("ml_weight", 0.65))
        blended = ml_w * ml_proba + (1.0 - ml_w) * rule_test
        return (blended >= 0.5).astype(int)

    raise ValueError(f"Unknown ensemble method: {method!r}")

def _add_ensemble_result(
    tracker: ResultsTracker,
    name: str,
    params: str,
    acc: float,
    fold_scores: list[float],
    **extra: Any,
) -> None:
    std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    tracker.add(
        ModelResult(
            name, params, len(fold_scores), acc, std, fold_scores, stage="ensemble"
        ),
        **extra,
    )


def run_ensembles(cfg: DictConfig, tracker: ResultsTracker) -> None:
    if not cfg.ensemble.enabled:
        return
    df = FeatureBuilder().read_raw(cfg.paths.train_csv)
    y = df[FeatureBuilder().cfg.target_col].values
    fkw = feature_kwargs(cfg)
    methods = list(cfg.ensemble.methods)

    print("\n=== Ensembles: diverse + rules ===")
    print(f"  Base models: {[n for n, _ in build_diverse_estimators(cfg)]}")
    print(f"  Feature mode: {cfg.ensemble.get('feature_mode', cfg.features.mode)}")

    if "rule_only" in methods:
        acc = evaluate_rule_only(df, cfg)
        n_rules = len(cfg.ensemble.rule_blend.get("use_rules", []))
        print(f"  rule_only (train acc): {acc:.4f}")
        _add_ensemble_result(tracker, "ensemble_rule_only", f"rules={n_rules}", acc, [acc])

    if "diverse_voting" in methods:
        acc, folds = evaluate_diverse_voting(df, cfg, fkw)
        print(f"  diverse_voting CV: {acc:.4f}")
        _add_ensemble_result(tracker, "ensemble_diverse_voting", "soft", acc, folds)

    if "diverse_mean" in methods:
        ml_oof = oof_diverse_mean_proba(df, cfg, fkw)
        acc = float(accuracy_score(y, (ml_oof >= 0.5).astype(int)))
        print(f"  diverse_mean OOF acc: {acc:.4f}")
        _add_ensemble_result(tracker, "ensemble_diverse_mean", "proba_mean", acc, [acc])

    if "rule_blend" in methods:
        blend_acc, folds = cv_rule_blend_by_splits(df, cfg, fkw)
        ml_w = float(cfg.ensemble.rule_blend.ml_weight)
        print(f"  rule_blend CV ({ml_w:.0%} ML + {1-ml_w:.0%} rules): {blend_acc:.4f}")
        _add_ensemble_result(tracker, "ensemble_rule_blend", f"ml_weight={ml_w}", blend_acc, folds)

    if "diverse_rule_blend" in methods:
        blend_acc, ml_acc, rule_acc = evaluate_rule_blend_oof(df, cfg, fkw)
        ml_w = float(cfg.ensemble.rule_blend.ml_weight)
        print(
            f"  diverse_rule_blend OOF: {blend_acc:.4f} "
            f"(ML={ml_acc:.4f}, rules={rule_acc:.4f}, ml_weight={ml_w})"
        )
        _add_ensemble_result(
            tracker,
            "ensemble_diverse_rule_blend",
            f"ml_weight={ml_w}",
            blend_acc,
            [blend_acc],
            ml_oof_acc=ml_acc,
            rule_acc=rule_acc,
        )
