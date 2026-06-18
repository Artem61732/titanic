"""
Titanic Kaggle — классический ML пайплайн.

Запуск из корня:
  python main.py
  python main.py --quick
  python -m ml.main --stage all
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml.feature_engineering import FeatureBuilder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Config & results


@dataclass
class ModelResult:
    name: str
    params: str
    n_splits: int
    val_acc_mean: float
    val_acc_std: float
    fold_scores: list[float]
    stage: str = "cv"
    pred_positive_rate_mean: float = 0.0
    target_rate: float = 0.0
    calibration_error: float = 0.0
    cv_scheme: str = "kfold"


class ResultsTracker:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []

    def add(self, result: ModelResult, **extra: Any) -> None:
        self.rows.append(
            {
                "name": result.name,
                "params": result.params,
                "n_splits": result.n_splits,
                "val_acc_mean": result.val_acc_mean,
                "val_acc_std": result.val_acc_std,
                "fold_scores": result.fold_scores,
                "stage": result.stage,
                **extra,
            }
        )

    def save(self) -> Path:
        df = pd.DataFrame(self.rows)
        csv_path = self.out_dir / "results.csv"
        json_path = self.out_dir / "results.json"
        df.to_csv(csv_path, index=False)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {csv_path}")
        return csv_path

    def leaderboard(self, top: int = 15) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame(self.rows).sort_values(
            "val_acc_mean", ascending=False
        )
        print(f"\n=== Top {top} (CV accuracy) ===")
        for _, r in df.head(top).iterrows():
            print(
                f"  {r['val_acc_mean']:.4f} +/- {r['val_acc_std']:.4f} | "
                f"{r['n_splits']}-fold | {r['name']} | {r['params']}"
            )
        return df.head(top)


def _rs(cfg: DictConfig) -> int:
    return int(cfg.experiment.random_state)


def build_logistic_from_submission_cfg(cfg: DictConfig) -> LogisticRegression:
    log_cfg = cfg.submission.get("logistic") or {}
    penalty = str(log_cfg.get("penalty", "l2")).lower()
    c = float(log_cfg.get("C", 0.1))
    max_iter = int(log_cfg.get("max_iter", 2000))
    rs = _rs(cfg)

    if penalty == "l1":
        return LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            C=c,
            max_iter=max_iter,
            random_state=rs,
        )
    if penalty == "elasticnet":
        return LogisticRegression(
            solver="saga",
            C=c,
            l1_ratio=float(log_cfg.get("l1_ratio", 0.5)),
            max_iter=max_iter,
            random_state=rs,
        )
    return LogisticRegression(C=c, l1_ratio=0.0, max_iter=max_iter, random_state=rs)


# Validation


@dataclass(frozen=True)
class CVSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    fold_id: int
    repeat_id: int
    scheme: str


@dataclass
class CVMetrics:
    accuracy_mean: float
    accuracy_std: float
    fold_scores: list[float]
    pred_positive_rate_mean: float
    pred_positive_rate_std: float
    fold_pred_rates: list[float]
    target_rate: float
    calibration_error: float
    n_evaluations: int
    scheme: str


def target_survival_rate(y: pd.Series, cfg: DictConfig) -> float:
    v = cfg.validation.get("target_survival_rate")
    if v is not None:
        return float(v)
    return float(y.mean())


def iter_cv_splits(
    y: pd.Series,
    cfg: DictConfig,
    scheme: str,
) -> Iterator[CVSplit]:
    """Генератор сплитов: holdout | kfold | repeated_kfold."""
    rs = int(cfg.experiment.random_state)
    n = len(y)
    idx = np.arange(n)

    if scheme == "holdout":
        holdout = float(cfg.validation.get("holdout_size", 0.2))
        train_idx, val_idx = train_test_split(
            idx,
            test_size=holdout,
            random_state=rs,
            stratify=y,
        )
        yield CVSplit(train_idx, val_idx, fold_id=1, repeat_id=0, scheme=scheme)
        return

    n_splits = int(
        cfg.validation.get("n_splits", cfg.experiment.get("n_splits", 5))
    )

    if scheme == "kfold":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(idx, y), start=1):
            yield CVSplit(train_idx, val_idx, fold_id=fold_id, repeat_id=0, scheme=scheme)
        return

    if scheme == "repeated_kfold":
        n_repeats = int(cfg.validation.get("n_repeats", 10))
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=rs,
        )
        for i, (train_idx, val_idx) in enumerate(rskf.split(idx, y), start=1):
            repeat_id = (i - 1) // n_splits + 1
            fold_id = (i - 1) % n_splits + 1
            yield CVSplit(
                train_idx, val_idx,
                fold_id=fold_id,
                repeat_id=repeat_id,
                scheme=scheme,
            )
        return

    raise ValueError(f"Unknown CV scheme: {scheme!r}")


def evaluate_cv(
    model_factory: Any,
    df: pd.DataFrame,
    y: pd.Series,
    cfg: DictConfig,
    scheme: str,
    *,
    build_fold_fn: Any,
    feature_mode: str,
    feature_kwargs: dict[str, Any],
    clone_model: Any,
) -> CVMetrics:
    fold_scores: list[float] = []
    fold_pred_rates: list[float] = []
    tgt_rate = target_survival_rate(y, cfg)

    for split in iter_cv_splits(y, cfg, scheme):
        fold = build_fold_fn(
            df, split.train_idx, split.val_idx, feature_mode, **feature_kwargs
        )
        model = clone_model(model_factory)
        model.fit(fold.X_train, fold.y_train)
        pred = model.predict(fold.X_val)
        fold_scores.append(float(accuracy_score(fold.y_val, pred)))
        fold_pred_rates.append(float(np.mean(pred)))

    n = len(fold_scores)
    acc_std = float(np.std(fold_scores, ddof=1)) if n > 1 else 0.0
    pred_std = float(np.std(fold_pred_rates, ddof=1)) if n > 1 else 0.0
    pred_mean = float(np.mean(fold_pred_rates))
    cal_err = abs(pred_mean - tgt_rate)

    return CVMetrics(
        accuracy_mean=float(np.mean(fold_scores)),
        accuracy_std=acc_std,
        fold_scores=fold_scores,
        pred_positive_rate_mean=pred_mean,
        pred_positive_rate_std=pred_std,
        fold_pred_rates=fold_pred_rates,
        target_rate=tgt_rate,
        calibration_error=cal_err,
        n_evaluations=n,
        scheme=scheme,
    )


def calibration_status(cal_error: float, cfg: DictConfig) -> str:
    tol = float(cfg.validation.get("calibration_tolerance", 0.05))
    if cal_error <= tol:
        return "ok"
    if cal_error <= tol * 2:
        return "warn"
    return "poor"


# CV helpers


def _feature_kwargs(cfg: DictConfig) -> dict[str, Any]:
    f = cfg.features
    return {
        "scale": f.scale,
        "drop_constant": f.drop_constant,
        "drop_correlated": f.drop_correlated,
        "correlated_threshold": f.correlated_threshold,
        "clip_outliers": f.clip_outliers,
        "outlier_iqr": f.outlier_iqr,
    }


def _cv_metrics_to_result(
    metrics: CVMetrics,
    name: str,
    params: str,
    stage: str,
) -> ModelResult:
    n_splits = metrics.n_evaluations
    return ModelResult(
        name=name,
        params=params,
        n_splits=n_splits,
        val_acc_mean=metrics.accuracy_mean,
        val_acc_std=metrics.accuracy_std,
        fold_scores=metrics.fold_scores,
        stage=stage,
        pred_positive_rate_mean=metrics.pred_positive_rate_mean,
        target_rate=metrics.target_rate,
        calibration_error=metrics.calibration_error,
        cv_scheme=metrics.scheme,
    )


def cross_validate_model(
    model: Any,
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    n_splits: int | None = None,
    cv_scheme: str | None = None,
    name: str = "model",
    params: str = "",
    stage: str = "cv",
) -> ModelResult:
    builder = FeatureBuilder()
    y = df[builder.cfg.target_col]
    mode = cfg.features.mode
    fkw = _feature_kwargs(cfg)

    if cv_scheme is None:
        if n_splits == 1:
            cv_scheme = "holdout"
        else:
            cv_scheme = "kfold"

    cfg_cv = cfg
    if n_splits is not None and cv_scheme == "kfold":
        cfg_cv = OmegaConf.merge(cfg, {"validation": {"n_splits": n_splits}})

    metrics = evaluate_cv(
        model,
        df,
        y,
        cfg_cv,
        cv_scheme,
        build_fold_fn=builder.build_fold,
        feature_mode=mode,
        feature_kwargs=fkw,
        clone_model=clone,
    )
    return _cv_metrics_to_result(metrics, name, params, stage)


def _print_cv_metrics(metrics: CVMetrics, cfg: DictConfig, label: str) -> None:
    status = calibration_status(metrics.calibration_error, cfg)
    print(
        f"  {label}: acc={metrics.accuracy_mean:.4f} +/- {metrics.accuracy_std:.4f} "
        f"({metrics.n_evaluations} evals) | "
        f"pred_rate={metrics.pred_positive_rate_mean:.4f} "
        f"(target {metrics.target_rate:.4f}) | "
        f"cal_err={metrics.calibration_error:.4f} [{status}]"
    )


def run_validation_suite(cfg: DictConfig, tracker: ResultsTracker) -> dict[str, Any]:
    """hold-out vs k-fold vs repeated k-fold + calibration."""
    if not cfg.validation.get("enabled", True):
        return {}

    builder = FeatureBuilder()
    df = builder.read_raw(cfg.paths.train_csv)
    y = df[builder.cfg.target_col]
    mode = cfg.submission.get("feature_mode") or cfg.features.mode
    cfg_eval = OmegaConf.merge(cfg, {"features": {"mode": mode}})
    fkw = _feature_kwargs(cfg_eval)
    from ml.create_submission import build_submission_model

    model = build_submission_model(cfg_eval)

    log_cfg = cfg_eval.submission.get("logistic") or {}
    model_label = (
        f"{cfg_eval.submission.model} C={log_cfg.get('C', '—')} | features={mode}"
    )

    print(f"\n=== Validation suite: {model_label} ===")
    print(f"  Train survival rate: {target_survival_rate(y, cfg):.4f}")

    schemes = list(cfg.validation.get("schemes", ["holdout", "kfold", "repeated_kfold"]))
    if cfg.experiment.get("quick", False):
        schemes = [s for s in schemes if s != "repeated_kfold"]

    report: dict[str, Any] = {
        "model": model_label,
        "target_rate": float(target_survival_rate(y, cfg)),
        "schemes": {},
    }
    metrics_by_scheme: dict[str, CVMetrics] = {}

    for scheme in schemes:
        cfg_scheme = cfg_eval
        if scheme == "repeated_kfold" and cfg.experiment.get("quick", False):
            cfg_scheme = OmegaConf.merge(cfg_eval, {"validation": {"n_repeats": 2}})

        metrics = evaluate_cv(
            model,
            df,
            y,
            cfg_scheme,
            scheme,
            build_fold_fn=builder.build_fold,
            feature_mode=mode,
            feature_kwargs=fkw,
            clone_model=clone,
        )
        metrics_by_scheme[scheme] = metrics
        _print_cv_metrics(metrics, cfg, scheme)

        tracker.add(
            _cv_metrics_to_result(
                metrics,
                f"validation_{scheme}",
                model_label,
                stage="validation",
            ),
            cv_scheme=scheme,
            calibration_status=calibration_status(metrics.calibration_error, cfg),
        )

        report["schemes"][scheme] = {
            "accuracy_mean": metrics.accuracy_mean,
            "accuracy_std": metrics.accuracy_std,
            "n_evaluations": metrics.n_evaluations,
            "pred_positive_rate_mean": metrics.pred_positive_rate_mean,
            "calibration_error": metrics.calibration_error,
            "calibration_status": calibration_status(metrics.calibration_error, cfg),
            "fold_scores": metrics.fold_scores,
        }

    if "holdout" in metrics_by_scheme and "kfold" in metrics_by_scheme:
        h = metrics_by_scheme["holdout"]
        k = metrics_by_scheme["kfold"]
        gap = k.accuracy_mean - h.accuracy_mean
        report["holdout_vs_kfold_gap"] = gap
        print(f"\n  Hold-out vs 5-fold gap (acc): {gap:+.4f}")
        if abs(gap) > 0.03:
            print("  [!] Razryv >3% — vozmozhen optimistichnyj/pessimistichnyj CV.")
        else:
            print("  [ok] Razryv v predelah 3% — CV stabilen.")

    if "repeated_kfold" in metrics_by_scheme and "kfold" in metrics_by_scheme:
        r = metrics_by_scheme["repeated_kfold"]
        k = metrics_by_scheme["kfold"]
        gap = r.accuracy_mean - k.accuracy_mean
        report["repeated_vs_kfold_gap"] = gap
        print(f"  Repeated ({r.n_evaluations}) vs 5-fold gap: {gap:+.4f}")

    if cfg.validation.get("save_report", True):
        out = Path(cfg.validation.get("report_path", "outputs/validation_report.json"))
        if not out.is_absolute():
            out = Path(__file__).resolve().parent / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Validation report -> {out}")

    return report


# Preprocess ablation


def _probe_model(cfg: DictConfig) -> LogisticRegression:
    return LogisticRegression(
        l1_ratio=0.0, C=1.0, max_iter=2000, random_state=int(cfg.experiment.random_state)
    )


def run_preprocess_ablation(cfg: DictConfig, tracker: ResultsTracker) -> None:
    if not cfg.preprocess_ablation.enabled:
        return
    builder = FeatureBuilder()
    df = builder.read_raw(cfg.paths.train_csv)
    y = df[builder.cfg.target_col]
    rs = int(cfg.experiment.random_state)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    train_idx, val_idx = next(skf.split(df, y))

    print("\n=== Preprocess ablation (probe: logistic L2) ===")
    prev_acc: float | None = None
    for step_name, fold in builder.preprocess_ablation_steps(
        df, train_idx, val_idx, cfg.features.mode
    ):
        if not np.isfinite(fold.X_train).all() or not np.isfinite(fold.X_val).all():
            n_nan = int(np.isnan(fold.X_train).sum() + np.isnan(fold.X_val).sum())
            print(f"  {step_name}: skip fit (NaN in features: {n_nan})")
            continue
        m = _probe_model(cfg)
        m.fit(fold.X_train, fold.y_train)
        acc = float(accuracy_score(fold.y_val, m.predict(fold.X_val)))
        delta = "" if prev_acc is None else f" (delta {acc - prev_acc:+.4f})"
        print(f"  {step_name}: acc={acc:.4f}{delta}")
        tracker.add(
            ModelResult(
                f"ablation_{step_name}",
                "logistic_l2",
                1,
                acc,
                0.0,
                [acc],
                stage="preprocess_ablation",
            ),
            step=step_name,
        )
        prev_acc = acc


# Model factories & grids


def iter_models(cfg: DictConfig) -> Iterator[tuple[str, str, Any]]:
    """(display_name, params_str, estimator)."""
    rs = _rs(cfg)
    enabled = set(cfg.models.enabled)
    quick = bool(cfg.experiment.get("quick", False))

    if "logistic" in enabled:
        yield "logistic", "default", LogisticRegression(max_iter=2000, random_state=rs)

    if "logistic_l1" in enabled:
        grid = cfg.models.logistic_c_grid
        if quick:
            grid = [1.0]
        for c in grid:
            yield (
                "logistic_l1",
                f"C={c}",
                LogisticRegression(
                    l1_ratio=1.0, solver="liblinear", C=float(c),
                    max_iter=2000, random_state=rs,
                ),
            )

    if "logistic_l2" in enabled:
        grid = cfg.models.logistic_c_grid
        if quick:
            grid = [1.0]
        for c in grid:
            yield (
                "logistic_l2",
                f"C={c}",
                LogisticRegression(
                    l1_ratio=0.0, C=float(c), max_iter=2000, random_state=rs,
                ),
            )

    if "logistic_elasticnet" in enabled:
        c_grid = cfg.models.logistic_c_grid
        l1_grid = cfg.models.elasticnet_l1_ratio_grid
        if quick:
            c_grid, l1_grid = [1.0], [0.5]
        for c, l1 in product(c_grid, l1_grid):
            yield (
                "logistic_elasticnet",
                f"C={c},l1_ratio={l1}",
                LogisticRegression(
                    solver="saga",
                    C=float(c),
                    l1_ratio=float(l1),
                    max_iter=3000,
                    random_state=rs,
                ),
            )

    if "knn" in enabled:
        knn = cfg.models.knn
        n_vals = knn.n_neighbors[:3] if quick else knn.n_neighbors
        w_vals = knn.weights
        m_vals = knn.metrics[:1] if quick else knn.metrics
        for n, w, metric in product(n_vals, w_vals, m_vals):
            yield (
                "knn",
                f"n={n},w={w},metric={metric}",
                KNeighborsClassifier(n_neighbors=int(n), weights=w, metric=metric),
            )

    if "decision_tree" in enabled:
        dt = cfg.models.decision_tree
        depths = dt.max_depth[:2] if quick else dt.max_depth
        leaves = dt.min_samples_leaf[:2] if quick else dt.min_samples_leaf
        for depth, leaf in product(depths, leaves):
            yield (
                "decision_tree",
                f"depth={depth},leaf={leaf}",
                DecisionTreeClassifier(
                    max_depth=None if depth is None else int(depth),
                    min_samples_leaf=int(leaf),
                    random_state=rs,
                ),
            )

    if "random_forest" in enabled:
        rf = cfg.models.random_forest
        est = rf.n_estimators[:1] if quick else rf.n_estimators
        depths = rf.max_depth[:2] if quick else rf.max_depth
        leaves = rf.min_samples_leaf[:2] if quick else rf.min_samples_leaf
        feats = rf.max_features[:1] if quick else rf.max_features
        for n_est, depth, leaf, mf in product(est, depths, leaves, feats):
            yield (
                "random_forest",
                f"n={n_est},d={depth},leaf={leaf},mf={mf}",
                RandomForestClassifier(
                    n_estimators=int(n_est),
                    max_depth=None if depth is None else int(depth),
                    min_samples_leaf=int(leaf),
                    max_features=mf,
                    random_state=rs,
                    n_jobs=-1,
                ),
            )

    if "catboost" in enabled:
        from catboost import CatBoostClassifier

        cb = cfg.models.catboost
        iters = cb.iterations[:1] if quick else cb.iterations
        depths = cb.depth[:1] if quick else cb.depth
        lrs = cb.learning_rate[:1] if quick else cb.learning_rate
        regs = cb.l2_leaf_reg[:1] if quick else cb.l2_leaf_reg
        for it, d, lr, reg in product(iters, depths, lrs, regs):
            yield (
                "catboost",
                f"iter={it},d={d},lr={lr},l2={reg}",
                CatBoostClassifier(
                    iterations=int(it),
                    depth=int(d),
                    learning_rate=float(lr),
                    l2_leaf_reg=float(reg),
                    verbose=0,
                    random_seed=rs,
                    allow_writing_files=False,
                ),
            )

    if "lightgbm" in enabled:
        import lightgbm as lgb

        lg = cfg.models.lightgbm
        est = lg.n_estimators[:1] if quick else lg.n_estimators
        leaves = lg.num_leaves[:1] if quick else lg.num_leaves
        lrs = lg.learning_rate[:1] if quick else lg.learning_rate
        mcs = lg.min_child_samples[:1] if quick else lg.min_child_samples
        for n_est, nl, lr, mc in product(est, leaves, lrs, mcs):
            yield (
                "lightgbm",
                f"n={n_est},leaves={nl},lr={lr},mc={mc}",
                lgb.LGBMClassifier(
                    n_estimators=int(n_est),
                    num_leaves=int(nl),
                    learning_rate=float(lr),
                    min_child_samples=int(mc),
                    random_state=rs,
                    verbosity=-1,
                    n_jobs=-1,
                ),
            )

    if "xgboost" in enabled:
        import xgboost as xgb

        xg = cfg.models.xgboost
        est = xg.n_estimators[:1] if quick else xg.n_estimators
        depths = xg.max_depth[:1] if quick else xg.max_depth
        lrs = xg.learning_rate[:1] if quick else xg.learning_rate
        subs = xg.subsample[:1] if quick else xg.subsample
        for n_est, d, lr, sub in product(est, depths, lrs, subs):
            yield (
                "xgboost",
                f"n={n_est},d={d},lr={lr},sub={sub}",
                xgb.XGBClassifier(
                    n_estimators=int(n_est),
                    max_depth=int(d),
                    learning_rate=float(lr),
                    subsample=float(sub),
                    random_state=rs,
                    verbosity=0,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
            )


def run_all_models(cfg: DictConfig, tracker: ResultsTracker) -> list[ModelResult]:
    df = FeatureBuilder().read_raw(cfg.paths.train_csv)
    n_splits = int(cfg.experiment.n_splits)
    results: list[ModelResult] = []

    print(f"\n=== Models CV ({n_splits}-fold) ===")
    for name, params, model in iter_models(cfg):
        res = cross_validate_model(
            model, df, cfg, n_splits=n_splits,
            name=name, params=params,
        )
        results.append(res)
        tracker.add(res)
        print(
            f"  {res.val_acc_mean:.4f} +/- {res.val_acc_std:.4f} | "
            f"{name} | {params}"
        )
    return results


# Ensembles: diverse voting + rule blend


def build_shallow_rf(cfg: DictConfig) -> RandomForestClassifier:
    rf = cfg.ensemble.diverse.random_forest
    return RandomForestClassifier(
        n_estimators=int(rf.get("n_estimators", 200)),
        max_depth=int(rf.get("max_depth", 5)),
        min_samples_leaf=int(rf.get("min_samples_leaf", 5)),
        max_features=rf.get("max_features", "sqrt"),
        random_state=_rs(cfg),
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
        random_seed=_rs(cfg),
        allow_writing_files=False,
    )


def build_diverse_estimators(cfg: DictConfig) -> list[tuple[str, Any]]:
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
        random_state=_rs(cfg),
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



def _parse_params(params: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(params).split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _nullable_int(value: str | None) -> int | None:
    if value is None or value in {"null", "None", ""}:
        return None
    return int(value)


def _build_estimator_from_row(row: dict, cfg: DictConfig) -> tuple[str, Any]:
    """Rebuild estimator from CV/tune results row (name + params)."""
    rs = _rs(cfg)
    name = str(row["name"])
    params = str(row.get("params", ""))

    if name.startswith("tuned_"):
        from ml.tune import build_model_from_params

        model_name = name.removeprefix("tuned_")
        best_params = json.loads(params)
        return name, build_model_from_params(model_name, best_params, cfg)

    p = _parse_params(params)

    if name == "random_forest":
        return name, RandomForestClassifier(
            n_estimators=int(p.get("n", 100)),
            max_depth=_nullable_int(p.get("d")),
            min_samples_leaf=int(p.get("leaf", 1)),
            max_features=p.get("mf", "sqrt"),
            random_state=rs,
            n_jobs=-1,
        )
    if name.startswith("logistic"):
        if name == "logistic_l1":
            return name, LogisticRegression(
                l1_ratio=1.0, solver="liblinear", C=float(p.get("C", 1.0)),
                max_iter=2000, random_state=rs,
            )
        if name == "logistic_elasticnet":
            return name, LogisticRegression(
                solver="saga",
                C=float(p.get("C", 1.0)),
                l1_ratio=float(p.get("l1_ratio", 0.5)),
                max_iter=3000,
                random_state=rs,
            )
        return name, LogisticRegression(
            l1_ratio=0.0, C=float(p.get("C", 1.0)), max_iter=2000, random_state=rs
        )
    if name == "knn":
        return name, KNeighborsClassifier(
            n_neighbors=int(p.get("n", 5)),
            weights=p.get("w", "uniform"),
            metric=p.get("metric", "euclidean"),
        )
    if name == "decision_tree":
        return name, DecisionTreeClassifier(
            max_depth=_nullable_int(p.get("depth")),
            min_samples_leaf=int(p.get("leaf", 1)),
            random_state=rs,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return name, CatBoostClassifier(
            iterations=int(p.get("iter", 300)),
            depth=int(p.get("d", 4)),
            learning_rate=float(p.get("lr", 0.05)),
            l2_leaf_reg=float(p.get("l2", 1.0)),
            verbose=0,
            random_seed=rs,
            allow_writing_files=False,
        )
    if name == "lightgbm":
        import lightgbm as lgb

        return name, lgb.LGBMClassifier(
            n_estimators=int(p.get("n", 200)),
            num_leaves=int(p.get("leaves", 31)),
            learning_rate=float(p.get("lr", 0.05)),
            min_child_samples=int(p.get("mc", 5)),
            random_state=rs,
            verbosity=-1,
            n_jobs=-1,
        )
    if name == "xgboost":
        import xgboost as xgb

        return name, xgb.XGBClassifier(
            n_estimators=int(p.get("n", 200)),
            max_depth=int(p.get("d", 3)),
            learning_rate=float(p.get("lr", 0.05)),
            subsample=float(p.get("sub", 0.8)),
            random_state=rs,
            verbosity=0,
            n_jobs=-1,
            eval_metric="logloss",
        )
    return name, LogisticRegression(max_iter=2000, random_state=rs)


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
            name,
            params,
            len(fold_scores),
            acc,
            std,
            fold_scores,
            stage="ensemble",
        ),
        **extra,
    )


def run_ensembles(cfg: DictConfig, tracker: ResultsTracker) -> None:
    if not cfg.ensemble.enabled:
        return
    df = FeatureBuilder().read_raw(cfg.paths.train_csv)
    y = df[FeatureBuilder().cfg.target_col].values
    fkw = _feature_kwargs(cfg)
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
        print(f"  diverse_voting (soft LR+RF+CatBoost) CV: {acc:.4f}")
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
        _add_ensemble_result(
            tracker, "ensemble_rule_blend", f"ml_weight={ml_w}", blend_acc, folds
        )

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


# Orchestration


def run_pipeline(cfg: DictConfig, stage: str) -> None:
    tracker = ResultsTracker(cfg.paths.results_dir)

    if stage in ("preprocess", "all"):
        run_preprocess_ablation(cfg, tracker)

    if stage in ("validation", "all"):
        run_validation_suite(cfg, tracker)

    if stage in ("models", "all"):
        run_all_models(cfg, tracker)

    if stage in ("tune", "all") and cfg.tune.enabled:
        from ml.tune import run_optuna_studies

        tune_results = run_optuna_studies(
            cfg,
            use_saved=bool(cfg.tune.get("use_saved", True)),
        )
        for row in tune_results:
            tracker.add(
                ModelResult(
                    f"tuned_{row['model']}",
                    json.dumps(row["best_params"]),
                    int(cfg.tune.cv_folds),
                    row["best_value"],
                    0.0,
                    [row["best_value"]],
                    stage="tune",
                ),
            )

    if stage in ("ensemble", "all"):
        run_ensembles(cfg, tracker)

    if tracker.rows:
        tracker.save()
        tracker.leaderboard()

    if stage in ("submit", "all"):
        from ml.create_submission import create_submission

        create_submission(cfg, tracker)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Titanic ML pipeline")
    p.add_argument(
        "--stage",
        default="all",
        choices=[
            "preprocess", "validation", "models",
            "tune", "ensemble", "submit", "all",
        ],
    )
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--quick", action="store_true", help="Smaller grids")
    return p.parse_args()


if __name__ == "__main__":
    from config import load_config

    args = parse_args()
    config = load_config(args.config)
    if args.quick:
        config.experiment.quick = True
        config.tune.n_trials = 5
        if "validation" in config:
            config.validation.n_repeats = 2
    run_pipeline(config, args.stage)
