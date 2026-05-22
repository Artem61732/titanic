"""
Titanic Kaggle — классический ML пайплайн.

Запуск из папки ml:
  python main.py --stage all
  python main.py --stage eda
  python main.py --stage models
  python main.py --stage tune
  python main.py --stage submit
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from feature_engineering import FeatureBuilder
from tune import build_model_from_params, run_optuna_studies

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config & results
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path | None = None) -> DictConfig:
    root = Path(__file__).resolve().parent
    path = Path(config_path) if config_path else root / "config.yaml"
    cfg = OmegaConf.load(path)
    for key in ("paths",):
        if key in cfg and hasattr(cfg[key], "items"):
            for k, v in cfg[key].items():
                p = Path(str(v))
                if not p.is_absolute():
                    cfg[key][k] = str((root / p).resolve())
    OmegaConf.resolve(cfg)
    return cfg


@dataclass
class ModelResult:
    name: str
    params: str
    n_splits: int
    val_acc_mean: float
    val_acc_std: float
    fold_scores: list[float]
    stage: str = "cv"


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


# ---------------------------------------------------------------------------
# EDA (п.1)
# ---------------------------------------------------------------------------


def run_eda(cfg: DictConfig) -> None:
    builder = FeatureBuilder()
    df = builder.read_raw(cfg.paths.train_csv)
    eda_dir = Path(cfg.paths.eda_dir)
    eda_dir.mkdir(parents=True, exist_ok=True)

    print("=== EDA: basic stats ===")
    print(df.describe(include="all").T.head(20))
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nTarget distribution:")
    print(df["Survived"].value_counts(normalize=True))

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(5, 4))
    df["Survived"].value_counts().plot(kind="bar", ax=ax, color=["#c44e52", "#55a868"])
    ax.set_title("Survived distribution")
    ax.set_xlabel("Survived")
    fig.tight_layout()
    fig.savefig(eda_dir / "target_distribution.png", dpi=120)
    plt.close(fig)

    num_cols = ["Age", "Fare", "SibSp", "Parch"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), num_cols):
        df[col].dropna().hist(ax=ax, bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(col)
    fig.suptitle("Numeric feature distributions")
    fig.tight_layout()
    fig.savefig(eda_dir / "numeric_histograms.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Sex", hue="Survived", ax=ax)
    ax.set_title("Survival by Sex")
    fig.tight_layout()
    fig.savefig(eda_dir / "survival_by_sex.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax)
    ax.set_title("Survival by Pclass")
    fig.tight_layout()
    fig.savefig(eda_dir / "survival_by_pclass.png", dpi=120)
    plt.close(fig)

    feat = builder.featurize(df)
    corr = feat.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation (engineered numeric)")
    fig.tight_layout()
    fig.savefig(eda_dir / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"EDA plots -> {eda_dir}")


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------


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


def cross_validate_model(
    model: Any,
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    n_splits: int | None = None,
    name: str = "model",
    params: str = "",
    stage: str = "cv",
) -> ModelResult:
    builder = FeatureBuilder()
    y = df[builder.cfg.target_col]
    n_splits = n_splits or int(cfg.experiment.n_splits)
    rs = int(cfg.experiment.random_state)
    mode = cfg.features.mode
    fkw = _feature_kwargs(cfg)

    fold_scores: list[float] = []

    if n_splits == 1:
        idx = np.arange(len(df))
        train_idx, val_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=rs,
            stratify=y,
        )
        splits = [(train_idx, val_idx)]
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        splits = list(skf.split(df, y))

    for train_idx, val_idx in splits:
        fold = builder.build_fold(df, train_idx, val_idx, mode, **fkw)
        m = clone(model)
        m.fit(fold.X_train, fold.y_train)
        pred = m.predict(fold.X_val)
        fold_scores.append(float(accuracy_score(fold.y_val, pred)))

    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    return ModelResult(name, params, n_splits, mean, std, fold_scores, stage)


def compare_folds(cfg: DictConfig, model: Any, name: str, tracker: ResultsTracker) -> None:
    for n in cfg.experiment.compare_n_splits:
        res = cross_validate_model(
            model, FeatureBuilder().read_raw(cfg.paths.train_csv), cfg,
            n_splits=int(n), name=name, params="default",
            stage="fold_compare",
        )
        tracker.add(res, experiment="fold_compare")
        print(f"  {n}-fold: {res.val_acc_mean:.4f} +/- {res.val_acc_std:.4f}")


# ---------------------------------------------------------------------------
# Preprocess ablation (п.2 — сравнение до/после)
# ---------------------------------------------------------------------------


def _probe_model(cfg: DictConfig) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, random_state=int(cfg.experiment.random_state)
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


# ---------------------------------------------------------------------------
# Model factories & grids (п.4)
# ---------------------------------------------------------------------------


def _rs(cfg: DictConfig) -> int:
    return int(cfg.experiment.random_state)


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
                    penalty="l1", solver="liblinear", C=float(c),
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
                    penalty="l2", C=float(c), max_iter=2000, random_state=rs,
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
                    penalty="elasticnet",
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


# ---------------------------------------------------------------------------
# Ensembles (п.6)
# ---------------------------------------------------------------------------


def _build_estimator_from_row(row: dict, cfg: DictConfig) -> tuple[str, Any]:
    """Восстановление лучших sklearn-моделей по имени (упрощённо)."""
    rs = _rs(cfg)
    name = row["name"]
    if name == "random_forest":
        return name, RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=rs, n_jobs=-1
        )
    if name.startswith("logistic"):
        return name, LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=rs)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return name, CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05,
            verbose=0, random_seed=rs, allow_writing_files=False,
        )
    if name == "lightgbm":
        import lightgbm as lgb
        return name, lgb.LGBMClassifier(
            n_estimators=400, num_leaves=31, random_state=rs, verbosity=-1
        )
    if name == "xgboost":
        import xgboost as xgb
        return name, xgb.XGBClassifier(
            n_estimators=400, max_depth=5, random_state=rs, verbosity=0
        )
    return name, LogisticRegression(max_iter=2000, random_state=rs)


def _oof_predict_proba(
    model: Any,
    df: pd.DataFrame,
    cfg: DictConfig,
) -> np.ndarray:
    builder = FeatureBuilder()
    y = df[builder.cfg.target_col]
    n_splits = int(cfg.experiment.n_splits)
    rs = int(cfg.experiment.random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    oof = np.zeros(len(df))
    fkw = _feature_kwargs(cfg)
    mode = cfg.features.mode

    for train_idx, val_idx in skf.split(df, y):
        fold = builder.build_fold(df, train_idx, val_idx, mode, **fkw)
        m = clone(model)
        m.fit(fold.X_train, fold.y_train)
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(fold.X_val)[:, 1]
        else:
            proba = m.decision_function(fold.X_val)
            proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
        oof[val_idx] = proba
    return oof


def run_ensembles(cfg: DictConfig, tracker: ResultsTracker) -> None:
    if not cfg.ensemble.enabled:
        return
    df = FeatureBuilder().read_raw(cfg.paths.train_csv)
    y = df[FeatureBuilder().cfg.target_col].values
    rs = _rs(cfg)

    lb = tracker.leaderboard(top=int(cfg.ensemble.top_k))
    if lb.empty:
        print("No CV results for ensembles — skip.")
        return

    estimators: list[tuple[str, Any]] = []
    for _, row in lb.iterrows():
        name, est = _build_estimator_from_row(row.to_dict(), cfg)
        estimators.append((f"{name}_{len(estimators)}", est))

    print(f"\n=== Ensembles ({len(estimators)} base models) ===")
    methods = list(cfg.ensemble.methods)

    if "average" in methods:
        probs = np.column_stack(
            [_oof_predict_proba(est, df, cfg) for _, est in estimators]
        )
        oof_pred = (probs.mean(axis=1) >= 0.5).astype(int)
        acc = float(accuracy_score(y, oof_pred))
        print(f"  average OOF acc: {acc:.4f}")
        tracker.add(
            ModelResult("ensemble_average", str(len(estimators)), 5, acc, 0.0, [acc],
                        stage="ensemble"),
        )

    if "voting" in methods:
        vote = VotingClassifier(estimators=estimators, voting="soft")
        res = cross_validate_model(
            vote, df, cfg, name="ensemble_voting", params="soft", stage="ensemble"
        )
        tracker.add(res)
        print(f"  voting CV: {res.val_acc_mean:.4f}")

    if "stacking_lr" in methods:
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, random_state=rs),
            cv=5,
            passthrough=False,
        )
        res = cross_validate_model(
            stack, df, cfg, name="ensemble_stacking_lr", params="5-fold",
            stage="ensemble",
        )
        tracker.add(res)
        print(f"  stacking+LR CV: {res.val_acc_mean:.4f}")

    if "stacking_ridge" in methods:
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=RidgeClassifier(),
            cv=5,
            passthrough=False,
        )
        res = cross_validate_model(
            stack, df, cfg, name="ensemble_stacking_ridge", params="5-fold",
            stage="ensemble",
        )
        tracker.add(res)
        print(f"  stacking+Ridge CV: {res.val_acc_mean:.4f}")


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def _best_row(tracker: ResultsTracker, prefer: str) -> dict | None:
    if not tracker.rows:
        return None
    df = pd.DataFrame(tracker.rows)
    if prefer != "best_cv":
        sub = df[df["name"] == prefer]
        if not sub.empty:
            return sub.sort_values("val_acc_mean", ascending=False).iloc[0].to_dict()
    cv = df[df["stage"] == "cv"]
    if cv.empty:
        cv = df
    return cv.sort_values("val_acc_mean", ascending=False).iloc[0].to_dict()


def _model_from_tuned(cfg: DictConfig) -> Any | None:
    summary = Path(cfg.paths.tune_dir) / "tune_summary.json"
    if not summary.exists():
        return None
    with summary.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        return None
    best = max(rows, key=lambda r: r["best_value"])
    return build_model_from_params(best["model"], best["best_params"], cfg)


def _build_logistic_from_submission_cfg(cfg: DictConfig) -> LogisticRegression:
    """LogisticRegression для submission по config.submission.logistic."""
    rs = _rs(cfg)
    log_cfg = cfg.submission.get("logistic") or {}
    penalty = str(log_cfg.get("penalty", "l2")).lower()
    c = float(log_cfg.get("C", 0.1))
    max_iter = int(log_cfg.get("max_iter", 2000))

    if penalty == "l1":
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=c,
            max_iter=max_iter,
            random_state=rs,
        )
    if penalty == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=c,
            l1_ratio=float(log_cfg.get("l1_ratio", 0.5)),
            max_iter=max_iter,
            random_state=rs,
        )
    # l2: sklearn 1.8+ — без penalty=, через C и l1_ratio=0
    return LogisticRegression(C=c, l1_ratio=0.0, max_iter=max_iter, random_state=rs)


def build_submission_model(cfg: DictConfig, tracker: ResultsTracker | None = None) -> Any:
    prefer = str(cfg.submission.model)
    if prefer == "tuned":
        tuned = _model_from_tuned(cfg)
        if tuned is not None:
            return tuned
        print("No tuned model found — fallback to submission.logistic.")
        return _build_logistic_from_submission_cfg(cfg)
    if prefer in ("logistic_l2", "logistic_l1", "logistic", "logistic_elasticnet"):
        return _build_logistic_from_submission_cfg(cfg)
    if tracker is not None:
        row = _best_row(tracker, prefer)
        if row is not None:
            _, model = _build_estimator_from_row(row, cfg)
            return model
    return _build_logistic_from_submission_cfg(cfg)


def create_submission(cfg: DictConfig, tracker: ResultsTracker) -> Path:
    builder = FeatureBuilder()
    train_df = builder.read_raw(cfg.paths.train_csv)
    test_df = builder.read_raw(cfg.paths.test_csv)
    mode = cfg.submission.get("feature_mode") or cfg.features.mode
    fkw = _feature_kwargs(cfg)

    model = build_submission_model(cfg, tracker)
    log = cfg.submission.get("logistic") or {}
    print(
        f"Submission model: {cfg.submission.model} | "
        f"features={mode} | C={log.get('C', '—')}"
    )

    matrices = builder.build_train_test(train_df, test_df, mode, **fkw)
    model.fit(matrices.X_train, matrices.y_train)
    preds = model.predict(matrices.X_test).astype(int)
    train_acc = float(accuracy_score(matrices.y_train, model.predict(matrices.X_train)))

    sub = pd.DataFrame(
        {
            "PassengerId": matrices.test_passenger_ids,
            "Survived": preds,
        }
    )
    out = Path(cfg.paths.submission_csv)
    sub.to_csv(out, index=False)
    print(
        f"Submission: {out} ({len(sub)} rows) | "
        f"train acc {train_acc:.4f} | pred rate {preds.mean():.4f}"
    )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_fold_comparison(cfg: DictConfig, tracker: ResultsTracker) -> None:
    print("\n=== Validation: 1-fold vs 5-fold (Logistic L2) ===")
    model = LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, random_state=_rs(cfg)
    )
    compare_folds(cfg, model, "logistic_l2_fold_compare", tracker)


def run_feature_mode_comparison(cfg: DictConfig, tracker: ResultsTracker) -> None:
    print("\n=== Encoding comparison (Logistic L2, 5-fold) ===")
    df = FeatureBuilder().read_raw(cfg.paths.train_csv)
    for mode in ("baseline", "onehot", "label"):
        cfg_mode = OmegaConf.merge(cfg, {"features": {"mode": mode}})
        res = cross_validate_model(
            LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=_rs(cfg)),
            df,
            cfg_mode,
            name=f"encoding_{mode}",
            params="logistic_l2",
            stage="encoding_compare",
        )
        tracker.add(res, feature_mode=mode)
        print(f"  {mode}: {res.val_acc_mean:.4f} +/- {res.val_acc_std:.4f}")


def run_pipeline(cfg: DictConfig, stage: str) -> None:
    tracker = ResultsTracker(cfg.paths.results_dir)

    if stage in ("eda", "all"):
        run_eda(cfg)

    if stage in ("preprocess", "all"):
        run_preprocess_ablation(cfg, tracker)

    if stage in ("validation", "all"):
        run_fold_comparison(cfg, tracker)
        run_feature_mode_comparison(cfg, tracker)

    if stage in ("models", "all"):
        run_all_models(cfg, tracker)

    if stage in ("tune", "all") and cfg.tune.enabled:
        tune_results = run_optuna_studies(cfg)
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
        create_submission(cfg, tracker)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Titanic ML pipeline")
    p.add_argument(
        "--stage",
        default="all",
        choices=[
            "eda", "preprocess", "validation", "models",
            "tune", "ensemble", "submit", "all",
        ],
    )
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--quick", action="store_true", help="Smaller grids")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.quick:
        config.experiment.quick = True
        config.tune.n_trials = 5
    run_pipeline(config, args.stage)
