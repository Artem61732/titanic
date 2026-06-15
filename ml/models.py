"""Model factories and CV grids."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterator

from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml.cv import cross_validate_model
from ml.feature_engineering import FeatureBuilder
from ml.results import ModelResult, ResultsTracker
from ml.utils import random_seed

def iter_models(cfg: DictConfig) -> Iterator[tuple[str, str, Any]]:
    """(display_name, params_str, estimator)."""
    rs = random_seed(cfg)
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

def build_estimator_from_cv_row(row: dict, cfg: DictConfig) -> tuple[str, Any]:
    """Rebuild estimator from a CV results row (name + params)."""
    name = row["name"]
    params = row.get("params", "")
    for model_name, model_params, estimator in iter_models(cfg):
        if model_name == name and model_params == params:
            return model_name, estimator
    rs = random_seed(cfg)
    return name, LogisticRegression(max_iter=2000, random_state=rs, l1_ratio=0.0)


def run_all_models(cfg: DictConfig, tracker: ResultsTracker) -> list[ModelResult]:
    df = FeatureBuilder().read_raw(cfg.paths.train_csv)
    n_splits = int(cfg.experiment.n_splits)
    results: list[ModelResult] = []

    print(f"\n=== Models CV ({n_splits}-fold) ===")
    for name, params, model in iter_models(cfg):
        res = cross_validate_model(
            model, df, cfg, n_splits=n_splits, name=name, params=params
        )
        results.append(res)
        tracker.add(res)
        print(
            f"  {res.val_acc_mean:.4f} +/- {res.val_acc_std:.4f} | {name} | {params}"
        )
    return results
