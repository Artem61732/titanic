"""Генерация submission.csv для Kaggle Titanic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

from ml.feature_engineering import FeatureBuilder
from ml.main import (
    ResultsTracker,
    _build_estimator_from_row,
    _feature_kwargs,
    build_logistic_from_submission_cfg,
    calibration_status,
    fit_predict_test,
    rule_based_predictions,
)

ENSEMBLE_SUBMISSION = {
    "ensemble_diverse_voting": "diverse_voting",
    "ensemble_diverse_mean": "diverse_mean",
    "ensemble_rule_blend": "diverse_rule_blend",
    "ensemble_diverse_rule_blend": "diverse_rule_blend",
    "rule_only": "rule_only",
}


def _best_row(tracker: Any, prefer: str) -> dict | None:
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
    from ml.tune import build_model_from_params

    summary = Path(cfg.paths.tune_dir) / "tune_summary.json"
    if not summary.exists():
        return None
    with summary.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        return None
    best = max(rows, key=lambda r: r["best_value"])
    return build_model_from_params(best["model"], best["best_params"], cfg)


def build_submission_model(cfg: DictConfig, tracker: Any | None = None) -> Any:
    prefer = str(cfg.submission.model)
    if prefer == "tuned":
        tuned = _model_from_tuned(cfg)
        if tuned is not None:
            return tuned
        print("No tuned model found — fallback to submission.logistic.")
        return build_logistic_from_submission_cfg(cfg)
    if prefer in ("logistic_l2", "logistic_l1", "logistic", "logistic_elasticnet"):
        return build_logistic_from_submission_cfg(cfg)
    if tracker is not None:
        row = _best_row(tracker, prefer)
        if row is not None:
            _, model = _build_estimator_from_row(row, cfg)
            return model
    return build_logistic_from_submission_cfg(cfg)


def create_submission(cfg: DictConfig, tracker: Any | None = None) -> Path:
    builder = FeatureBuilder()
    train_df = builder.read_raw(cfg.paths.train_csv)
    test_df = builder.read_raw(cfg.paths.test_csv)
    prefer = str(cfg.submission.model)
    mode = (
        cfg.submission.get("feature_mode")
        or cfg.ensemble.get("feature_mode")
        or cfg.features.mode
    )
    fkw = _feature_kwargs(cfg)
    matrices = builder.build_train_test(train_df, test_df, mode, **fkw)

    if prefer in ENSEMBLE_SUBMISSION:
        method = ENSEMBLE_SUBMISSION[prefer]
        ml_w = float(cfg.ensemble.rule_blend.get("ml_weight", 0.65))
        print(
            f"Submission: {prefer} ({method}) | features={mode} | "
            f"ml_weight={ml_w if 'rule' in method else '—'}"
        )
        preds = fit_predict_test(train_df, test_df, cfg, fkw, method=method)
        if prefer == "rule_only":
            train_pred = rule_based_predictions(train_df, cfg)
        else:
            train_pred = fit_predict_test(train_df, train_df, cfg, fkw, method=method)
        train_acc = float(accuracy_score(matrices.y_train, train_pred))
        passenger_ids = matrices.test_passenger_ids
    elif prefer == "tuned":
        model = build_submission_model(cfg, tracker)
        print(f"Submission: tuned | features={mode}")
        model.fit(matrices.X_train, matrices.y_train)
        preds = model.predict(matrices.X_test).astype(int)
        train_acc = float(accuracy_score(matrices.y_train, model.predict(matrices.X_train)))
        passenger_ids = matrices.test_passenger_ids
    else:
        model = build_submission_model(cfg, tracker)
        log = cfg.submission.get("logistic") or {}
        print(f"Submission: {prefer} | features={mode} | C={log.get('C', '—')}")
        model.fit(matrices.X_train, matrices.y_train)
        preds = model.predict(matrices.X_test).astype(int)
        train_acc = float(accuracy_score(matrices.y_train, model.predict(matrices.X_train)))
        passenger_ids = matrices.test_passenger_ids

    tgt_rate = float(matrices.y_train.mean())
    cal_err = abs(float(preds.mean()) - tgt_rate)
    cal_status = calibration_status(cal_err, cfg)

    sub = pd.DataFrame({"PassengerId": passenger_ids, "Survived": preds})
    out = Path(cfg.paths.submission_csv)
    sub.to_csv(out, index=False)
    print(
        f"Saved: {out} ({len(sub)} rows) | train acc {train_acc:.4f} | "
        f"pred rate {preds.mean():.4f} (target {tgt_rate:.4f}) | "
        f"cal_err {cal_err:.4f} [{cal_status}]"
    )
    return out


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    results = ResultsTracker(config.paths.results_dir)
    create_submission(config, results)
