from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from config import (
    FeatureMode,
    fold_seed,
    load_config,
    resolve_feature_mode,
    set_seed,
)
from dl.feature_engineering import (
    FeatureBuilder,
    TrainTestEmbedding,
    TrainTestOneHot,
)
from dl.train import (
    MatrixDataset,
    TabularDataset,
    TitanicEmbeddingMLP,
    TitanicMLP,
    _train_dataloader,
    make_criterion,
    make_scheduler,
    train_epoch,
    train_fold,
    _resolve_amp_dtype,
    _make_grad_scaler,
    _use_amp,
    _step_scheduler,
)


@torch.no_grad()
def predict_onehot(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        MatrixDataset(X),
        batch_size=batch_size,
        shuffle=False,
    )
    preds: list[np.ndarray] = []
    for x in loader:
        x = x.to(device)
        out = model(x)
        preds.append((torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.int64))
    return np.concatenate(preds, axis=0).ravel()


@torch.no_grad()
def predict_embedding(
    model: TitanicEmbeddingMLP,
    cat: np.ndarray,
    num: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TabularDataset(cat, num),
        batch_size=batch_size,
        shuffle=False,
    )
    preds: list[np.ndarray] = []
    for cat_x, num_x in loader:
        cat_x, num_x = cat_x.to(device), num_x.to(device)
        out = model(cat_x, num_x)
        preds.append((torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.int64))
    return np.concatenate(preds, axis=0).ravel()


def _fit_holdout_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: pd.Series,
    device: torch.device,
    cfg: DictConfig,
) -> int:
    """Early stopping на holdout; возвращает число эпох для финального fit."""
    _, _, epochs_run = train_fold(
        model,
        train_loader,
        val_loader,
        y_train,
        device,
        cfg,
        verbose=False,
    )
    return max(epochs_run, 1)


def _train_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    y_train: pd.Series,
    device: torch.device,
    cfg: DictConfig,
    epochs: int,
) -> None:
    criterion = make_criterion(y_train, cfg.use_class_weights, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = make_scheduler(optimizer, cfg)
    use_amp = _use_amp(cfg, device)
    amp_dtype = _resolve_amp_dtype(cfg)
    scaler = _make_grad_scaler(use_amp)

    for epoch in range(1, epochs + 1):
        train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        if cfg.scheduler == "cosine" and scheduler is not None:
            _step_scheduler(scheduler, 0.0)
        elif cfg.scheduler == "plateau":
            pass  # без val на полном train plateau не шагаем


def train_and_predict_onehot(
    data: TrainTestOneHot,
    device: torch.device,
    cfg: DictConfig,
    holdout_fraction: float,
    random_state: int,
) -> np.ndarray:
    idx = np.arange(len(data.y_train))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=data.y_train,
    )

    scaler = StandardScaler()
    X_all = scaler.fit_transform(data.X_train)
    X_test = scaler.transform(data.X_test)

    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    y_tr = data.y_train.iloc[tr_idx]
    y_va = data.y_train.iloc[va_idx]

    holdout_seed = fold_seed(random_state)
    set_seed(holdout_seed)
    holdout_train = _train_dataloader(
        MatrixDataset(X_tr, y_tr), cfg.batch_size, holdout_seed
    )
    holdout_val = DataLoader(
        MatrixDataset(X_va, y_va), batch_size=cfg.batch_size, shuffle=False
    )

    model = TitanicMLP(X_all.shape[1], dropout=cfg.mlp_dropout).to(device)
    best_epochs = _fit_holdout_epochs(
        model, holdout_train, holdout_val, y_tr, device, cfg
    )

    final_seed = fold_seed(random_state, fold=1)
    set_seed(final_seed)
    final_model = TitanicMLP(X_all.shape[1], dropout=cfg.mlp_dropout).to(device)
    full_loader = _train_dataloader(
        MatrixDataset(X_all, data.y_train), cfg.batch_size, final_seed
    )
    _train_fixed_epochs(
        final_model, full_loader, data.y_train, device, cfg, best_epochs
    )
    return predict_onehot(final_model, X_test, device, cfg.batch_size)


def train_and_predict_embedding(
    data: TrainTestEmbedding,
    device: torch.device,
    cfg: DictConfig,
    holdout_fraction: float,
    random_state: int,
) -> np.ndarray:
    idx = np.arange(len(data.y_train))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=data.y_train,
    )

    scaler = StandardScaler()
    num_all = scaler.fit_transform(data.num_train)
    num_test = scaler.transform(data.num_test)

    holdout_seed = fold_seed(random_state)
    set_seed(holdout_seed)
    holdout_train = _train_dataloader(
        TabularDataset(
            data.cat_train[tr_idx], num_all[tr_idx], data.y_train.iloc[tr_idx]
        ),
        cfg.batch_size,
        holdout_seed,
    )
    holdout_val = DataLoader(
        TabularDataset(data.cat_train[va_idx], num_all[va_idx], data.y_train.iloc[va_idx]),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    n_cont = num_all.shape[1]
    model = TitanicEmbeddingMLP(
        data.cardinalities,
        n_cont,
        emb_dim=cfg.embedding_dim,
        dropout=cfg.mlp_dropout,
    ).to(device)
    best_epochs = _fit_holdout_epochs(
        model, holdout_train, holdout_val, data.y_train.iloc[tr_idx], device, cfg
    )

    final_seed = fold_seed(random_state, fold=1)
    set_seed(final_seed)
    final_model = TitanicEmbeddingMLP(
        data.cardinalities,
        n_cont,
        emb_dim=cfg.embedding_dim,
        dropout=cfg.mlp_dropout,
    ).to(device)
    full_loader = _train_dataloader(
        TabularDataset(data.cat_train, num_all, data.y_train),
        cfg.batch_size,
        final_seed,
    )
    _train_fixed_epochs(
        final_model, full_loader, data.y_train, device, cfg, best_epochs
    )
    return predict_embedding(
        final_model, data.cat_test, num_test, device, cfg.batch_size
    )


def create_submission(
    experiment: DictConfig | None = None,
    *,
    output_path: Path | str | None = None,
    feature_mode: FeatureMode | str | None = None,
) -> Path:
    exp = experiment if experiment is not None else load_config()
    set_seed(exp.random_state)
    mode = resolve_feature_mode(feature_mode, exp)
    out_path = Path(
        output_path
        or exp.paths.get("dl_submission_csv", exp.paths.submission_csv)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = FeatureBuilder()

    df_train = builder.read_raw(exp.paths.train_csv)
    df_test = builder.read_raw(exp.paths.test_csv)
    features = builder.build_train_test(df_train, df_test, mode)

    print(f"Feature mode: {mode.value}")
    print(f"Train rows: {len(df_train)}, test rows: {len(df_test)}")
    print(f"Device: {device}")

    if isinstance(features, TrainTestOneHot):
        preds = train_and_predict_onehot(
            features,
            device,
            exp.train,
            holdout_fraction=exp.val_size,
            random_state=exp.random_state,
        )
    elif isinstance(features, TrainTestEmbedding):
        preds = train_and_predict_embedding(
            features,
            device,
            exp.train,
            holdout_fraction=exp.val_size,
            random_state=exp.random_state,
        )
    else:
        raise TypeError(f"Unexpected features type: {type(features)}")

    submission = pd.DataFrame(
        {
            "PassengerId": features.test_passenger_ids.values,
            "Survived": preds.astype(int),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    print(f"Saved {len(submission)} predictions -> {out_path.resolve()}")
    print(submission["Survived"].value_counts().to_string())
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Titanic Kaggle submission")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to submission CSV (default: from config)",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in FeatureMode if m != FeatureMode.BASELINE],
        default=None,
        help="onehot or embedding (default: from config)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides, e.g. train.lr=0.001 feature_mode=embedding",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exp = load_config(args.config, overrides=args.overrides or None)
    create_submission(
        experiment=exp,
        feature_mode=args.mode,
        output_path=args.output,
    )
