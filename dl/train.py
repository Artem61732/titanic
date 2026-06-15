from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from omegaconf import DictConfig

from config import (
    FeatureMode,
    fold_seed,
    load_config,
    resolve_feature_mode,
    set_seed,
    shuffle_generator,
    train_config_label,
    with_train_overrides,
)
from dl.feature_engineering import (
    BaselineFoldData,
    EmbeddingFoldData,
    FeatureBuilder,
    OneHotFoldData,
)

# Datasets


class MatrixDataset(Dataset):
    """Один тензор признаков (one-hot / baseline)."""

    def __init__(self, X: np.ndarray, y: pd.Series | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = (
            torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
            if y is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class TabularDataset(Dataset):
    """Категории (embedding) + непрерывные признаки."""

    def __init__(
        self,
        cat: np.ndarray,
        num: np.ndarray,
        y: pd.Series | None = None,
    ):
        self.cat = torch.tensor(cat, dtype=torch.long)
        self.num = torch.tensor(num, dtype=torch.float32)
        self.y = (
            torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
            if y is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.cat)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.cat[idx], self.num[idx], self.y[idx]
        return self.cat[idx], self.num[idx]


# Models


class TitanicMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        hidden = max(32, min(128, input_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TitanicEmbeddingMLP(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        num_dim: int,
        emb_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, emb_dim) for n in cat_cardinalities]
        )
        mlp_in = len(cat_cardinalities) * emb_dim + num_dim
        hidden = max(32, min(128, mlp_in * 2))
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        parts = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.mlp(torch.cat(parts + [num_x], dim=1))


# Training loop


def _forward(model: nn.Module, batch, device: torch.device):
    if len(batch) == 3:
        cat_x, num_x, y = batch
        cat_x, num_x, y = cat_x.to(device), num_x.to(device), y.to(device)
        return model(cat_x, num_x), y
    x, y = batch
    return model(x.to(device)), y.to(device)


def _resolve_amp_dtype(cfg: DictConfig) -> torch.dtype:
    dtype = str(getattr(cfg, "amp_dtype", "fp16")).lower()
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def _use_amp(cfg: DictConfig, device: torch.device) -> bool:
    return bool(getattr(cfg, "use_amp", False)) and device.type == "cuda"


def _make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler=None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            out, y = _forward(model, batch, device)
            loss = criterion(out, y)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            out, y = _forward(model, batch, device)
            loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def make_criterion(
    y_train: pd.Series, use_class_weights: bool, device: torch.device
) -> nn.BCEWithLogitsLoss:
    if not use_class_weights:
        return nn.BCEWithLogitsLoss()
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_scheduler(optimizer, cfg: DictConfig):
    if cfg.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    return None


def _step_scheduler(scheduler, val_loss: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


# Fold preparation


def _train_dataloader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator(seed),
    )


def _make_loaders_from_baseline(
    data: BaselineFoldData, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(data.X_train.values)
    X_va = scaler.transform(data.X_val.values)
    train_loader = _train_dataloader(
        MatrixDataset(X_tr, data.y_train), batch_size, seed
    )
    val_loader = DataLoader(
        MatrixDataset(X_va, data.y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, X_tr, X_va


def _make_loaders_from_onehot(
    data: OneHotFoldData, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(data.X_train.values)
    X_va = scaler.transform(data.X_val.values)
    train_loader = _train_dataloader(
        MatrixDataset(X_tr, data.y_train), batch_size, seed
    )
    val_loader = DataLoader(
        MatrixDataset(X_va, data.y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, X_tr, X_va


def _make_loaders_from_embedding(
    data: EmbeddingFoldData, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(data.num_train)
    num_va = scaler.transform(data.num_val)
    train_loader = _train_dataloader(
        TabularDataset(data.cat_train, num_tr, data.y_train),
        batch_size,
        seed,
    )
    val_loader = DataLoader(
        TabularDataset(data.cat_val, num_va, data.y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def prepare_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mode: FeatureMode,
    batch_size: int,
    device: torch.device,
    random_state: int,
    fold: int,
    builder: FeatureBuilder,
    train_cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, nn.Module, pd.Series]:
    seed = fold_seed(random_state, fold)
    set_seed(seed)
    fold_data = builder.build_fold(df, train_idx, val_idx, mode)

    if isinstance(fold_data, BaselineFoldData):
        train_loader, val_loader, X_tr, _ = _make_loaders_from_baseline(
            fold_data, batch_size, seed
        )
        model = TitanicMLP(X_tr.shape[1], dropout=train_cfg.mlp_dropout).to(device)
        return train_loader, val_loader, model, fold_data.y_train

    if isinstance(fold_data, OneHotFoldData):
        train_loader, val_loader, X_tr, _ = _make_loaders_from_onehot(
            fold_data, batch_size, seed
        )
        model = TitanicMLP(X_tr.shape[1], dropout=train_cfg.mlp_dropout).to(device)
        return train_loader, val_loader, model, fold_data.y_train

    if isinstance(fold_data, EmbeddingFoldData):
        train_loader, val_loader = _make_loaders_from_embedding(
            fold_data, batch_size, seed
        )
        n_cont = len(builder.cfg.cont_cols)
        model = TitanicEmbeddingMLP(
            fold_data.cardinalities,
            n_cont,
            emb_dim=train_cfg.embedding_dim,
            dropout=train_cfg.mlp_dropout,
        ).to(device)
        return train_loader, val_loader, model, fold_data.y_train

    raise RuntimeError(f"Unexpected fold data type: {type(fold_data)}")


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: pd.Series,
    device: torch.device,
    cfg: DictConfig,
    verbose: bool = False,
) -> tuple[float, float, int]:
    criterion = make_criterion(y_train, cfg.use_class_weights, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = make_scheduler(optimizer, cfg)
    use_amp = _use_amp(cfg, device)
    amp_dtype = _resolve_amp_dtype(cfg)
    scaler = _make_grad_scaler(use_amp)

    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0
    last_val_loss, last_val_acc = 0.0, 0.0

    for epoch in range(1, cfg.max_epochs + 1):
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
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        _step_scheduler(scheduler, val_loss)

        if val_loss < best_val_loss - cfg.early_stopping_min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        last_val_loss, last_val_acc = val_loss, val_acc

        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  ep {epoch:03d} | val loss {val_loss:.4f} "
                f"| val acc {val_acc:.4f} | lr {lr_now:.2e}"
            )

        if stale_epochs >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        last_val_loss, last_val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

    return last_val_loss, last_val_acc, epoch


# Cross-validation & grid search

_MODE_LABELS = {
    FeatureMode.BASELINE: "baseline (6 feat)",
    FeatureMode.ONEHOT: "one-hot (engineered)",
    FeatureMode.EMBEDDING: "embedding (engineered)",
}


def stratified_kfold_cv(
    experiment: DictConfig | None = None,
    *,
    data_path: str | None = None,
    feature_mode: FeatureMode | str | None = None,
    train_config: DictConfig | None = None,
    verbose_folds: bool = False,
) -> dict:
    exp = experiment if experiment is not None else load_config()
    path = data_path or str(exp.paths.train_csv)
    mode = resolve_feature_mode(feature_mode, exp)
    train_cfg = train_config or exp.train
    builder = FeatureBuilder()

    set_seed(exp.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = builder.read_raw(path)
    y = df[builder.cfg.target_col]

    skf = StratifiedKFold(
        n_splits=exp.n_splits,
        shuffle=True,
        random_state=exp.random_state,
    )

    label = _MODE_LABELS.get(mode, str(mode))
    print(f"\n--- {label} | {train_config_label(train_cfg)} ---")

    fold_val_accs: list[float] = []
    fold_val_losses: list[float] = []
    fold_epochs: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, y), start=1
    ):
        train_loader, val_loader, model, y_train = prepare_fold(
            df,
            train_idx,
            val_idx,
            mode,
            train_cfg.batch_size,
            device,
            exp.random_state,
            fold,
            builder,
            train_cfg,
        )
        val_loss, val_acc, epochs_run = train_fold(
            model,
            train_loader,
            val_loader,
            y_train,
            device,
            train_cfg,
            verbose=verbose_folds,
        )
        fold_val_losses.append(val_loss)
        fold_val_accs.append(val_acc)
        fold_epochs.append(epochs_run)
        print(
            f"Fold {fold}/{exp.n_splits} | epochs {epochs_run} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    acc_mean = float(np.mean(fold_val_accs))
    acc_std = float(np.std(fold_val_accs, ddof=1)) if exp.n_splits > 1 else 0.0
    loss_mean = float(np.mean(fold_val_losses))
    loss_std = float(np.std(fold_val_losses, ddof=1)) if exp.n_splits > 1 else 0.0

    print(
        f"\nStratified {exp.n_splits}-fold ({label}) | "
        f"Val Acc: {acc_mean:.4f} +/- {acc_std:.4f} | "
        f"Val Loss: {loss_mean:.4f} +/- {loss_std:.4f} | "
        f"avg epochs {np.mean(fold_epochs):.1f}"
    )

    return {
        "train_config": train_cfg,
        "feature_mode": mode,
        "val_acc_per_fold": fold_val_accs,
        "val_loss_per_fold": fold_val_losses,
        "val_acc_mean": acc_mean,
        "val_acc_std": acc_std,
        "val_loss_mean": loss_mean,
        "val_loss_std": loss_std,
        "epochs_per_fold": fold_epochs,
    }


def get_loaders(
    data_path: str | None = None,
    experiment: DictConfig | None = None,
    feature_mode: FeatureMode | str | None = None,
):
    """Один stratified train/val split для быстрых экспериментов."""
    exp = experiment if experiment is not None else load_config()
    path = data_path or str(exp.paths.train_csv)
    mode = resolve_feature_mode(feature_mode, exp)
    batch_size = exp.train.batch_size
    builder = FeatureBuilder()

    df = builder.read_raw(path)
    y = df[builder.cfg.target_col]
    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=exp.val_size,
        random_state=exp.random_state,
        stratify=y,
    )

    seed = fold_seed(exp.random_state)
    set_seed(seed)
    fold_data = builder.build_fold(df, tr_idx, va_idx, mode)

    if isinstance(fold_data, EmbeddingFoldData):
        scaler = StandardScaler()
        num_tr = scaler.fit_transform(fold_data.num_train)
        num_va = scaler.transform(fold_data.num_val)
        train_loader = _train_dataloader(
            TabularDataset(fold_data.cat_train, num_tr, fold_data.y_train),
            batch_size,
            seed,
        )
        val_loader = DataLoader(
            TabularDataset(fold_data.cat_val, num_va, fold_data.y_val),
            batch_size=batch_size,
            shuffle=False,
        )
        return (
            train_loader,
            val_loader,
            fold_data.cardinalities,
            len(builder.cfg.cont_cols),
        )

    if isinstance(fold_data, OneHotFoldData):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(fold_data.X_train.values)
        X_va = scaler.transform(fold_data.X_val.values)
        train_loader = _train_dataloader(
            MatrixDataset(X_tr, fold_data.y_train), batch_size, seed
        )
        val_loader = DataLoader(
            MatrixDataset(X_va, fold_data.y_val),
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, val_loader, fold_data.X_train.shape[1]

    raise ValueError("get_loaders supports onehot and embedding modes only")


def default_hyperparameter_grid(
    experiment: DictConfig | None = None,
) -> list[DictConfig]:
    exp = experiment if experiment is not None else load_config()
    base = with_train_overrides(
        exp, max_epochs=80, early_stopping_patience=10
    )
    return [
        with_train_overrides(base, batch_size=32, scheduler="plateau"),
        with_train_overrides(base, batch_size=64, scheduler="plateau"),
        with_train_overrides(base, batch_size=128, scheduler="plateau"),
        with_train_overrides(base, batch_size=64, scheduler="cosine"),
        with_train_overrides(base, lr=5e-4, weight_decay=1e-4),
        with_train_overrides(base, lr=2e-3, weight_decay=1e-4),
        with_train_overrides(base, lr=1e-3, weight_decay=1e-3),
        with_train_overrides(base, use_class_weights=True),
    ]


def hyperparameter_grid_search(
    experiment: DictConfig | None = None,
    *,
    data_path: str | None = None,
    feature_mode: FeatureMode | str | None = None,
    configs: list[DictConfig] | None = None,
) -> list[dict]:
    exp = experiment if experiment is not None else load_config()
    mode = resolve_feature_mode(feature_mode, exp)
    configs = configs or default_hyperparameter_grid(exp)

    results = []
    print(f"Grid search: {len(configs)} configs, {mode.value}, {exp.n_splits}-fold")

    for i, cfg in enumerate(configs, start=1):
        print(f"\n[{i}/{len(configs)}] {train_config_label(cfg.train)}")
        out = stratified_kfold_cv(
            exp,
            data_path=data_path,
            feature_mode=mode,
            train_config=cfg.train,
        )
        results.append(
            {
                "config": cfg,
                "val_acc_mean": out["val_acc_mean"],
                "val_acc_std": out["val_acc_std"],
                "val_loss_mean": out["val_loss_mean"],
            }
        )

    results.sort(key=lambda r: r["val_acc_mean"], reverse=True)
    print("\n=== Top configs (by mean val acc) ===")
    for rank, row in enumerate(results[:5], start=1):
        c = row["config"]
        print(
            f"{rank}. acc {row['val_acc_mean']:.4f} +/- {row['val_acc_std']:.4f} | "
            f"{train_config_label(c.train)}"
        )
    return results


# CLI

if __name__ == "__main__":
    from dl.main import evaluate_dnn_experiments

    evaluate_dnn_experiments(run_cosine=True, run_grid=True)
