"""Загрузка настроек из config.yaml через OmegaConf."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

CONFIG_YAML = Path(__file__).with_name("config.yaml")


def fold_seed(base: int, fold: int = 0) -> int:
    """Один random_state в конфиге; для k-fold: base + номер фолда."""
    return int(base) + int(fold)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """
    Синхронизирует генераторы Python / NumPy / PyTorch (и CUDA при наличии).
    deterministic=True: стабильнее на GPU, может быть чуть медленнее.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def shuffle_generator(seed: int) -> torch.Generator:
    """Для DataLoader(shuffle=True) — воспроизводимый порядок батчей."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


class FeatureMode(str, Enum):
    BASELINE = "baseline"
    ONEHOT = "onehot"
    EMBEDDING = "embedding"


def load_config(
    path: Path | str | None = None,
    overrides: dict[str, Any] | list[str] | None = None,
) -> DictConfig:
    """
    Загружает YAML и опционально применяет overrides.

    overrides как dict: {"train": {"lr": 0.001}}
    overrides как list (CLI-стиль): ["train.lr=0.001", "feature_mode=embedding"]
    """
    cfg = OmegaConf.load(path or CONFIG_YAML)
    if overrides:
        if isinstance(overrides, dict):
            cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        else:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def resolve_feature_mode(mode: FeatureMode | str | None, cfg: DictConfig) -> FeatureMode:
    raw = mode if mode is not None else cfg.feature_mode
    if isinstance(raw, FeatureMode):
        return raw
    return FeatureMode(str(raw).lower())


def train_config_label(train_cfg: DictConfig) -> str:
    amp_suffix = ""
    if bool(getattr(train_cfg, "use_amp", False)):
        amp_suffix = f" amp={getattr(train_cfg, 'amp_dtype', 'fp16')}"
    return (
        f"lr={train_cfg.lr:g} wd={train_cfg.weight_decay:g} "
        f"bs={train_cfg.batch_size} sched={train_cfg.scheduler} "
        f"cw={train_cfg.use_class_weights}{amp_suffix}"
    )


def with_train_overrides(cfg: DictConfig, **train_kwargs: Any) -> DictConfig:
    """Копия конфига с переопределёнными полями в секции train."""
    return OmegaConf.merge(cfg, OmegaConf.create({"train": train_kwargs}))


DEFAULT_EXPERIMENT: DictConfig = load_config()

# Удобные алиасы для type hints в других модулях
ExperimentConfig = DictConfig
TrainConfig = DictConfig
