"""Единый конфиг проекта: merge трёх YAML + DL-хелперы."""

from __future__ import annotations

import random
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from paths import ROOT, ensure_output_dirs

CONFIG_LAYERS = (
    ROOT / "config.yaml",
    ROOT / "ml" / "config.yaml",
    ROOT / "dl" / "config.yaml",
)

_PATH_ALIASES = (
    ("train", "train_csv"),
    ("test", "test_csv"),
    ("submission", "submission_csv"),
    ("dl_submission", "dl_submission_csv"),
)


class FeatureMode(str, Enum):
    BASELINE = "baseline"
    ONEHOT = "onehot"
    EMBEDDING = "embedding"


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | list[str] | None = None,
) -> DictConfig:
    """Загрузить merged-конфиг из трёх YAML (или один файл) и применить overrides."""
    if config_path is not None:
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create({})
        for layer in CONFIG_LAYERS:
            if layer.exists():
                cfg = OmegaConf.merge(cfg, OmegaConf.load(layer))

    if overrides:
        if isinstance(overrides, dict):
            cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        else:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    _resolve_paths(cfg)
    _sync_path_aliases(cfg)
    OmegaConf.resolve(cfg)
    ensure_output_dirs()
    return cfg


def _resolve_paths(cfg: DictConfig) -> None:
    if "paths" not in cfg:
        return
    for key, value in cfg.paths.items():
        path = Path(str(value))
        if not path.is_absolute():
            cfg.paths[key] = str((ROOT / path).resolve())


def _sync_path_aliases(cfg: DictConfig) -> None:
    if "paths" not in cfg:
        return
    paths = cfg.paths
    for left, right in _PATH_ALIASES:
        if left in paths and right not in paths:
            paths[right] = paths[left]
        elif right in paths and left not in paths:
            paths[left] = paths[right]


def fold_seed(base: int, fold: int = 0) -> int:
    return int(base) + int(fold)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
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
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


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
    return OmegaConf.merge(cfg, OmegaConf.create({"train": train_kwargs}))
