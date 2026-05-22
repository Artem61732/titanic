from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FeatureMode


@dataclass(frozen=True)
class FeatureSettings:
    """Параметры feature engineering (не в YAML)."""

    target_col: str = "Survived"
    cat_cols: tuple[str, ...] = ("Pclass", "Sex", "Embarked", "Title", "Deck")
    cont_cols: tuple[str, ...] = (
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "HasCabin",
    )
    baseline_cols: tuple[str, ...] = (
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    )
    sex_map: tuple[tuple[str, int], ...] = (("male", 0), ("female", 1))
    default_embarked: str = "S"
    rare_titles: frozenset[str] = frozenset(
        {
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        }
    )
    title_aliases: tuple[tuple[str, str], ...] = (
        ("Mlle", "Miss"),
        ("Ms", "Miss"),
        ("Mme", "Mrs"),
    )
    common_titles: frozenset[str] = frozenset({"Mr", "Mrs", "Miss", "Master"})
    valid_deck_letters: frozenset[str] = frozenset("ABCDEFGT")
    unknown_deck: str = "U"
    rare_title_label: str = "Rare"


DEFAULT_FEATURE_SETTINGS = FeatureSettings()


@dataclass(frozen=True)
class ImputationStats:
    age_median: float
    fare_median: float
    embarked_mode: str


@dataclass
class OneHotFoldData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    feature_columns: list[str]


@dataclass
class EmbeddingFoldData:
    cat_train: np.ndarray
    cat_val: np.ndarray
    num_train: np.ndarray
    num_val: np.ndarray
    y_train: pd.Series
    y_val: pd.Series
    cardinalities: list[int]


@dataclass
class BaselineFoldData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series


@dataclass
class TrainTestOneHot:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    test_passenger_ids: pd.Series


@dataclass
class TrainTestEmbedding:
    cat_train: np.ndarray
    cat_test: np.ndarray
    num_train: np.ndarray
    num_test: np.ndarray
    y_train: pd.Series
    test_passenger_ids: pd.Series
    cardinalities: list[int]


class FeatureBuilder:
    """Пайплайн признаков с имputation только по train-части сплита."""

    def __init__(self, settings: FeatureSettings | None = None):
        self.cfg = settings or DEFAULT_FEATURE_SETTINGS
        self._sex_map = dict(self.cfg.sex_map)
        self._rare_titles = self.cfg.rare_titles
        self._common_titles = self.cfg.common_titles
        self._title_aliases = dict(self.cfg.title_aliases)

    def read_raw(self, path: str | pd.PathLike) -> pd.DataFrame:
        return pd.read_csv(path)

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()
        out["Title"] = out["Name"].apply(self._parse_title)
        out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
        out["HasCabin"] = out["Cabin"].notna().astype(int)
        out["Deck"] = out["Cabin"].apply(self._deck_from_cabin)
        out["Sex"] = out["Sex"].map(self._sex_map)
        return out

    def imputation_stats(self, train_raw: pd.DataFrame) -> ImputationStats:
        embarked = train_raw["Embarked"]
        if embarked.notna().any():
            emb_mode = embarked.dropna().mode().iloc[0]
        else:
            emb_mode = self.cfg.default_embarked
        return ImputationStats(
            age_median=float(train_raw["Age"].median()),
            fare_median=float(train_raw["Fare"].median()),
            embarked_mode=str(emb_mode),
        )

    def apply_imputation(
        self, df_feat: pd.DataFrame, stats: ImputationStats
    ) -> pd.DataFrame:
        out = df_feat.copy()
        out["Age"] = out["Age"].fillna(stats.age_median)
        out["Fare"] = out["Fare"].fillna(stats.fare_median)
        out["Embarked"] = out["Embarked"].fillna(stats.embarked_mode)
        return out

    def build_fold(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        mode: FeatureMode,
    ) -> OneHotFoldData | EmbeddingFoldData | BaselineFoldData:
        if mode == FeatureMode.BASELINE:
            return self._build_baseline(df, train_idx, val_idx)
        if mode == FeatureMode.ONEHOT:
            return self._build_onehot(df, train_idx, val_idx)
        if mode == FeatureMode.EMBEDDING:
            return self._build_embedding(df, train_idx, val_idx)
        raise ValueError(f"Unknown feature mode: {mode!r}")

    def build_train_test(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        mode: FeatureMode,
    ) -> TrainTestOneHot | TrainTestEmbedding:
        """
        Признаки для финальной модели: imputation/vocab только по train,
        test трансформируется теми же правилами.
        """
        stats = self.imputation_stats(df_train)
        tr_i = self.apply_imputation(self.featurize(df_train), stats)
        te_i = self.apply_imputation(self.featurize(df_test), stats)
        y_train = df_train[self.cfg.target_col]
        test_ids = df_test["PassengerId"]

        if mode == FeatureMode.ONEHOT:
            X_tr, columns = self._encode_onehot(tr_i, reference_columns=None)
            X_te, _ = self._encode_onehot(te_i, reference_columns=columns)
            return TrainTestOneHot(
                X_train=X_tr.values,
                X_test=X_te.values,
                y_train=y_train,
                test_passenger_ids=test_ids,
            )

        if mode == FeatureMode.EMBEDDING:
            vocab, cardinalities = self._fit_category_vocab(tr_i)
            cat_tr = self._encode_categories(tr_i, vocab)
            cat_te = self._encode_categories(te_i, vocab)
            num_tr = tr_i[list(self.cfg.cont_cols)].astype(np.float64).values
            num_te = te_i[list(self.cfg.cont_cols)].astype(np.float64).values
            return TrainTestEmbedding(
                cat_train=cat_tr,
                cat_test=cat_te,
                num_train=num_tr,
                num_test=num_te,
                y_train=y_train,
                test_passenger_ids=test_ids,
                cardinalities=cardinalities,
            )

        raise ValueError(
            f"build_train_test supports onehot and embedding, got {mode!r}"
        )

    def _labels(
        self, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> tuple[pd.Series, pd.Series]:
        y = df[self.cfg.target_col]
        return y.iloc[train_idx], y.iloc[val_idx]

    def _prepare_imputed_splits(
        self, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_raw = df.iloc[train_idx]
        val_raw = df.iloc[val_idx]
        stats = self.imputation_stats(train_raw)
        tr_i = self.apply_imputation(self.featurize(train_raw), stats)
        va_i = self.apply_imputation(self.featurize(val_raw), stats)
        return tr_i, va_i

    def _build_baseline(
        self, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> BaselineFoldData:
        train_raw = df.iloc[train_idx]
        val_raw = df.iloc[val_idx]
        stats = self.imputation_stats(train_raw)
        y_train, y_val = self._labels(df, train_idx, val_idx)

        def block(raw: pd.DataFrame) -> pd.DataFrame:
            d = raw.copy()
            d["Sex"] = d["Sex"].map(self._sex_map)
            d["Age"] = d["Age"].fillna(stats.age_median)
            d["Fare"] = d["Fare"].fillna(stats.fare_median)
            return d[list(self.cfg.baseline_cols)]

        return BaselineFoldData(
            X_train=block(train_raw),
            X_val=block(val_raw),
            y_train=y_train,
            y_val=y_val,
        )

    def _build_onehot(
        self, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> OneHotFoldData:
        tr_i, va_i = self._prepare_imputed_splits(df, train_idx, val_idx)
        y_train, y_val = self._labels(df, train_idx, val_idx)
        X_train, columns = self._encode_onehot(tr_i, reference_columns=None)
        X_val, _ = self._encode_onehot(va_i, reference_columns=columns)
        return OneHotFoldData(X_train, X_val, y_train, y_val, columns)

    def _build_embedding(
        self, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> EmbeddingFoldData:
        tr_i, va_i = self._prepare_imputed_splits(df, train_idx, val_idx)
        y_train, y_val = self._labels(df, train_idx, val_idx)
        vocab, cardinalities = self._fit_category_vocab(tr_i)
        cat_train = self._encode_categories(tr_i, vocab)
        cat_val = self._encode_categories(va_i, vocab)
        num_train = tr_i[list(self.cfg.cont_cols)].astype(np.float64).values
        num_val = va_i[list(self.cfg.cont_cols)].astype(np.float64).values
        return EmbeddingFoldData(
            cat_train, cat_val, num_train, num_val, y_train, y_val, cardinalities
        )

    def _encode_onehot(
        self,
        df_imputed: pd.DataFrame,
        reference_columns: list[str] | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        cfg = self.cfg
        X_num = df_imputed[list(cfg.cont_cols)].astype(np.float64)
        dums = pd.get_dummies(
            df_imputed[list(cfg.cat_cols)],
            columns=list(cfg.cat_cols),
            dtype=float,
        )
        X = pd.concat(
            [X_num.reset_index(drop=True), dums.reset_index(drop=True)], axis=1
        )
        if reference_columns is not None:
            X = X.reindex(columns=reference_columns, fill_value=0.0)
        return X, list(X.columns)

    def _fit_category_vocab(
        self, train_imputed: pd.DataFrame
    ) -> tuple[dict, list[int]]:
        vocab: dict = {}
        cardinalities: list[int] = []

        vocab["Pclass"] = {1: 0, 2: 1, 3: 2}
        cardinalities.append(3)

        vocab["Sex"] = {0: 0, 1: 1}
        cardinalities.append(2)

        for col in ("Embarked", "Title", "Deck"):
            vals = sorted(train_imputed[col].astype(str).unique())
            mapping = {v: i for i, v in enumerate(vals)}
            unk_idx = len(vals)
            vocab[col] = {"map": mapping, "unk": unk_idx}
            cardinalities.append(unk_idx + 1)

        return vocab, cardinalities

    def _encode_categories(
        self, df_imputed: pd.DataFrame, vocab: dict
    ) -> np.ndarray:
        n = len(df_imputed)
        out = np.zeros((n, len(self.cfg.cat_cols)), dtype=np.int64)

        pclass = df_imputed["Pclass"].astype(int)
        out[:, 0] = pclass.map(vocab["Pclass"]).fillna(0).astype(np.int64)

        sex = df_imputed["Sex"].astype(int)
        out[:, 1] = sex.map(vocab["Sex"]).fillna(0).astype(np.int64)

        for j, col in enumerate(("Embarked", "Title", "Deck"), start=2):
            spec = vocab[col]
            m, unk = spec["map"], spec["unk"]
            out[:, j] = (
                df_imputed[col]
                .astype(str)
                .map(lambda v, mapping=m, u=unk: mapping.get(v, u))
                .astype(np.int64)
            )

        return out

    def _parse_title(self, name: str) -> str:
        cfg = self.cfg
        try:
            title = str(name).split(",")[1].split(".")[0].strip()
        except (IndexError, AttributeError):
            return cfg.rare_title_label
        if title in self._rare_titles:
            return cfg.rare_title_label
        if title in self._title_aliases:
            return self._title_aliases[title]
        if title in self._common_titles:
            return title
        return cfg.rare_title_label

    def _deck_from_cabin(self, cabin) -> str:
        cfg = self.cfg
        if pd.isna(cabin) or cabin == "":
            return cfg.unknown_deck
        ch = str(cabin)[0].upper()
        if ch in self.cfg.valid_deck_letters:
            return ch
        return cfg.unknown_deck


def load_baseline_xy(
    path: str | pd.PathLike,
    settings: FeatureSettings | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Legacy: 6 признаков, глобальные медианы по всему файлу."""
    cfg = settings or DEFAULT_FEATURE_SETTINGS
    builder = FeatureBuilder(cfg)
    df = builder.read_raw(path)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Sex"] = df["Sex"].map(dict(cfg.sex_map))
    X = df[list(cfg.baseline_cols)]
    y = df[cfg.target_col]
    return X, y
