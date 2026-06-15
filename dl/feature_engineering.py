from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FeatureMode
from ml.feature_engineering import FeatureBuilder as MLFeatureBuilder

_DL_CAT_COLS = ("Pclass", "Sex", "Embarked", "Title", "Deck")
_DL_CONT_COLS = (
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "HasCabin",
)


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
    """DL matrices on top of the shared ML feature pipeline."""

    def __init__(self) -> None:
        self._ml = MLFeatureBuilder()
        self.cfg = self._ml.cfg

    def read_raw(self, path: str | pd.PathLike) -> pd.DataFrame:
        return self._ml.read_raw(path)

    def _transform_split(
        self, train_raw: pd.DataFrame, other_raw: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        stats = self._ml.fit_pipeline_stats(train_raw)
        train_df = self._to_dl_frame(self._ml.transform(train_raw, stats))
        other_df = self._to_dl_frame(self._ml.transform(other_raw, stats))
        return train_df, other_df

    def _to_dl_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Deck"] = out["DeckCoarse"].astype(str)
        out["Embarked"] = out["Embarked"].astype(str)
        out["Title"] = out["Title"].astype(str)
        return out

    def build_fold(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        mode: FeatureMode,
    ) -> OneHotFoldData | EmbeddingFoldData | BaselineFoldData:
        train_raw = df.iloc[train_idx]
        val_raw = df.iloc[val_idx]
        y_train = df[self.cfg.target_col].iloc[train_idx]
        y_val = df[self.cfg.target_col].iloc[val_idx]

        if mode == FeatureMode.BASELINE:
            stats = self._ml.fit_pipeline_stats(train_raw)

            def baseline_block(raw: pd.DataFrame) -> pd.DataFrame:
                block = self._to_dl_frame(self._ml.transform(raw, stats))
                return block[
                    ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
                ].astype(float)

            return BaselineFoldData(
                X_train=baseline_block(train_raw),
                X_val=baseline_block(val_raw),
                y_train=y_train,
                y_val=y_val,
            )

        tr_i, va_i = self._transform_split(train_raw, val_raw)
        if mode == FeatureMode.ONEHOT:
            X_train, columns = self._encode_onehot(tr_i, reference_columns=None)
            X_val, _ = self._encode_onehot(va_i, reference_columns=columns)
            return OneHotFoldData(X_train, X_val, y_train, y_val, columns)

        if mode == FeatureMode.EMBEDDING:
            vocab, cardinalities = self._fit_category_vocab(tr_i)
            return EmbeddingFoldData(
                self._encode_categories(tr_i, vocab),
                self._encode_categories(va_i, vocab),
                tr_i[list(_DL_CONT_COLS)].astype(np.float64).values,
                va_i[list(_DL_CONT_COLS)].astype(np.float64).values,
                y_train,
                y_val,
                cardinalities,
            )

        raise ValueError(f"Unknown feature mode: {mode!r}")

    def build_train_test(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        mode: FeatureMode,
    ) -> TrainTestOneHot | TrainTestEmbedding:
        tr_i, te_i = self._transform_split(df_train, df_test)
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
            return TrainTestEmbedding(
                cat_train=self._encode_categories(tr_i, vocab),
                cat_test=self._encode_categories(te_i, vocab),
                num_train=tr_i[list(_DL_CONT_COLS)].astype(np.float64).values,
                num_test=te_i[list(_DL_CONT_COLS)].astype(np.float64).values,
                y_train=y_train,
                test_passenger_ids=test_ids,
                cardinalities=cardinalities,
            )

        raise ValueError(
            f"build_train_test supports onehot and embedding, got {mode!r}"
        )

    def _encode_onehot(
        self,
        df_imputed: pd.DataFrame,
        reference_columns: list[str] | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        X_num = df_imputed[list(_DL_CONT_COLS)].astype(np.float64)
        dums = pd.get_dummies(
            df_imputed[list(_DL_CAT_COLS)],
            columns=list(_DL_CAT_COLS),
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
        out = np.zeros((n, len(_DL_CAT_COLS)), dtype=np.int64)

        pclass = df_imputed["Pclass"].astype(int)
        out[:, 0] = pclass.map(vocab["Pclass"]).fillna(0).astype(np.int64)

        sex = df_imputed["Sex"].astype(int)
        out[:, 1] = sex.map(vocab["Sex"]).fillna(0).astype(np.int64)

        for j, col in enumerate(("Embarked", "Title", "Deck"), start=2):
            spec = vocab[col]
            mapping, unk = spec["map"], spec["unk"]
            out[:, j] = (
                df_imputed[col]
                .astype(str)
                .map(lambda v, m=mapping, u=unk: m.get(v, u))
                .astype(np.int64)
            )

        return out


def load_baseline_xy(
    path: str | pd.PathLike,
) -> tuple[pd.DataFrame, pd.Series]:
    builder = FeatureBuilder()
    df = builder.read_raw(path)
    stats = builder._ml.fit_pipeline_stats(df)
    block = builder._to_dl_frame(builder._ml.transform(df, stats))
    X = block[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].astype(float)
    y = df[builder.cfg.target_col]
    return X, y
