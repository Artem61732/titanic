"""Предобработка (п.2) и feature engineering (п.5) для Titanic ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureMode(str, Enum):
    BASELINE = "baseline"
    ONEHOT = "onehot"
    LABEL = "label"


@dataclass(frozen=True)
class FeatureSettings:
    target_col: str = "Survived"
    id_col: str = "PassengerId"
    child_age_max: float = 12.0
    min_group_size_for_rate: int = 2
    fare_bin_count: int = 4
    # Категории для one-hot / label (Deck — укрупнённый, без редких букв)
    cat_cols: tuple[str, ...] = ("Pclass", "Sex", "Embarked", "Title", "DeckCoarse")
    cont_cols: tuple[str, ...] = (
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "LogFare",
        "FarePerPerson",
        "FareBin",
        "FamilySize",
        "TicketGroupSize",
        "IsAlone",
        "HasCabin",
        "IsWoman",
        "IsChild",
        "Age*Pclass",
        "SurnameSurvivalRate",
        "TicketSurvivalRate",
    )
    baseline_cols: tuple[str, ...] = (
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "LogFare",
        "IsWoman",
        "IsChild",
        "TicketGroupSize",
        "SurnameSurvivalRate",
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
class PipelineStats:
    """Статистики, посчитанные только по train (без leakage)."""

    age_median: float
    fare_median: float
    embarked_mode: str
    global_survival_rate: float
    age_by_title_pclass: dict[tuple[str, int], float]
    fare_by_pclass: dict[int, float]
    fare_bin_edges: np.ndarray
    ticket_group_sizes: dict[str, int]
    surname_survival: dict[str, float]
    ticket_survival: dict[str, float]


# обратная совместимость
ImputationStats = PipelineStats


@dataclass
class FoldMatrices:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: pd.Series
    y_val: pd.Series
    feature_names: list[str]
    cat_feature_indices: list[int] = field(default_factory=list)


@dataclass
class TrainTestMatrices:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    test_passenger_ids: pd.Series
    feature_names: list[str]
    cat_feature_indices: list[int] = field(default_factory=list)


@dataclass
class PreprocessReport:
    n_missing_before: dict[str, int]
    n_missing_after: dict[str, int]
    constant_dropped: list[str]
    correlated_dropped: list[str]
    outlier_clipped: dict[str, int]


class FeatureBuilder:
    """Leakage-safe пайплайн: fit только на train-части каждого сплита."""

    def __init__(self, settings: FeatureSettings | None = None):
        self.cfg = settings or DEFAULT_FEATURE_SETTINGS
        self._sex_map = dict(self.cfg.sex_map)
        self._rare_titles = self.cfg.rare_titles
        self._common_titles = self.cfg.common_titles
        self._title_aliases = dict(self.cfg.title_aliases)

    def read_raw(self, path: str | pd.PathLike) -> pd.DataFrame:
        return pd.read_csv(path)

    # --- п.5 Feature engineering (без таргета) -----------------------------------

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()
        out["Title"] = out["Name"].apply(self._parse_title)
        out["Surname"] = out["Name"].apply(self._parse_surname)
        out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
        out["HasCabin"] = out["Cabin"].notna().astype(int)
        deck = out["Cabin"].apply(self._deck_from_cabin)
        out["DeckCoarse"] = deck.map(self._coarse_deck)
        out["Sex"] = out["Sex"].map(self._sex_map)
        out["IsWoman"] = (out["Sex"] == 1).astype(int)
        out["IsChild"] = out.apply(self._is_child_row, axis=1).astype(int)
        out["Ticket"] = out["Ticket"].astype(str).str.strip()
        return out

    def fit_pipeline_stats(self, train_raw: pd.DataFrame) -> PipelineStats:
        """Fit imputation, bins и group survival rates только по train."""
        cfg = self.cfg
        feat = self.featurize(train_raw)
        target = train_raw[cfg.target_col]
        global_rate = float(target.mean())

        embarked = train_raw["Embarked"]
        emb_mode = (
            str(embarked.dropna().mode().iloc[0])
            if embarked.notna().any()
            else cfg.default_embarked
        )

        age_grp = (
            feat.groupby(["Title", "Pclass"], observed=True)["Age"]
            .median()
            .dropna()
        )
        age_by_title_pclass = {
            (str(title), int(pclass)): float(age)
            for (title, pclass), age in age_grp.items()
        }

        fare_by_pclass = {
            int(pclass): float(med)
            for pclass, med in feat.groupby("Pclass")["Fare"].median().items()
        }

        fare_valid = feat["Fare"].dropna()
        if len(fare_valid) >= cfg.fare_bin_count:
            edges = np.unique(
                np.quantile(
                    fare_valid,
                    np.linspace(0, 1, cfg.fare_bin_count + 1),
                )
            )
        else:
            edges = np.array([0.0, float(fare_valid.max() + 1.0)])
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])

        ticket_sizes = feat["Ticket"].value_counts().to_dict()
        ticket_sizes = {str(k): int(v) for k, v in ticket_sizes.items()}

        surname_survival = self._group_survival_rates(
            feat, target, "Surname", cfg.min_group_size_for_rate, global_rate
        )
        ticket_survival = self._group_survival_rates(
            feat, target, "Ticket", cfg.min_group_size_for_rate, global_rate
        )

        return PipelineStats(
            age_median=float(train_raw["Age"].median()),
            fare_median=float(train_raw["Fare"].median()),
            embarked_mode=emb_mode,
            global_survival_rate=global_rate,
            age_by_title_pclass=age_by_title_pclass,
            fare_by_pclass=fare_by_pclass,
            fare_bin_edges=edges,
            ticket_group_sizes=ticket_sizes,
            surname_survival=surname_survival,
            ticket_survival=ticket_survival,
        )

    def imputation_stats(self, train_raw: pd.DataFrame) -> PipelineStats:
        return self.fit_pipeline_stats(train_raw)

    def transform(
        self, df_raw: pd.DataFrame, stats: PipelineStats
    ) -> pd.DataFrame:
        """Featurize + imputation + производные числовые признаки."""
        out = self.featurize(df_raw)
        cfg = self.cfg

        out["Age"] = out.apply(
            lambda r: self._impute_age(r, stats), axis=1
        )
        out["Fare"] = out.apply(
            lambda r: self._impute_fare(r, stats), axis=1
        )
        out["Embarked"] = out["Embarked"].fillna(stats.embarked_mode)

        out["TicketGroupSize"] = (
            out["Ticket"]
            .map(lambda t, m=stats.ticket_group_sizes: m.get(str(t), 1))
            .astype(int)
        )
        out["FarePerPerson"] = out["Fare"] / out["TicketGroupSize"].clip(lower=1)
        out["LogFare"] = np.log1p(out["Fare"].clip(lower=0))
        out["FareBin"] = (
            pd.cut(
                out["Fare"],
                bins=stats.fare_bin_edges,
                labels=False,
                include_lowest=True,
            )
            .fillna(0)
            .astype(int)
        )
        out["Age*Pclass"] = out["Age"] * out["Pclass"]
        out["IsChild"] = out.apply(self._is_child_row, axis=1).astype(int)

        out["SurnameSurvivalRate"] = (
            out["Surname"]
            .map(lambda s, m=stats.surname_survival: m.get(str(s), stats.global_survival_rate))
            .astype(float)
        )
        out["TicketSurvivalRate"] = (
            out["Ticket"]
            .map(lambda t, m=stats.ticket_survival: m.get(str(t), stats.global_survival_rate))
            .astype(float)
        )
        return out

    def apply_imputation(
        self, df_feat: pd.DataFrame, stats: PipelineStats
    ) -> pd.DataFrame:
        """Legacy alias: ожидает уже featurize-нутый df или raw — пересобираем."""
        if "Title" not in df_feat.columns:
            return self.transform(df_feat, stats)
        return self.transform(df_feat, stats)

    def missing_counts(
        self, df: pd.DataFrame, cols: list[str] | None = None
    ) -> dict[str, int]:
        use = cols or [
            c for c in df.columns if c not in (self.cfg.id_col, self.cfg.target_col)
        ]
        return {c: int(df[c].isna().sum()) for c in use if c in df.columns}

    # --- п.2 Предобработка ---------------------------------------------------------

    def clip_outliers_iqr(
        self, train: pd.DataFrame, val: pd.DataFrame, cols: list[str], iqr: float = 1.5
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
        tr, va = train.copy(), val.copy()
        clipped: dict[str, int] = {}
        for col in cols:
            if col not in tr.columns:
                continue
            q1, q3 = tr[col].quantile(0.25), tr[col].quantile(0.75)
            iqr_val = q3 - q1
            lo, hi = q1 - iqr * iqr_val, q3 + iqr * iqr_val
            before = int((tr[col] < lo).sum() + (tr[col] > hi).sum())
            tr[col] = tr[col].clip(lo, hi)
            va[col] = va[col].clip(lo, hi)
            clipped[col] = before
        return tr, va, clipped

    def drop_constant_columns(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        nunique = X_train.nunique(dropna=False)
        const = nunique[nunique <= 1].index.tolist()
        if not const:
            return X_train, X_val, []
        return (
            X_train.drop(columns=const),
            X_val.drop(columns=const),
            const,
        )

    def drop_correlated_columns(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        threshold: float = 0.95,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
        if not to_drop:
            return X_train, X_val, []
        return (
            X_train.drop(columns=to_drop),
            X_val.drop(columns=to_drop),
            to_drop,
        )

    def build_fold(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        mode: FeatureMode | str,
        *,
        scale: bool = True,
        drop_constant: bool = True,
        drop_correlated: bool = True,
        correlated_threshold: float = 0.95,
        clip_outliers: bool = True,
        outlier_iqr: float = 1.5,
    ) -> FoldMatrices:
        mode = FeatureMode(mode) if isinstance(mode, str) else mode
        train_raw, val_raw = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = (
            df[self.cfg.target_col].iloc[train_idx],
            df[self.cfg.target_col].iloc[val_idx],
        )

        if mode == FeatureMode.BASELINE:
            return self._finalize_baseline(train_raw, val_raw, y_train, y_val, scale)

        stats = self.fit_pipeline_stats(train_raw)
        tr_i = self.transform(train_raw, stats)
        va_i = self.transform(val_raw, stats)

        if clip_outliers:
            clip_cols = [c for c in self.cfg.cont_cols if c in tr_i.columns]
            tr_i, va_i, _ = self.clip_outliers_iqr(
                tr_i, va_i, clip_cols, iqr=outlier_iqr
            )

        if mode == FeatureMode.ONEHOT:
            X_tr, cols = self._encode_onehot(tr_i, reference_columns=None)
            X_va, _ = self._encode_onehot(va_i, reference_columns=cols)
        elif mode == FeatureMode.LABEL:
            vocab = self._fit_label_vocab(tr_i)
            X_tr, cols = self._encode_label(tr_i, vocab)
            X_va, _ = self._encode_label(va_i, vocab)
        else:
            raise ValueError(f"Unknown feature mode: {mode!r}")

        if drop_constant:
            X_tr, X_va, _ = self.drop_constant_columns(X_tr, X_va)
        if drop_correlated:
            X_tr, X_va, _ = self.drop_correlated_columns(
                X_tr, X_va, threshold=correlated_threshold
            )

        X_tr_arr, X_va_arr, names = self._to_arrays(X_tr, X_va)
        if scale:
            scaler = StandardScaler()
            X_tr_arr = scaler.fit_transform(X_tr_arr)
            X_va_arr = scaler.transform(X_va_arr)

        return FoldMatrices(
            X_train=X_tr_arr,
            X_val=X_va_arr,
            y_train=y_train,
            y_val=y_val,
            feature_names=names,
            cat_feature_indices=self._cat_indices(names, mode),
        )

    def build_train_test(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        mode: FeatureMode | str,
        *,
        scale: bool = True,
        drop_constant: bool = True,
        drop_correlated: bool = True,
        correlated_threshold: float = 0.95,
        clip_outliers: bool = True,
        outlier_iqr: float = 1.5,
    ) -> TrainTestMatrices:
        mode = FeatureMode(mode) if isinstance(mode, str) else mode
        y_train = df_train[self.cfg.target_col]
        test_ids = df_test[self.cfg.id_col]

        if mode == FeatureMode.BASELINE:
            stats = self.fit_pipeline_stats(df_train)
            tr_i = self.transform(df_train, stats)
            te_i = self.transform(df_test, stats)
            X_tr = tr_i[list(self.cfg.baseline_cols)].astype(float)
            X_te = te_i[list(self.cfg.baseline_cols)].astype(float)
            names = list(self.cfg.baseline_cols)
            X_tr_arr, X_te_arr = X_tr.values, X_te.values
            if scale:
                scaler = StandardScaler()
                X_tr_arr = scaler.fit_transform(X_tr_arr)
                X_te_arr = scaler.transform(X_te_arr)
            return TrainTestMatrices(
                X_train=X_tr_arr,
                X_test=X_te_arr,
                y_train=y_train,
                test_passenger_ids=test_ids,
                feature_names=names,
            )

        stats = self.fit_pipeline_stats(df_train)
        tr_i = self.transform(df_train, stats)
        te_i = self.transform(df_test, stats)
        if clip_outliers:
            clip_cols = [c for c in self.cfg.cont_cols if c in tr_i.columns]
            tr_i, te_i, _ = self.clip_outliers_iqr(
                tr_i, te_i, clip_cols, iqr=outlier_iqr
            )

        if mode == FeatureMode.ONEHOT:
            X_tr, cols = self._encode_onehot(tr_i, reference_columns=None)
            X_te, _ = self._encode_onehot(te_i, reference_columns=cols)
        else:
            vocab = self._fit_label_vocab(tr_i)
            X_tr, cols = self._encode_label(tr_i, vocab)
            X_te, _ = self._encode_label(te_i, vocab)

        if drop_constant:
            X_tr, X_te, _ = self.drop_constant_columns(X_tr, X_te)
        if drop_correlated:
            X_tr, X_te, _ = self.drop_correlated_columns(
                X_tr, X_te, threshold=correlated_threshold
            )

        X_tr_arr, X_te_arr, names = self._to_arrays(X_tr, X_te)
        if scale:
            scaler = StandardScaler()
            X_tr_arr = scaler.fit_transform(X_tr_arr)
            X_te_arr = scaler.transform(X_te_arr)

        return TrainTestMatrices(
            X_train=X_tr_arr,
            X_test=X_te_arr,
            y_train=y_train,
            test_passenger_ids=test_ids,
            feature_names=names,
            cat_feature_indices=self._cat_indices(names, mode),
        )

    def preprocess_ablation_steps(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        mode: FeatureMode | str = FeatureMode.ONEHOT,
    ) -> list[tuple[str, FoldMatrices]]:
        mode = FeatureMode(mode) if isinstance(mode, str) else mode
        steps: list[tuple[str, FoldMatrices]] = []

        raw_tr, raw_va = df.iloc[train_idx], df.iloc[val_idx]
        y_tr = df[self.cfg.target_col].iloc[train_idx]
        y_va = df[self.cfg.target_col].iloc[val_idx]

        if mode == FeatureMode.BASELINE:
            m0 = self._finalize_baseline(raw_tr, raw_va, y_tr, y_va, scale=False)
            steps.append(("raw_baseline", m0))
            m1 = self._finalize_baseline(raw_tr, raw_va, y_tr, y_va, scale=True)
            steps.append(("scaled_baseline", m1))
            return steps

        stats = self.fit_pipeline_stats(raw_tr)
        tr0 = self.featurize(raw_tr)
        va0 = self.featurize(raw_va)
        steps.append(
            (
                "engineered_no_impute",
                self._matrices_from_frames(
                    tr0, va0, y_tr, y_va, mode, scale=False, skip_clean=True
                ),
            )
        )

        tr1 = self.transform(raw_tr, stats)
        va1 = self.transform(raw_va, stats)
        steps.append(
            (
                "after_imputation",
                self._matrices_from_frames(
                    tr1, va1, y_tr, y_va, mode, scale=False, skip_clean=True
                ),
            )
        )

        tr2, va2, _ = self.clip_outliers_iqr(
            tr1,
            va1,
            [c for c in self.cfg.cont_cols if c in tr1.columns],
            iqr=1.5,
        )
        steps.append(
            (
                "after_outlier_clip",
                self._matrices_from_frames(
                    tr2, va2, y_tr, y_va, mode, scale=False, skip_clean=True
                ),
            )
        )

        full = self.build_fold(
            df,
            train_idx,
            val_idx,
            mode,
            scale=True,
            drop_constant=True,
            drop_correlated=True,
            clip_outliers=True,
        )
        steps.append(("full_pipeline", full))
        return steps

    # --- encoders ------------------------------------------------------------------

    def _encode_onehot(
        self,
        df_imputed: pd.DataFrame,
        reference_columns: list[str] | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        cfg = self.cfg
        cont = [c for c in cfg.cont_cols if c in df_imputed.columns]
        X_num = df_imputed[cont].astype(np.float64)
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

    def _fit_label_vocab(self, train_imputed: pd.DataFrame) -> dict[str, dict]:
        vocab: dict[str, dict] = {}
        for col in self.cfg.cat_cols:
            vals = sorted(train_imputed[col].astype(str).unique())
            mapping = {v: i for i, v in enumerate(vals)}
            vocab[col] = {"map": mapping, "unk": len(vals)}
        return vocab

    def _encode_label(
        self,
        df_imputed: pd.DataFrame,
        vocab: dict[str, dict],
    ) -> tuple[pd.DataFrame, list[str]]:
        cfg = self.cfg
        cont = [c for c in cfg.cont_cols if c in df_imputed.columns]
        frames = [df_imputed[cont].astype(np.float64).reset_index(drop=True)]
        for col in cfg.cat_cols:
            m, unk = vocab[col]["map"], vocab[col]["unk"]
            enc = (
                df_imputed[col]
                .astype(str)
                .map(lambda v, mapping=m, u=unk: mapping.get(v, u))
                .astype(float)
            )
            frames.append(enc.rename(col).reset_index(drop=True))
        X = pd.concat(frames, axis=1)
        return X, list(X.columns)

    def _finalize_baseline(
        self,
        train_raw: pd.DataFrame,
        val_raw: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        scale: bool,
    ) -> FoldMatrices:
        stats = self.fit_pipeline_stats(train_raw)
        tr_i = self.transform(train_raw, stats)
        va_i = self.transform(val_raw, stats)
        X_tr = tr_i[list(self.cfg.baseline_cols)].astype(float).values
        X_va = va_i[list(self.cfg.baseline_cols)].astype(float).values
        names = list(self.cfg.baseline_cols)
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)
        return FoldMatrices(X_tr, X_va, y_train, y_val, names)

    def _matrices_from_frames(
        self,
        tr: pd.DataFrame,
        va: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        mode: FeatureMode,
        *,
        scale: bool,
        skip_clean: bool,
    ) -> FoldMatrices:
        if mode == FeatureMode.ONEHOT:
            X_tr, cols = self._encode_onehot(tr, reference_columns=None)
            X_va, _ = self._encode_onehot(va, reference_columns=cols)
        else:
            vocab = self._fit_label_vocab(tr)
            X_tr, cols = self._encode_label(tr, vocab)
            X_va, _ = self._encode_label(va, vocab)

        if not skip_clean:
            X_tr, X_va, _ = self.drop_constant_columns(X_tr, X_va)
            X_tr, X_va, _ = self.drop_correlated_columns(X_tr, X_va)

        X_tr_arr, X_va_arr, names = self._to_arrays(X_tr, X_va)
        if scale:
            scaler = StandardScaler()
            X_tr_arr = scaler.fit_transform(X_tr_arr)
            X_va_arr = scaler.transform(X_va_arr)
        return FoldMatrices(
            X_tr_arr,
            X_va_arr,
            y_train,
            y_val,
            names,
            cat_feature_indices=self._cat_indices(names, mode),
        )

    @staticmethod
    def _to_arrays(
        X_tr: pd.DataFrame, X_va: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        names = list(X_tr.columns)
        return X_tr.values.astype(float), X_va.values.astype(float), names

    def _cat_indices(self, names: list[str], mode: FeatureMode) -> list[int]:
        if mode != FeatureMode.LABEL:
            return []
        cat_names = set(self.cfg.cat_cols)
        return [i for i, n in enumerate(names) if n in cat_names]

    def _group_survival_rates(
        self,
        feat: pd.DataFrame,
        target: pd.Series,
        column: str,
        min_size: int,
        global_rate: float,
    ) -> dict[str, float]:
        tmp = feat[[column]].copy()
        tmp["_y"] = target.values
        counts = tmp[column].value_counts()
        multi = counts[counts >= min_size].index
        if len(multi) == 0:
            return {}
        rates = (
            tmp[tmp[column].isin(multi)]
            .groupby(column, observed=True)["_y"]
            .mean()
        )
        return {str(k): float(v) for k, v in rates.items()}

    def _impute_age(self, row: pd.Series, stats: PipelineStats) -> float:
        if pd.notna(row.get("Age")):
            return float(row["Age"])
        key = (str(row["Title"]), int(row["Pclass"]))
        if key in stats.age_by_title_pclass:
            return stats.age_by_title_pclass[key]
        return stats.age_median

    def _impute_fare(self, row: pd.Series, stats: PipelineStats) -> float:
        if pd.notna(row.get("Fare")):
            return float(row["Fare"])
        pclass = int(row["Pclass"])
        if pclass in stats.fare_by_pclass:
            return stats.fare_by_pclass[pclass]
        return stats.fare_median

    def _is_child_row(self, row: pd.Series) -> bool:
        cfg = self.cfg
        if str(row.get("Title", "")) == "Master":
            return True
        age = row.get("Age")
        if pd.notna(age) and float(age) < cfg.child_age_max:
            return True
        return False

    @staticmethod
    def _coarse_deck(deck: str) -> str:
        if deck == "U":
            return "Unknown"
        if deck in {"A", "B", "C"}:
            return "ABC"
        if deck in {"D", "E"}:
            return "DE"
        return "FG"

    def _parse_surname(self, name: str) -> str:
        try:
            return str(name).split(",")[0].strip()
        except (IndexError, AttributeError):
            return "Unknown"

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
        if ch in cfg.valid_deck_letters:
            return ch
        return cfg.unknown_deck
