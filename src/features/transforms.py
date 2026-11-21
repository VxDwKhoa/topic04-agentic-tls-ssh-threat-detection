"""
Train-only feature sanitization, quantile clipping, scaling, and target label encoding
for NF-UQ-NIDS-v2 L3/L4 metadata features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.data_loader import L3_L4_FEATURE_COLUMNS


class FlowFeatureTransformer:
    """
    Sanitizes `+inf`/`-inf`/`NaN` in float64, imputes with Train-only medians,
    clips extremes to Train-only quantile bounds, and scales via Train-only StandardScaler.
    """

    def __init__(
        self,
        feature_cols: Sequence[str] = L3_L4_FEATURE_COLUMNS,
        clip_quantiles: Tuple[float, float] = (0.0, 0.9999),
        output_dtype: str = "float32",
    ) -> None:
        if len(feature_cols) == 0:
            raise ValueError("feature_cols must not be empty.")
        if len(clip_quantiles) != 2 or not (0.0 <= clip_quantiles[0] <= clip_quantiles[1] <= 1.0):
            raise ValueError(
                f"clip_quantiles must satisfy 0.0 <= low <= high <= 1.0, got {clip_quantiles}"
            )
        if output_dtype not in ("float32", "float64"):
            raise ValueError(f"output_dtype must be 'float32' or 'float64', got '{output_dtype}'")

        self.feature_cols: Tuple[str, ...] = tuple(feature_cols)
        self.clip_quantiles: Tuple[float, float] = (float(clip_quantiles[0]), float(clip_quantiles[1]))
        self.output_dtype: str = output_dtype

        self.scaler_: StandardScaler = StandardScaler()
        self.medians_: Optional[np.ndarray] = None
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required feature columns: {missing}")

    def _extract_float64(self, df: pd.DataFrame) -> np.ndarray:
        self._validate_columns(df)
        cols = list(self.feature_cols)
        sub = df[cols]
        arrays = []
        for col in cols:
            series = sub[col]
            if pd.api.types.is_numeric_dtype(series):
                arr = series.to_numpy(dtype=np.float64, copy=True)
            else:
                arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=True)
            arrays.append(arr)
        X = np.column_stack(arrays).astype(np.float64, copy=False)
        return np.where(np.isfinite(X), X, np.nan)

    def fit(self, df_train: pd.DataFrame) -> "FlowFeatureTransformer":
        if len(df_train) == 0:
            raise ValueError("Cannot fit FlowFeatureTransformer on an empty DataFrame.")

        X = self._extract_float64(df_train)
        n_features = X.shape[1]

        medians = np.zeros(n_features, dtype=np.float64)
        for idx in range(n_features):
            col_vals = X[:, idx]
            finite_mask = np.isfinite(col_vals)
            if np.any(finite_mask):
                medians[idx] = float(np.median(col_vals[finite_mask]))
            else:
                medians[idx] = 0.0
        self.medians_ = medians

        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X = np.where(nan_mask, self.medians_, X)

        q_low, q_high = self.clip_quantiles
        self.lower_bounds_ = np.quantile(X, q_low, axis=0).astype(np.float64)
        self.upper_bounds_ = np.quantile(X, q_high, axis=0).astype(np.float64)

        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        self.scaler_.scale_ = np.where(self.scaler_.scale_ < 1e-12, 1.0, self.scaler_.scale_)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("FlowFeatureTransformer must be fitted before calling transform().")
        self._validate_columns(df)

        out_dtype = np.dtype(self.output_dtype)
        if len(df) == 0:
            return np.empty((0, len(self.feature_cols)), dtype=out_dtype)

        assert self.medians_ is not None
        assert self.lower_bounds_ is not None
        assert self.upper_bounds_ is not None

        X = self._extract_float64(df)
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X = np.where(nan_mask, self.medians_, X)

        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        X = (X - self.scaler_.mean_) / self.scaler_.scale_
        X_out = X.astype(out_dtype, copy=False)

        if not np.all(np.isfinite(X_out)):
            raise ValueError("Non-finite values detected after FlowFeatureTransformer.transform().")

        return X_out

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        return self.fit(df_train).transform(df_train)

    def get_state(self) -> Dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("Cannot serialize an unfitted FlowFeatureTransformer.")
        assert self.medians_ is not None
        assert self.lower_bounds_ is not None
        assert self.upper_bounds_ is not None

        return {
            "feature_cols": list(self.feature_cols),
            "clip_quantiles": [float(self.clip_quantiles[0]), float(self.clip_quantiles[1])],
            "output_dtype": self.output_dtype,
            "medians": [float(x) for x in self.medians_],
            "lower_bounds": [float(x) for x in self.lower_bounds_],
            "upper_bounds": [float(x) for x in self.upper_bounds_],
            "mean": [float(x) for x in self.scaler_.mean_],
            "scale": [float(x) for x in self.scaler_.scale_],
            "var": [float(x) for x in self.scaler_.var_],
            "n_samples_seen": int(self.scaler_.n_samples_seen_),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "FlowFeatureTransformer":
        obj = cls(
            feature_cols=state["feature_cols"],
            clip_quantiles=(float(state["clip_quantiles"][0]), float(state["clip_quantiles"][1])),
            output_dtype=str(state.get("output_dtype", "float32")),
        )
        obj.medians_ = np.asarray(state["medians"], dtype=np.float64)
        obj.lower_bounds_ = np.asarray(state["lower_bounds"], dtype=np.float64)
        obj.upper_bounds_ = np.asarray(state["upper_bounds"], dtype=np.float64)

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
        scaler.scale_ = np.where(scaler.scale_ < 1e-12, 1.0, scaler.scale_)
        scaler.var_ = np.asarray(state["var"], dtype=np.float64)
        scaler.n_features_in_ = len(obj.feature_cols)
        scaler.n_samples_seen_ = int(state.get("n_samples_seen", 0))
        obj.scaler_ = scaler
        obj.is_fitted_ = True
        return obj


class TargetLabelEncoder:
    """
    Deterministic multi-class label encoder fitted on Train `attack_family` classes,
    mapping unseen or missing labels safely to `unknown_index` (-1) / `unknown_label`.
    """

    def __init__(self, unknown_index: int = -1, unknown_label: str = "UNKNOWN_UNSEEN") -> None:
        self.unknown_index: int = int(unknown_index)
        self.unknown_label: str = str(unknown_label)
        self.classes_: List[str] = []
        self.class_to_idx_: Dict[str, int] = {}
        self.idx_to_class_: Dict[int, str] = {}
        self.is_fitted_: bool = False

    def fit(self, y: Union[pd.Series, Sequence[Any], np.ndarray]) -> "TargetLabelEncoder":
        values = list(y)
        if len(values) == 0:
            raise ValueError("Cannot fit TargetLabelEncoder on empty labels.")

        unique_classes = sorted({str(v) for v in values if pd.notna(v)})
        if not unique_classes:
            raise ValueError("TargetLabelEncoder found no non-null label classes to fit.")

        self.classes_ = unique_classes
        self.class_to_idx_ = {cls_name: idx for idx, cls_name in enumerate(self.classes_)}
        self.idx_to_class_ = {idx: cls_name for idx, cls_name in enumerate(self.classes_)}
        self.is_fitted_ = True
        return self

    def transform(self, y: Union[pd.Series, Sequence[Any], np.ndarray]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("TargetLabelEncoder must be fitted before calling transform().")

        encoded = [
            self.class_to_idx_.get(str(v), self.unknown_index) if pd.notna(v) else self.unknown_index
            for v in y
        ]
        return np.asarray(encoded, dtype=np.int64)

    def fit_transform(self, y: Union[pd.Series, Sequence[Any], np.ndarray]) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, indices: Union[Sequence[int], np.ndarray]) -> List[str]:
        if not self.is_fitted_:
            raise RuntimeError("TargetLabelEncoder must be fitted before calling inverse_transform().")
        return [self.idx_to_class_.get(int(idx), self.unknown_label) for idx in indices]

    @property
    def num_classes(self) -> int:
        return len(self.classes_)

    def known_mask(self, y_encoded: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        return np.asarray(y_encoded, dtype=np.int64) != self.unknown_index

    def get_state(self) -> Dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("Cannot serialize an unfitted TargetLabelEncoder.")
        return {
            "classes": list(self.classes_),
            "unknown_index": self.unknown_index,
            "unknown_label": self.unknown_label,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TargetLabelEncoder":
        obj = cls(
            unknown_index=int(state.get("unknown_index", -1)),
            unknown_label=str(state.get("unknown_label", "UNKNOWN_UNSEEN")),
        )
        obj.classes_ = [str(c) for c in state["classes"]]
        obj.class_to_idx_ = {cls_name: idx for idx, cls_name in enumerate(obj.classes_)}
        obj.idx_to_class_ = {idx: cls_name for idx, cls_name in enumerate(obj.classes_)}
        obj.is_fitted_ = True
        return obj
