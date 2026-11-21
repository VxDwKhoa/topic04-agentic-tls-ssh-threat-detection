"""
PyTorch Dataset abstractions for tabular L3/L4 flow modeling (Weeks 5-6)
and pure temporal sequence modeling (Week 7, no graph features).
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.data_loader import L3_L4_FEATURE_COLUMNS
from src.features.transforms import FlowFeatureTransformer, TargetLabelEncoder

FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "src_host_token",
        "dst_host_token",
        "session_id",
        "flow_order_id",
        "window_id",
        "time_bucket",
        "label",
        "attack_family",
        "attack_label",
        "attack_layers",
        "attack_mapping",
        "ground_truth_source",
    }
)

_FORBIDDEN_L7_PATTERNS = (
    re.compile(r"^http_"),
    re.compile(r"^ssh_"),
    re.compile(r"^decrypted_"),
    re.compile(r"^tls_session_key"),
    re.compile(r"^private_key"),
    re.compile(r"^raw_pcap"),
    re.compile(r"^payload"),
    re.compile(r"^l7_proto$"),
    re.compile(r"^dns_query_"),
    re.compile(r"^dns_ttl_"),
    re.compile(r"^ftp_command_"),
    re.compile(r"^ipv4_(src|dst)_addr$"),
    re.compile(r"^raw_s(ni|an)$"),
)


def _validate_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    cols = list(feature_cols)
    if not cols:
        raise ValueError("feature_cols must not be empty.")

    forbidden_hits = [c for c in cols if c in FORBIDDEN_FEATURE_COLUMNS]
    if forbidden_hits:
        raise ValueError(
            f"feature_cols contains forbidden identity/metadata/label columns: {forbidden_hits}"
        )

    l7_hits = [c for c in cols if any(p.search(c.lower()) for p in _FORBIDDEN_L7_PATTERNS)]
    if l7_hits:
        raise ValueError(f"feature_cols contains forbidden L7/raw-IP columns: {l7_hits}")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing requested feature columns: {missing}")

    return cols


def _extract_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    transformer: Optional[FlowFeatureTransformer],
) -> np.ndarray:
    if len(df) == 0:
        return np.empty((0, len(feature_cols)), dtype=np.float32)

    if transformer is not None:
        arr = transformer.transform(df)
        arr = np.asarray(arr, dtype=np.float32)
    else:
        arrays = []
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                col_arr = series.to_numpy(dtype=np.float64, copy=True)
            else:
                col_arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=True)
            arrays.append(col_arr)
        X64 = np.column_stack(arrays).astype(np.float64, copy=False)
        X64 = np.where(np.isfinite(X64), X64, 0.0)
        arr = X64.astype(np.float32, copy=False)

    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values detected in extracted dataset features.")
    return arr


class TabularFlowDataset(Dataset):
    """PyTorch Dataset wrapping sanitized L3/L4 tabular flow features and optional targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        transformer: Optional[FlowFeatureTransformer] = None,
        label_col: Optional[str] = "label",
        target_encoder: Optional[TargetLabelEncoder] = None,
        feature_cols: Sequence[str] = L3_L4_FEATURE_COLUMNS,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.feature_cols: List[str] = _validate_feature_columns(df, feature_cols)
        self.num_features: int = len(self.feature_cols)
        self.num_samples: int = int(len(df))
        self.label_col = label_col

        if len(df) == 0:
            self.X: torch.Tensor = torch.empty((0, self.num_features), dtype=dtype)
            if label_col is not None and label_col in df.columns:
                self.y: Optional[torch.Tensor] = torch.empty((0,), dtype=torch.long)
            else:
                self.y = None
            return

        X_arr = _extract_features(df, self.feature_cols, transformer)
        self.X = torch.from_numpy(X_arr).to(dtype=dtype)

        if label_col is not None and label_col in df.columns:
            if target_encoder is not None:
                y_arr = target_encoder.transform(df[label_col])
            else:
                y_arr = pd.to_numeric(df[label_col], errors="raise").to_numpy(dtype=np.int64)
            self.y = torch.from_numpy(np.array(y_arr, dtype=np.int64, copy=True))
        else:
            self.y = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "feature_cols": list(self.feature_cols),
            "has_labels": self.y is not None,
        }


class TemporalSequenceDataset(Dataset):
    """
    Pure L3/L4 temporal sequence Dataset grouped strictly by `src_host_token`
    and ordered chronologically by `flow_order_id` (no graph features, no padding).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transformer: Optional[FlowFeatureTransformer] = None,
        seq_len: int = 8,
        stride: int = 4,
        feature_cols: Sequence[str] = L3_L4_FEATURE_COLUMNS,
        group_col: str = "src_host_token",
        order_col: str = "flow_order_id",
        label_col: Optional[str] = "label",
        time_bucket_col: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.seq_len: int = int(seq_len)
        self.stride: int = int(stride)
        self.group_col: str = group_col
        self.order_col: str = order_col
        self.label_col: Optional[str] = label_col
        self.time_bucket_col: Optional[str] = time_bucket_col
        self.feature_cols: List[str] = _validate_feature_columns(df, feature_cols)
        self.num_features: int = len(self.feature_cols)

        if group_col not in df.columns:
            raise KeyError(f"Grouping column '{group_col}' not found in DataFrame.")
        if order_col not in df.columns:
            raise KeyError(f"Ordering column '{order_col}' not found in DataFrame.")
        if time_bucket_col is not None and time_bucket_col not in df.columns:
            raise KeyError(f"Time-bucket column '{time_bucket_col}' not found in DataFrame.")

        self.num_hosts: int = 0
        self.num_active_hosts: int = 0
        self.num_skipped_hosts: int = 0
        self.num_skipped_flows: int = 0
        self.sequence_host_tokens: List[Any] = []

        has_labels = label_col is not None and label_col in df.columns

        if len(df) == 0:
            self.X: torch.Tensor = torch.empty((0, self.seq_len, self.num_features), dtype=dtype)
            self.y: Optional[torch.Tensor] = torch.empty((0,), dtype=torch.long) if has_labels else None
            self.num_sequences: int = 0
            warnings.warn(
                "TemporalSequenceDataset received an empty DataFrame; returning 0 sequences.",
                UserWarning,
                stacklevel=2,
            )
            return

        X_all = _extract_features(df, self.feature_cols, transformer)
        y_all = (
            pd.to_numeric(df[label_col], errors="raise").to_numpy(dtype=np.int64)
            if has_labels and label_col is not None
            else None
        )

        sort_cols = [group_col, order_col] if time_bucket_col is None else [group_col, time_bucket_col, order_col]
        meta_df = df[sort_cols].copy()
        meta_df["_pos"] = np.arange(len(df), dtype=np.int64)
        meta_sorted = meta_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        sort_pos = meta_sorted["_pos"].to_numpy(dtype=np.int64)
        X_sorted = X_all[sort_pos]
        y_sorted = y_all[sort_pos] if y_all is not None else None

        group_keys: Union[str, List[str]] = group_col if time_bucket_col is None else [group_col, time_bucket_col]
        grouped = meta_sorted.groupby(group_keys, sort=False, observed=True)

        X_windows: List[np.ndarray] = []
        y_windows: List[int] = []

        for key, idx_arr in grouped.indices.items():
            n = len(idx_arr)
            self.num_hosts += 1
            if n < self.seq_len:
                self.num_skipped_hosts += 1
                self.num_skipped_flows += n
                continue

            self.num_active_hosts += 1
            host_token = key[0] if isinstance(key, tuple) else key
            for start in range(0, n - self.seq_len + 1, self.stride):
                win_idx = idx_arr[start : start + self.seq_len]
                X_windows.append(X_sorted[win_idx])
                if y_sorted is not None:
                    y_windows.append(int(np.max(y_sorted[win_idx])))
                self.sequence_host_tokens.append(host_token)

        if not X_windows:
            self.X = torch.empty((0, self.seq_len, self.num_features), dtype=dtype)
            self.y = torch.empty((0,), dtype=torch.long) if y_sorted is not None else None
            self.num_sequences = 0
            warnings.warn(
                f"TemporalSequenceDataset produced 0 sequences for seq_len={self.seq_len}, "
                f"stride={self.stride} (num_hosts={self.num_hosts}, skipped={self.num_skipped_hosts}).",
                UserWarning,
                stacklevel=2,
            )
            return

        stacked = np.stack(X_windows, axis=0).astype(np.float32, copy=False)
        self.X = torch.from_numpy(stacked).to(dtype=dtype)
        self.y = torch.tensor(y_windows, dtype=torch.long) if y_sorted is not None else None
        self.num_sequences = int(self.X.shape[0])

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "stride": self.stride,
            "num_features": self.num_features,
            "num_sequences": self.num_sequences,
            "num_hosts": self.num_hosts,
            "num_active_hosts": self.num_active_hosts,
            "num_skipped_hosts": self.num_skipped_hosts,
            "num_skipped_flows": self.num_skipped_flows,
            "group_col": self.group_col,
            "order_col": self.order_col,
            "time_bucket_col": self.time_bucket_col,
            "label_aggregation": "max",
        }

    def get_class_distribution(self) -> Dict[int, int]:
        if self.y is None or self.num_sequences == 0:
            return {}
        unique_vals, counts = torch.unique(self.y, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique_vals.tolist(), counts.tolist())}
