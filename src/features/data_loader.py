"""
Split-aware Parquet data loader & Topic 04 No-L7 privacy audit integration
for NF-UQ-NIDS-v2 (full dataset and sample fallback).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.utils.config import ResolvedPaths, discover_repo_root

L3_L4_FEATURE_COLUMNS: Tuple[str, ...] = (
    "src_port_class",
    "dst_port",
    "proto",
    "in_bytes",
    "in_pkts",
    "out_bytes",
    "out_pkts",
    "tcp_flags",
    "client_tcp_flags",
    "server_tcp_flags",
    "flow_duration_ms",
    "duration_in",
    "duration_out",
    "min_ttl",
    "max_ttl",
    "longest_flow_pkt",
    "shortest_flow_pkt",
    "min_ip_pkt_len",
    "max_ip_pkt_len",
    "src_to_dst_second_bytes",
    "dst_to_src_second_bytes",
    "retransmitted_in_bytes",
    "retransmitted_in_pkts",
    "retransmitted_out_bytes",
    "retransmitted_out_pkts",
    "src_to_dst_avg_throughput",
    "dst_to_src_avg_throughput",
    "num_pkts_up_to_128_bytes",
    "num_pkts_128_to_256_bytes",
    "num_pkts_256_to_512_bytes",
    "num_pkts_512_to_1024_bytes",
    "num_pkts_1024_to_1514_bytes",
    "tcp_win_max_in",
    "tcp_win_max_out",
    "icmp_type",
    "icmp_ipv4_type",
    "is_tls_port",
    "is_ssh_port",
    "bytes_ratio",
    "pkts_ratio",
)

UNSEEN_ATTACK_FAMILIES: FrozenSet[str] = frozenset({"Worms", "Shellcode", "Ransomware"})

REQUIRED_METADATA_COLUMNS: Tuple[str, ...] = (
    "flow_order_id",
    "src_host_token",
    "dst_host_token",
    "label",
    "attack_family",
)

IPV4_REGEX = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")


def _import_privacy_validator(repo_root: Optional[Path] = None) -> Callable[..., Dict[str, Any]]:
    """Robustly import `validate_dataframe_privacy` from `data/schemas/privacy_validator.py`."""
    try:
        from data.schemas.privacy_validator import validate_dataframe_privacy

        return validate_dataframe_privacy
    except ImportError:
        pass

    if "privacy_validator" in sys.modules:
        mod = sys.modules["privacy_validator"]
        if hasattr(mod, "validate_dataframe_privacy"):
            return getattr(mod, "validate_dataframe_privacy")

    candidate_roots = []
    if repo_root is not None:
        candidate_roots.append(repo_root)
    discovered = discover_repo_root()
    if discovered is not None and discovered not in candidate_roots:
        candidate_roots.append(discovered)
    fallback_root = Path(__file__).resolve().parents[2]
    if fallback_root not in candidate_roots:
        candidate_roots.append(fallback_root)

    for root in candidate_roots:
        validator_path = root / "data" / "schemas" / "privacy_validator.py"
        if validator_path.is_file():
            spec = importlib.util.spec_from_file_location("topic04_privacy_validator", validator_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, "validate_dataframe_privacy")

    raise ImportError("Could not locate data/schemas/privacy_validator.py to enforce No-L7 audit.")


class NFUQDataLoader:
    """Split-aware Parquet loader for NF-UQ-NIDS-v2 enforcing Topic 04 No-L7 privacy rules."""

    def __init__(
        self,
        paths: ResolvedPaths,
        schema_path: Optional[Union[str, Path]] = None,
        feature_cols: Sequence[str] = L3_L4_FEATURE_COLUMNS,
        strict_privacy: bool = True,
    ) -> None:
        self.paths = paths
        if schema_path is not None:
            self.schema_path: Optional[Path] = Path(schema_path)
        elif paths.schema_path is not None:
            self.schema_path = paths.schema_path
        elif paths.repo_root is not None:
            candidate = paths.repo_root / "data" / "schemas" / "schema_allowlist.json"
            self.schema_path = candidate if candidate.is_file() else None
        else:
            self.schema_path = None

        self.feature_cols: Tuple[str, ...] = tuple(feature_cols)
        self.strict_privacy = strict_privacy
        self.privacy_reports: Dict[str, Dict[str, Any]] = {}
        self._validate_fn = _import_privacy_validator(paths.repo_root)

    def validate_privacy(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """Run Topic 04 No-L7 privacy audit, full-column IPv4 check, and feature column check."""
        report = self._validate_fn(
            df,
            schema_path=str(self.schema_path) if self.schema_path is not None else None,
            dataset_name=dataset_name,
        )

        # Full-column unique check on pseudonymized host tokens (privacy_validator only checks head(500))
        violations = list(report.get("violations", []))
        for host_col in ("src_host_token", "dst_host_token"):
            if host_col in df.columns:
                unique_vals = df[host_col].dropna().astype(str).unique()
                raw_ips = [v for v in unique_vals if IPV4_REGEX.match(v)]
                if raw_ips:
                    already_flagged = any(host_col in msg for msg in violations)
                    if not already_flagged:
                        violations.append(
                            f"Column '{host_col}' contains raw un-pseudonymized IPv4 addresses (e.g. {raw_ips[0]})."
                        )

        report["violations"] = violations
        report["passed_no_l7_audit"] = len(violations) == 0
        self.privacy_reports[dataset_name] = report

        if self.strict_privacy and not report["passed_no_l7_audit"]:
            raise ValueError(f"Privacy audit failed for {dataset_name}: {report['violations']}")

        missing_features = [c for c in self.feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required feature columns: {missing_features}"
            )

        return report

    @staticmethod
    def _sort_by_flow_order(df: pd.DataFrame) -> pd.DataFrame:
        if "flow_order_id" in df.columns and len(df) > 0:
            return df.sort_values("flow_order_id", kind="mergesort").reset_index(drop=True)
        return df.reset_index(drop=True)

    @classmethod
    def _split_sample_temporal(cls, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Deterministic temporal + unseen-attack split for sample fallback mode:
        1. Separate UNSEEN_ATTACK_FAMILIES first.
        2. Sort remaining known rows by flow_order_id ascending (kind='mergesort').
        3. Slice 70% train / 15% val / 15% test_known.
        4. Combine test_known + unseen sorted by flow_order_id for df_test.
        """
        if "attack_family" in df_raw.columns:
            unseen_mask = df_raw["attack_family"].astype(str).isin(UNSEEN_ATTACK_FAMILIES)
        else:
            unseen_mask = pd.Series(False, index=df_raw.index)

        df_unseen = cls._sort_by_flow_order(df_raw.loc[unseen_mask].copy())
        df_known = cls._sort_by_flow_order(df_raw.loc[~unseen_mask].copy())

        n = len(df_known)
        if n == 0:
            df_train = df_known.copy()
            df_val = df_known.iloc[0:0].copy()
            df_test_known = df_known.iloc[0:0].copy()
        elif n == 1:
            df_train = df_known.iloc[:1].reset_index(drop=True)
            df_val = df_known.iloc[0:0].copy()
            df_test_known = df_known.iloc[0:0].copy()
        elif n == 2:
            df_train = df_known.iloc[:1].reset_index(drop=True)
            df_val = df_known.iloc[1:2].reset_index(drop=True)
            df_test_known = df_known.iloc[0:0].copy()
        else:
            n_train = max(1, int(n * 0.70))
            n_val = max(1, int(n * 0.15))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            df_train = df_known.iloc[:n_train].reset_index(drop=True)
            df_val = df_known.iloc[n_train : n_train + n_val].reset_index(drop=True)
            df_test_known = df_known.iloc[n_train + n_val :].reset_index(drop=True)

        df_test = cls._sort_by_flow_order(
            pd.concat([df_test_known, df_unseen], axis=0, ignore_index=True)
        )
        return df_train, df_val, df_test, df_unseen

    def load_tabular_splits(
        self,
        max_train_rows: Optional[int] = None,
        max_eval_rows: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load train, val, test, and unseen_holdout DataFrames with mandatory privacy audits."""
        if max_train_rows is not None and max_train_rows <= 0:
            raise ValueError(f"max_train_rows must be > 0 or None, got {max_train_rows}")
        if max_eval_rows is not None and max_eval_rows <= 0:
            raise ValueError(f"max_eval_rows must be > 0 or None, got {max_eval_rows}")

        if not self.paths.is_sample_mode:
            train_path = self.paths.split_tabular_path("train")
            val_path = self.paths.split_tabular_path("val")
            test_path = self.paths.split_tabular_path("test")
            unseen_path = self.paths.unseen_holdout_path

            assert train_path is not None and val_path is not None and test_path is not None
            df_train = pd.read_parquet(train_path)
            df_val = pd.read_parquet(val_path)
            df_test = pd.read_parquet(test_path)

            if unseen_path is not None and unseen_path.is_file():
                df_unseen = pd.read_parquet(unseen_path)
            elif "attack_family" in df_test.columns:
                df_unseen = df_test[
                    df_test["attack_family"].astype(str).isin(UNSEEN_ATTACK_FAMILIES)
                ].copy()
            else:
                df_unseen = df_test.iloc[0:0].copy()
        else:
            if self.paths.sample_path is None or not self.paths.sample_path.is_file():
                raise FileNotFoundError(f"Sample parquet not found: {self.paths.sample_path}")
            df_raw = pd.read_parquet(self.paths.sample_path)
            self.validate_privacy(df_raw, "sample_raw")
            df_train, df_val, df_test, df_unseen = self._split_sample_temporal(df_raw)

        if max_train_rows is not None:
            df_train = df_train.iloc[:max_train_rows].copy()
        if max_eval_rows is not None:
            df_val = df_val.iloc[:max_eval_rows].copy()
            df_test = df_test.iloc[:max_eval_rows].copy()
            df_unseen = df_unseen.iloc[:max_eval_rows].copy()

        splits = {
            "train": df_train,
            "val": df_val,
            "test": df_test,
            "unseen_holdout": df_unseen,
        }
        for name, df_split in splits.items():
            self.validate_privacy(df_split, name)

        return splits

    def load_temporal_sequences(
        self,
        split: str = "train",
        windows: Optional[Sequence[int]] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load temporal sequence flow DataFrames sorted by (src_host_token, flow_order_id)."""
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train', 'val', or 'test'.")
        if max_rows is not None and max_rows <= 0:
            raise ValueError(f"max_rows must be > 0 or None, got {max_rows}")

        if self.paths.is_sample_mode:
            df = self.load_tabular_splits()[split]
        else:
            seq_dir = self.paths.sequences_dir(split)
            if seq_dir is not None and seq_dir.is_dir():
                window_files: List[Tuple[int, Path]] = []
                for p in sorted(seq_dir.glob("window_*/sequences.parquet")):
                    win_name = p.parent.name
                    try:
                        win_idx = int(win_name.split("_")[-1])
                    except ValueError:
                        win_idx = len(window_files)
                    window_files.append((win_idx, p))
                window_files.sort(key=lambda item: item[0])

                if windows is not None:
                    allowed = set(int(w) for w in windows)
                    window_files = [item for item in window_files if item[0] in allowed]

                if window_files:
                    frames = [pd.read_parquet(p) for _, p in window_files]
                    df = pd.concat(frames, axis=0, ignore_index=True)
                elif windows is not None:
                    raise FileNotFoundError(
                        f"No matching sequence window files found in {seq_dir} for windows={list(windows)}"
                    )
                else:
                    tab_path = self.paths.split_tabular_path(split)
                    assert tab_path is not None
                    df = pd.read_parquet(tab_path)
            else:
                tab_path = self.paths.split_tabular_path(split)
                assert tab_path is not None
                df = pd.read_parquet(tab_path)

        sort_cols = [c for c in ("src_host_token", "flow_order_id") if c in df.columns]
        if sort_cols and len(df) > 0:
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if max_rows is not None:
            df = df.iloc[:max_rows].copy()

        self.validate_privacy(df, f"{split}_sequences")
        return df

    def get_metadata_manifest(self) -> Optional[Dict[str, Any]]:
        """Read metadata_manifest.json if present (for provenance inspection only)."""
        if self.paths.manifest_path is not None and self.paths.manifest_path.is_file():
            with open(self.paths.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
