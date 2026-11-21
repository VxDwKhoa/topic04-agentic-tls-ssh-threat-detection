"""
Shared fixtures, canonical schema constants, and opaque-box test helpers
for the Topic 04 E2E test suite.

Collection-safety rule: NEVER import `src.*` at module top-level here.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import inspect
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import nbformat
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
LOCAL_FULL_DATASET_DIR = WORKSPACE_ROOT / "NF-UQ-NIDS-v2"
SAMPLE_PARQUET_PATH = REPO_ROOT / "data" / "public_loaders" / "sample_data" / "nfuq_sample_1500.parquet"
MANIFEST_AUDIT_PATH = REPO_ROOT / "data" / "public_loaders" / "sample_data" / "nfuq_manifest_and_audit.json"
SCHEMA_ALLOWLIST_PATH = REPO_ROOT / "data" / "schemas" / "schema_allowlist.json"
NOTEBOOK_DIR = WORKSPACE_ROOT / "Notebook"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_LOCAL_CHECKPOINT_DIR = NOTEBOOK_DIR / "Checkpoint"

KAGGLE_DATASET_DIR = "/kaggle/input/datasets/vxdwkhoa/Dataset_NF/NF-UQ-NIDS-v2"
KAGGLE_CHECKPOINT_DIR = "/kaggle/working/Checkpoint"

NOTEBOOK_FILENAMES: Tuple[str, ...] = (
    "week5_rules_random_forest.ipynb",
    "week6_isolation_forest_anomaly.ipynb",
    "week7_pytorch_lstm_temporal.ipynb",
)

EXPECTED_L3_L4_FEATURE_COLUMNS: List[str] = [
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
]

EXPECTED_IDENTITY_COLUMNS: List[str] = [
    "session_id",
    "flow_order_id",
    "window_id",
    "src_host_token",
    "dst_host_token",
]

EXPECTED_LABEL_COLUMNS: List[str] = [
    "label",
    "attack_family",
    "attack_label",
    "attack_layers",
    "attack_mapping",
    "ground_truth_source",
]

IPV4_PATTERN = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
HOST_TOKEN_PATTERN = re.compile(r"^host_[0-9a-f]{16}$")


def ensure_repo_on_sys_path() -> None:
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def import_privacy_validator():
    schemas_dir = str(REPO_ROOT / "data" / "schemas")
    if schemas_dir not in sys.path:
        sys.path.insert(0, schemas_dir)
    return importlib.import_module("privacy_validator")


def import_preprocess_module():
    loaders_dir = str(REPO_ROOT / "data" / "public_loaders")
    if loaders_dir not in sys.path:
        sys.path.insert(0, loaders_dir)
    return importlib.import_module("preprocess_and_split_nfuq")


def import_src_module(module_path: str):
    ensure_repo_on_sys_path()
    return importlib.import_module(module_path)


def discover_class(
    module: Any,
    candidate_names: Sequence[str] = (),
    base_cls: Optional[Type[Any]] = None,
) -> Type[Any]:
    """Locate an OOP class in a module by canonical names or subclass relationship."""
    for name in candidate_names:
        obj = getattr(module, name, None)
        if inspect.isclass(obj):
            return obj

    if base_cls is not None:
        matches = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if obj is not base_cls and issubclass(obj, base_cls) and obj.__module__ == module.__name__
        ]
        if matches:
            return matches[0]

    raise AssertionError(
        f"Expected OOP class {list(candidate_names)} (or subclass of {base_cls}) in {module.__name__}"
    )


@contextmanager
def temporary_env_vars(updates: Dict[str, Optional[str]]):
    previous: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, old_v in previous.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v


def _pseudonym_token(seed_text: str) -> str:
    digest = hmac.new(b"topic04-e2e-synthetic-test-key", seed_text.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"host_{digest[:16]}"


def make_synthetic_l3_l4_flows(
    num_rows: int = 120,
    seed: int = 42,
    attack_families: Sequence[str] = ("Benign", "DoS", "Brute Force"),
    benign_ratio: float = 0.5,
    start_order_id: int = 1,
    num_hosts: int = 4,
    host_prefix: str = "synth_src",
    include_time_bucket: bool = True,
) -> pd.DataFrame:
    """
    Create a deterministic, strictly L3/L4 compliant synthetic DataFrame
    with zero L7 fields and zero raw IPv4 addresses.
    """
    rng = np.random.default_rng(seed)
    n_benign = int(round(num_rows * benign_ratio)) if "Benign" in attack_families else 0
    non_benign = [fam for fam in attack_families if fam != "Benign"]

    families: List[str] = []
    labels: List[int] = []
    for idx in range(num_rows):
        if idx < n_benign or not non_benign:
            families.append("Benign")
            labels.append(0)
        else:
            fam = non_benign[(idx - n_benign) % len(non_benign)]
            families.append(fam)
            labels.append(1)

    # Shuffle assignment deterministically while keeping flow_order_id strictly monotonic
    perm = rng.permutation(num_rows)
    families = [families[i] for i in perm]
    labels = [labels[i] for i in perm]

    src_hosts = [_pseudonym_token(f"{host_prefix}_{h}") for h in range(max(1, num_hosts))]
    dst_hosts = [_pseudonym_token(f"{host_prefix}_dst_{h}") for h in range(max(1, min(3, num_hosts)))]

    data: Dict[str, Any] = {
        "session_id": [f"sess_{host_prefix}_{start_order_id + i}" for i in range(num_rows)],
        "flow_order_id": np.arange(start_order_id, start_order_id + num_rows, dtype=np.int64),
        "window_id": (np.arange(num_rows, dtype=np.int32) // max(1, num_rows // 4)).astype(np.int32),
        "src_host_token": [src_hosts[i % len(src_hosts)] for i in range(num_rows)],
        "dst_host_token": [dst_hosts[i % len(dst_hosts)] for i in range(num_rows)],
    }

    is_attack = np.array(labels, dtype=np.int64)
    # Give attacks distinguishable L3/L4 patterns (higher pkts/retrans/SYN burst, shorter duration)
    data["src_port_class"] = rng.integers(0, 3, size=num_rows, dtype=np.int8)
    data["dst_port"] = np.where(is_attack == 1, rng.choice([22, 443, 80, 445], size=num_rows), 443).astype(np.int32)
    data["proto"] = np.full(num_rows, 6, dtype=np.int64)
    data["in_bytes"] = np.where(is_attack == 1, rng.integers(4000, 40000, size=num_rows), rng.integers(300, 3000, size=num_rows)).astype(np.int64)
    data["in_pkts"] = np.where(is_attack == 1, rng.integers(40, 250, size=num_rows), rng.integers(3, 25, size=num_rows)).astype(np.int64)
    data["out_bytes"] = np.where(is_attack == 1, rng.integers(60, 800, size=num_rows), rng.integers(500, 5000, size=num_rows)).astype(np.int64)
    data["out_pkts"] = np.where(is_attack == 1, rng.integers(1, 10, size=num_rows), rng.integers(4, 30, size=num_rows)).astype(np.int64)
    data["tcp_flags"] = np.where(is_attack == 1, 22, 24).astype(np.int64)
    data["client_tcp_flags"] = np.where(is_attack == 1, 2, 24).astype(np.int64)
    data["server_tcp_flags"] = np.where(is_attack == 1, 20, 24).astype(np.int64)
    data["flow_duration_ms"] = np.where(is_attack == 1, rng.integers(1, 50, size=num_rows), rng.integers(100, 2500, size=num_rows)).astype(np.int64)
    data["duration_in"] = (data["flow_duration_ms"] // 2).astype(np.int64)
    data["duration_out"] = (data["flow_duration_ms"] - data["duration_in"]).astype(np.int64)
    data["min_ttl"] = rng.integers(50, 65, size=num_rows, dtype=np.int64)
    data["max_ttl"] = (data["min_ttl"] + rng.integers(0, 5, size=num_rows)).astype(np.int64)
    data["longest_flow_pkt"] = rng.integers(400, 1500, size=num_rows, dtype=np.int64)
    data["shortest_flow_pkt"] = rng.integers(40, 120, size=num_rows, dtype=np.int64)
    data["min_ip_pkt_len"] = data["shortest_flow_pkt"]
    data["max_ip_pkt_len"] = data["longest_flow_pkt"]
    data["src_to_dst_second_bytes"] = (data["in_bytes"] / np.maximum(1.0, data["flow_duration_ms"] / 1000.0)).astype(np.float32)
    data["dst_to_src_second_bytes"] = (data["out_bytes"] / np.maximum(1.0, data["flow_duration_ms"] / 1000.0)).astype(np.float32)
    data["retransmitted_in_bytes"] = np.where(is_attack == 1, rng.integers(200, 1500, size=num_rows), 0).astype(np.int64)
    data["retransmitted_in_pkts"] = np.where(is_attack == 1, rng.integers(2, 15, size=num_rows), 0).astype(np.int64)
    data["retransmitted_out_bytes"] = np.zeros(num_rows, dtype=np.int64)
    data["retransmitted_out_pkts"] = np.zeros(num_rows, dtype=np.int64)
    data["src_to_dst_avg_throughput"] = (data["in_bytes"] * 8.0 / np.maximum(1.0, data["flow_duration_ms"])).astype(np.float32)
    data["dst_to_src_avg_throughput"] = (data["out_bytes"] * 8.0 / np.maximum(1.0, data["flow_duration_ms"])).astype(np.float32)
    data["num_pkts_up_to_128_bytes"] = rng.integers(1, 20, size=num_rows, dtype=np.int64)
    data["num_pkts_128_to_256_bytes"] = rng.integers(0, 10, size=num_rows, dtype=np.int64)
    data["num_pkts_256_to_512_bytes"] = rng.integers(0, 8, size=num_rows, dtype=np.int64)
    data["num_pkts_512_to_1024_bytes"] = rng.integers(0, 6, size=num_rows, dtype=np.int64)
    data["num_pkts_1024_to_1514_bytes"] = rng.integers(0, 6, size=num_rows, dtype=np.int64)
    data["tcp_win_max_in"] = rng.integers(1024, 65535, size=num_rows, dtype=np.int64)
    data["tcp_win_max_out"] = rng.integers(1024, 65535, size=num_rows, dtype=np.int64)
    data["icmp_type"] = np.zeros(num_rows, dtype=np.int64)
    data["icmp_ipv4_type"] = np.zeros(num_rows, dtype=np.int64)
    data["is_tls_port"] = (data["dst_port"] == 443).astype(np.int8)
    data["is_ssh_port"] = (data["dst_port"] == 22).astype(np.int8)
    data["bytes_ratio"] = (data["in_bytes"] / np.maximum(1.0, data["out_bytes"])).astype(np.float32)
    data["pkts_ratio"] = (data["in_pkts"] / np.maximum(1.0, data["out_pkts"])).astype(np.float32)

    data["label"] = np.array(labels, dtype=np.int64)
    data["attack_family"] = families
    data["attack_label"] = families
    data["attack_layers"] = ["L3/L4" if fam != "Benign" else "NONE" for fam in families]
    data["attack_mapping"] = ["T1498" if fam != "Benign" else "NONE" for fam in families]
    data["ground_truth_source"] = ["synthetic_l3_l4_fixture"] * num_rows
    if include_time_bucket:
        data["time_bucket"] = (np.arange(num_rows, dtype=np.int16) % 10).astype(np.int16)

    return pd.DataFrame(data)


def make_synthetic_splits(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Create complete synthetic splits mimicking NF-UQ-NIDS-v2 structure:
    - train: 240 rows, Benign + DoS + Brute Force (52 cols)
    - val: 120 rows, Benign + DoS + Brute Force (52 cols)
    - test: 120 rows, Benign + DoS + Brute Force + Shellcode + Worms + holdout hosts (52 cols)
    - unseen_holdout: 48 rows, Shellcode + Worms only, label == 1 (51 cols, no time_bucket)
    """
    df_train = make_synthetic_l3_l4_flows(
        num_rows=240,
        seed=seed,
        attack_families=("Benign", "DoS", "Brute Force"),
        benign_ratio=0.5,
        start_order_id=1,
        num_hosts=6,
        host_prefix="train_host",
        include_time_bucket=True,
    )
    df_val = make_synthetic_l3_l4_flows(
        num_rows=120,
        seed=seed + 1,
        attack_families=("Benign", "DoS", "Brute Force"),
        benign_ratio=0.5,
        start_order_id=1001,
        num_hosts=4,
        host_prefix="val_host",
        include_time_bucket=True,
    )
    df_test_main = make_synthetic_l3_l4_flows(
        num_rows=120,
        seed=seed + 2,
        attack_families=("Benign", "DoS", "Brute Force", "Shellcode", "Worms"),
        benign_ratio=0.5,
        start_order_id=2001,
        num_hosts=4,
        host_prefix="test_holdout_host",
        include_time_bucket=True,
    )
    df_unseen = make_synthetic_l3_l4_flows(
        num_rows=48,
        seed=seed + 3,
        attack_families=("Shellcode", "Worms"),
        benign_ratio=0.0,
        start_order_id=3001,
        num_hosts=2,
        host_prefix="unseen_host",
        include_time_bucket=False,
    )
    return {
        "train": df_train,
        "val": df_val,
        "test": df_test_main,
        "unseen_holdout": df_unseen,
    }


def collect_notebook_text_outputs(nb: nbformat.NotebookNode) -> str:
    chunks: List[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            otype = out.get("output_type")
            if otype == "stream":
                chunks.append(out.get("text", ""))
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {})
                if "text/plain" in data:
                    chunks.append(str(data["text/plain"]))
    return "\n".join(chunks)


def execute_notebook_headless(
    nb_path: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    timeout: int = 300,
) -> Tuple[nbformat.NotebookNode, str]:
    """
    Execute an `.ipynb` notebook in-process via nbconvert.preprocessors.ExecutePreprocessor
    with isolated env vars, returning `(executed_nb, combined_stdout_text)`.
    """
    from nbconvert.preprocessors import ExecutePreprocessor

    assert nb_path.exists(), f"Notebook missing: {nb_path}"
    nb = nbformat.read(str(nb_path), as_version=4)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_updates = {
        "TOPIC04_DATA_DIR": str(data_dir),
        "TOPIC04_CHECKPOINT_DIR": str(checkpoint_dir),
    }
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    with temporary_env_vars(env_updates):
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

    errors = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    errors.append((idx, out.get("ename"), out.get("evalue")))
    assert not errors, f"Notebook {nb_path.name} had cell execution errors: {errors}"
    return nb, collect_notebook_text_outputs(nb)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def notebook_dir() -> Path:
    return NOTEBOOK_DIR


@pytest.fixture(scope="session")
def experiments_dir() -> Path:
    return EXPERIMENTS_DIR


@pytest.fixture(scope="session")
def sample_parquet_path() -> Path:
    assert SAMPLE_PARQUET_PATH.exists(), f"Missing sample parquet: {SAMPLE_PARQUET_PATH}"
    return SAMPLE_PARQUET_PATH


@pytest.fixture(scope="session")
def local_dataset_dir() -> Path:
    return LOCAL_FULL_DATASET_DIR


@pytest.fixture(scope="session")
def schema_allowlist_path() -> Path:
    assert SCHEMA_ALLOWLIST_PATH.exists(), f"Missing schema allowlist: {SCHEMA_ALLOWLIST_PATH}"
    return SCHEMA_ALLOWLIST_PATH


@pytest.fixture(scope="session")
def sample_df(sample_parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(sample_parquet_path)


@pytest.fixture
def isolated_checkpoint_dir(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "Checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


@pytest.fixture
def synthetic_clean_splits() -> Dict[str, pd.DataFrame]:
    return make_synthetic_splits(seed=42)
