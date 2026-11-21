"""
Unit and integration tests for Milestone M1 core src/ infrastructure:
- Dual-environment PathResolver, ExperimentConfig, and reproducibility utilities
- NFUQDataLoader (sample mode, full-dataset compatibility, No-L7 & IPv4 privacy audits)
- FlowFeatureTransformer (float64 sanitization, quantile clipping, Train-only scaling, JSON state)
- TargetLabelEncoder (known + unseen attack classes, JSON state)
- TabularFlowDataset & TemporalSequenceDataset (grouping, sorting, short/empty guards, parity)
- CheckpointManager (.joblib and weights_only=True .pt state_dict + metadata JSON)
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.utils.config as config_mod
from src.features import (
    L3_L4_FEATURE_COLUMNS,
    NFUQDataLoader,
    TabularFlowDataset,
    TargetLabelEncoder,
    TemporalSequenceDataset,
    FlowFeatureTransformer,
    UNSEEN_ATTACK_FAMILIES,
)
from src.models import CheckpointManager
from src.utils import (
    ExperimentConfig,
    PathResolver,
    discover_repo_root,
    get_environment_info,
    set_global_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_FULL_DIR = Path(r"D:\Crypto\DoAn\NF-UQ-NIDS-v2")
SAMPLE_PARQUET = REPO_ROOT / "data" / "public_loaders" / "sample_data" / "nfuq_sample_1500.parquet"


def _make_synthetic_flows(
    n_rows: int = 16,
    seed: int = 42,
    host_prefix: str = "host_aaaabbbbcccc",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: Dict[str, object] = {
        "session_id": [f"sess_{i}" for i in range(n_rows)],
        "flow_order_id": np.arange(100, 100 + n_rows, dtype=np.int64),
        "window_id": np.zeros(n_rows, dtype=np.int32),
        "src_host_token": [f"{host_prefix}{i % 2:04x}" for i in range(n_rows)],
        "dst_host_token": [f"host_dddd11112222{i % 3:04x}" for i in range(n_rows)],
        "label": np.array([i % 2 for i in range(n_rows)], dtype=np.int64),
        "attack_family": ["Benign" if i % 2 == 0 else "DoS" for i in range(n_rows)],
        "time_bucket": np.zeros(n_rows, dtype=np.int16),
    }
    for idx, col in enumerate(L3_L4_FEATURE_COLUMNS):
        if col in ("src_port_class", "proto", "is_tls_port", "is_ssh_port"):
            data[col] = rng.integers(0, 3, size=n_rows, dtype=np.int64)
        elif "second_bytes" in col or "throughput" in col or "ratio" in col:
            data[col] = rng.uniform(10.0, 5000.0, size=n_rows).astype(np.float32) + idx
        else:
            data[col] = rng.integers(1, 1000, size=n_rows, dtype=np.int64)
    return pd.DataFrame(data)


def test_discover_repo_root_from_arbitrary_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    found = discover_repo_root()
    assert found == REPO_ROOT


def test_path_resolver_priority_and_dual_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOPIC04_DATA_DIR", raising=False)
    monkeypatch.delenv("TOPIC04_CHECKPOINT_DIR", raising=False)

    # Create mock Kaggle direct & nested structures and local mock
    kaggle_root = tmp_path / "kaggle_input" / "NF-UQ-NIDS-v2"
    (kaggle_root / "train").mkdir(parents=True)
    (kaggle_root / "train" / "flows_tabular.parquet").write_bytes(b"placeholder")
    (kaggle_root / "metadata_manifest.json").write_text("{}", encoding="utf-8")

    kaggle_working = tmp_path / "kaggle_working"
    kaggle_working.mkdir()

    custom_dir = tmp_path / "custom_data"
    (custom_dir / "train").mkdir(parents=True)
    (custom_dir / "train" / "flows_tabular.parquet").write_bytes(b"placeholder")

    env_dir = tmp_path / "env_data"
    (env_dir / "train").mkdir(parents=True)
    (env_dir / "train" / "flows_tabular.parquet").write_bytes(b"placeholder")

    monkeypatch.setattr(config_mod, "KAGGLE_DATASET_DIR", kaggle_root)
    monkeypatch.setattr(config_mod, "KAGGLE_WORKING_DIR", kaggle_working)
    monkeypatch.setattr(config_mod, "KAGGLE_CHECKPOINT_DIR", kaggle_working / "Checkpoint")

    # 1. Explicit arg takes highest priority and resolve() is side-effect-free (no mkdir)
    explicit_ckpt = tmp_path / "explicit_ckpt_not_created"
    resolved_explicit = PathResolver.resolve(dataset_dir=custom_dir, checkpoint_dir=explicit_ckpt)
    assert resolved_explicit.mode == "custom"
    assert resolved_explicit.dataset_dir == custom_dir
    assert resolved_explicit.checkpoint_dir == explicit_ckpt
    assert not explicit_ckpt.exists()
    assert resolved_explicit.split_tabular_path("train") == custom_dir / "train" / "flows_tabular.parquet"
    assert resolved_explicit.unseen_holdout_path == custom_dir / "test" / "unseen_attack_holdout.parquet"
    assert resolved_explicit.sequences_dir("val") == custom_dir / "val" / "sequences_temporal"
    assert isinstance(resolved_explicit.to_dict(), dict)

    # 2. TOPIC04_DATA_DIR & TOPIC04_CHECKPOINT_DIR env vars
    monkeypatch.setenv("TOPIC04_DATA_DIR", str(env_dir))
    monkeypatch.setenv("TOPIC04_CHECKPOINT_DIR", str(tmp_path / "env_ckpt"))
    resolved_env = PathResolver.resolve()
    assert resolved_env.mode == "custom"
    assert resolved_env.dataset_dir == env_dir
    assert resolved_env.checkpoint_dir == tmp_path / "env_ckpt"

    # 3. Kaggle mount priority (direct and nested) over local
    monkeypatch.delenv("TOPIC04_DATA_DIR")
    monkeypatch.delenv("TOPIC04_CHECKPOINT_DIR")
    resolved_kaggle = PathResolver.resolve()
    assert resolved_kaggle.mode == "kaggle"
    assert not resolved_kaggle.is_sample_mode
    assert resolved_kaggle.dataset_dir == kaggle_root
    assert resolved_kaggle.checkpoint_dir == kaggle_working / "Checkpoint"

    # Nested Kaggle directory structure
    nested_kaggle = tmp_path / "kaggle_nested"
    (nested_kaggle / "NF-UQ-NIDS-v2" / "train").mkdir(parents=True)
    (nested_kaggle / "NF-UQ-NIDS-v2" / "train" / "flows_tabular.parquet").write_bytes(b"placeholder")
    monkeypatch.setattr(config_mod, "KAGGLE_DATASET_DIR", nested_kaggle)
    resolved_nested = PathResolver.resolve()
    assert resolved_nested.mode == "kaggle"
    assert resolved_nested.dataset_dir == nested_kaggle / "NF-UQ-NIDS-v2"

    # 4. prefer_sample forces sample mode
    resolved_sample = PathResolver.resolve(prefer_sample=True)
    assert resolved_sample.mode == "local_sample"
    assert resolved_sample.is_sample_mode is True
    assert resolved_sample.sample_path == SAMPLE_PARQUET
    assert resolved_sample.split_tabular_path("train") is None
    assert resolved_sample.unseen_holdout_path is None
    assert resolved_sample.sequences_dir("train") is None

    # 5. Missing explicit path fails closed
    with pytest.raises(FileNotFoundError):
        PathResolver.resolve(dataset_dir=tmp_path / "does_not_exist")


def test_experiment_config_and_seed_reproducibility() -> None:
    cfg = ExperimentConfig()
    assert cfg.seed == 42
    assert cfg.seq_len == 8
    assert cfg.stride == 4
    assert cfg.device in ("cpu", "cuda")
    assert cfg.to_dict()["seed"] == 42

    with pytest.raises(ValueError):
        ExperimentConfig(seq_len=0)
    with pytest.raises(ValueError):
        ExperimentConfig(stride=0)

    env1 = set_global_seed(42)
    r1 = random.random()
    n1 = float(np.random.rand())
    t1 = float(torch.rand(1).item())

    set_global_seed(42)
    r2 = random.random()
    n2 = float(np.random.rand())
    t2 = float(torch.rand(1).item())

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2

    env_info = get_environment_info()
    for key in ("python", "numpy", "pandas", "pyarrow", "scikit_learn", "torch", "joblib"):
        assert key in env_info
    assert env1["seed"] == "42"


def test_nfuq_data_loader_sample_mode_and_privacy() -> None:
    paths = PathResolver.resolve(prefer_sample=True)
    loader = NFUQDataLoader(paths)
    splits = loader.load_tabular_splits()

    assert set(splits.keys()) == {"train", "val", "test", "unseen_holdout"}
    df_train = splits["train"]
    df_val = splits["val"]
    df_test = splits["test"]
    df_unseen = splits["unseen_holdout"]

    assert len(df_train) + len(df_val) + len(df_test) == 1500
    assert len(df_unseen) > 0
    assert set(df_unseen["attack_family"].unique()).issubset(UNSEEN_ATTACK_FAMILIES)

    # Zero session overlap across splits
    train_ids = set(df_train["session_id"])
    val_ids = set(df_val["session_id"])
    test_ids = set(df_test["session_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Temporal ordering preserved without leakage
    assert df_train["flow_order_id"].is_monotonic_increasing
    assert df_val["flow_order_id"].is_monotonic_increasing
    assert df_train["flow_order_id"].max() <= df_val["flow_order_id"].min()

    # Privacy audit reports recorded and clean
    for split_name in ("train", "val", "test", "unseen_holdout"):
        assert loader.privacy_reports[split_name]["passed_no_l7_audit"] is True
        assert loader.privacy_reports[split_name]["violations"] == []

    df_seq = loader.load_temporal_sequences(split="train")
    assert len(df_seq) == len(df_train)
    assert loader.privacy_reports["train_sequences"]["passed_no_l7_audit"] is True


def test_nfuq_data_loader_rejects_l7_and_raw_ip(tmp_path: Path) -> None:
    paths = PathResolver.resolve(prefer_sample=True)
    loader = NFUQDataLoader(paths)

    # 1. L7 column injection
    bad_l7 = _make_synthetic_flows(n_rows=10)
    bad_l7["http_body"] = "GET /secret HTTP/1.1"
    with pytest.raises(ValueError, match="Privacy audit failed"):
        loader.validate_privacy(bad_l7, "bad_l7")

    # 2. Raw IPv4 leaked beyond row 500 (testing full-column IPv4 check)
    bad_ip = _make_synthetic_flows(n_rows=550)
    bad_ip.loc[540, "src_host_token"] = "192.168.10.25"
    with pytest.raises(ValueError, match="Privacy audit failed"):
        loader.validate_privacy(bad_ip, "bad_ip_late")


@pytest.mark.skipif(not LOCAL_FULL_DIR.exists(), reason="Full local NF-UQ-NIDS-v2 dataset not present")
def test_nfuq_data_loader_full_dataset_compat() -> None:
    paths = PathResolver.resolve(dataset_dir=LOCAL_FULL_DIR)
    assert paths.is_sample_mode is False
    loader = NFUQDataLoader(paths)
    splits = loader.load_tabular_splits(max_train_rows=2000, max_eval_rows=2000)

    assert len(splits["train"]) == 2000
    assert len(splits["val"]) == 2000
    assert len(splits["test"]) == 2000
    assert len(splits["unseen_holdout"]) == 635
    assert set(splits["unseen_holdout"]["label"].unique()) == {1}
    assert set(splits["unseen_holdout"]["attack_family"].unique()) == {"Shellcode", "Worms"}

    seq_df = loader.load_temporal_sequences(split="train", windows=[0], max_rows=500)
    assert len(seq_df) == 500
    assert loader.privacy_reports["train_sequences"]["passed_no_l7_audit"] is True


def test_flow_feature_transformer_inf_nan_and_overflow() -> None:
    df_train = _make_synthetic_flows(n_rows=30, seed=7)
    # Inject +inf, -inf, NaN, float32-scale overflow (1.872e38), constant column, and all-NaN column
    df_train.loc[0, "src_to_dst_second_bytes"] = np.inf
    df_train.loc[1, "src_to_dst_second_bytes"] = -np.inf
    df_train.loc[2, "dst_to_src_second_bytes"] = np.nan
    df_train.loc[3, "dst_to_src_second_bytes"] = 1.872e38
    df_train["icmp_type"] = 5  # zero-variance constant column
    df_train["icmp_ipv4_type"] = np.nan  # all-NaN column

    transformer = FlowFeatureTransformer()
    with pytest.raises(RuntimeError):
        transformer.transform(df_train)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X = transformer.fit_transform(df_train)
    assert not any("All-NaN slice" in str(w.message) for w in caught)

    assert X.shape == (30, len(L3_L4_FEATURE_COLUMNS))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert not np.isnan(X).any()

    # Empty eval DataFrame returns (0, 40) safely
    X_empty = transformer.transform(df_train.iloc[0:0])
    assert X_empty.shape == (0, len(L3_L4_FEATURE_COLUMNS))
    assert X_empty.dtype == np.float32


def test_flow_feature_transformer_train_only_no_leakage_and_serialization() -> None:
    df_train = _make_synthetic_flows(n_rows=40, seed=11)
    df_val = _make_synthetic_flows(n_rows=15, seed=99)
    df_val.loc[0, "src_to_dst_second_bytes"] = np.inf
    df_val.loc[1, "dst_to_src_second_bytes"] = 1.872e38
    df_val.loc[2, "in_bytes"] = np.nan

    transformer = FlowFeatureTransformer().fit(df_train)
    mean_before = transformer.scaler_.mean_.copy()
    scale_before = transformer.scaler_.scale_.copy()
    upper_before = transformer.upper_bounds_.copy()
    medians_before = transformer.medians_.copy()

    X_val = transformer.transform(df_val)
    assert np.isfinite(X_val).all()
    assert np.array_equal(transformer.scaler_.mean_, mean_before)
    assert np.array_equal(transformer.scaler_.scale_, scale_before)
    assert np.array_equal(transformer.upper_bounds_, upper_before)
    assert np.array_equal(transformer.medians_, medians_before)

    state = transformer.get_state()
    json_str = json.dumps(state, allow_nan=False)
    restored = FlowFeatureTransformer.from_state(json.loads(json_str))
    X_val_restored = restored.transform(df_val)
    assert np.allclose(X_val, X_val_restored, atol=1e-6)


def test_target_label_encoder_known_and_unseen() -> None:
    train_classes = [f"Attack_{i:02d}" for i in range(18)] + ["Benign"]
    encoder = TargetLabelEncoder().fit(train_classes)
    assert encoder.num_classes == 19

    y_enc = encoder.transform(["Benign", "Attack_00", "Shellcode", "Worms", "NewAttack", None, np.nan])
    assert y_enc.dtype == np.int64
    assert y_enc[0] >= 0
    assert y_enc[1] == 0
    assert list(y_enc[2:]) == [-1, -1, -1, -1, -1]
    assert list(encoder.known_mask(y_enc)) == [True, True, False, False, False, False, False]

    inv = encoder.inverse_transform(y_enc)
    assert inv[:2] == ["Benign", "Attack_00"]
    assert inv[2:] == ["UNKNOWN_UNSEEN"] * 5

    restored = TargetLabelEncoder.from_state(json.loads(json.dumps(encoder.get_state(), allow_nan=False)))
    assert np.array_equal(restored.transform(["Benign", "Shellcode"]), y_enc[[0, 2]])


def test_tabular_flow_dataset_shapes_and_empty() -> None:
    df = _make_synthetic_flows(n_rows=12, seed=5)
    transformer = FlowFeatureTransformer().fit(df)

    ds = TabularFlowDataset(df, transformer=transformer, label_col="label")
    assert len(ds) == 12
    x0, y0 = ds[0]
    assert x0.shape == (len(L3_L4_FEATURE_COLUMNS),)
    assert x0.dtype == torch.float32
    assert y0.dtype == torch.long

    batch_x, batch_y = next(iter(DataLoader(ds, batch_size=4)))
    assert batch_x.shape == (4, len(L3_L4_FEATURE_COLUMNS))
    assert batch_y.shape == (4,)

    ds_empty = TabularFlowDataset(df.iloc[0:0], transformer=transformer, label_col="label")
    assert len(ds_empty) == 0
    assert ds_empty.X.shape == (0, len(L3_L4_FEATURE_COLUMNS))

    with pytest.raises(ValueError, match="forbidden"):
        TabularFlowDataset(df, feature_cols=["src_host_token", "in_bytes"])


def test_temporal_sequence_dataset_grouping_sorting_labeling() -> None:
    # Two hosts, deliberately shuffled rows
    df_h1 = _make_synthetic_flows(n_rows=6, seed=1, host_prefix="host_111100000000")
    df_h1["src_host_token"] = "host_alpha"
    df_h1["flow_order_id"] = [10, 20, 30, 40, 50, 60]
    df_h1["label"] = [0, 0, 0, 0, 1, 0]

    df_h2 = _make_synthetic_flows(n_rows=4, seed=2, host_prefix="host_222200000000")
    df_h2["src_host_token"] = "host_beta"
    df_h2["flow_order_id"] = [5, 15, 25, 35]
    df_h2["label"] = [0, 0, 0, 0]

    df_shuffled = pd.concat([df_h1, df_h2], ignore_index=True).sample(frac=1.0, random_state=42)
    transformer = FlowFeatureTransformer().fit(df_shuffled)

    ds = TemporalSequenceDataset(df_shuffled, transformer=transformer, seq_len=4, stride=2)
    # host_alpha: len=6 -> windows [0:4] (label=0), [2:6] (label=max([0,0,1,0])=1)
    # host_beta: len=4 -> window [0:4] (label=0)
    assert len(ds) == 3
    assert ds.X.shape == (3, 4, len(L3_L4_FEATURE_COLUMNS))
    assert ds.num_hosts == 2
    assert ds.num_active_hosts == 2
    assert ds.num_skipped_hosts == 0
    assert ds.y.tolist() == [0, 1, 0]
    assert ds.get_class_distribution() == {0: 2, 1: 1}
    meta = ds.get_metadata()
    assert meta["seq_len"] == 4 and meta["stride"] == 2 and meta["num_sequences"] == 3

    with pytest.raises(ValueError, match="forbidden"):
        TemporalSequenceDataset(df_shuffled, feature_cols=["flow_order_id", "in_bytes"])


def test_temporal_sequence_dataset_short_groups_and_zero_sequences() -> None:
    df_long = _make_synthetic_flows(n_rows=5, seed=3)
    df_long["src_host_token"] = "host_long"
    df_long["flow_order_id"] = [1, 2, 3, 4, 5]

    df_short = _make_synthetic_flows(n_rows=2, seed=4)
    df_short["src_host_token"] = "host_short"
    df_short["flow_order_id"] = [10, 20]

    df_mixed = pd.concat([df_long, df_short], ignore_index=True)
    ds_mixed = TemporalSequenceDataset(df_mixed, seq_len=4, stride=2)
    assert len(ds_mixed) == 1
    assert ds_mixed.num_hosts == 2
    assert ds_mixed.num_active_hosts == 1
    assert ds_mixed.num_skipped_hosts == 1
    assert ds_mixed.num_skipped_flows == 2

    # All short hosts -> 0 sequences + UserWarning, zero fabricated rows
    with pytest.warns(UserWarning):
        ds_zero = TemporalSequenceDataset(df_short, seq_len=4, stride=2)
    assert len(ds_zero) == 0
    assert ds_zero.X.shape == (0, 4, len(L3_L4_FEATURE_COLUMNS))
    assert ds_zero.get_class_distribution() == {}

    # Empty DataFrame -> (0, seq_len, 40) + UserWarning
    with pytest.warns(UserWarning):
        ds_empty = TemporalSequenceDataset(df_short.iloc[0:0], seq_len=4, stride=2)
    assert len(ds_empty) == 0
    assert ds_empty.X.shape == (0, 4, len(L3_L4_FEATURE_COLUMNS))

    # Parity check against preprocess_and_split_nfuq.build_lstm_sequences on repo sample
    sys.path.insert(0, str(REPO_ROOT / "data" / "public_loaders"))
    from preprocess_and_split_nfuq import build_lstm_sequences

    df_sample = pd.read_parquet(SAMPLE_PARQUET)
    X_ref, y_ref = build_lstm_sequences(df_sample, seq_len=4, stride=2)
    ds_parity = TemporalSequenceDataset(df_sample, transformer=None, seq_len=4, stride=2)
    assert ds_parity.num_sequences == len(X_ref)
    assert np.allclose(ds_parity.X.numpy(), X_ref)
    assert np.array_equal(ds_parity.y.numpy(), y_ref)


def test_checkpoint_manager_joblib_and_pt_roundtrip(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "ckpt")

    # 1. Joblib artifact + metadata sanitization (inf/nan -> None, Path -> str)
    sklearn_like = {"weights": [1, 2, 3], "threshold": 0.75}
    res_j = mgr.save_checkpoint(
        "week5_rf",
        sklearn_like,
        metadata={
            "seed": 42,
            "feature_cols": list(L3_L4_FEATURE_COLUMNS),
            "metrics": {"f1": 0.95, "bad_inf": float("inf"), "bad_nan": float("nan")},
            "hyperparameters": {"n_estimators": 100},
            "path_ref": tmp_path,
        },
    )
    assert res_j["artifact_path"].suffix == ".joblib"
    loaded_obj, loaded_meta = mgr.load_checkpoint("week5_rf")
    assert loaded_obj == sklearn_like
    assert loaded_meta["seed"] == 42
    assert loaded_meta["metrics"]["f1"] == 0.95
    assert loaded_meta["metrics"]["bad_inf"] is None
    assert loaded_meta["metrics"]["bad_nan"] is None
    assert "environment" in loaded_meta

    # 2. PyTorch nn.Module -> state_dict (.pt) loaded with weights_only=True
    net = nn.Linear(40, 2)
    res_pt = mgr.save_checkpoint(
        "week7_lstm",
        net,
        metadata={"seq_len": 8, "stride": 4},
    )
    assert res_pt["artifact_path"].suffix == ".pt"
    loaded_state, loaded_pt_meta = mgr.load_checkpoint("week7_lstm")
    assert isinstance(loaded_state, dict)
    assert torch.equal(loaded_state["weight"], net.state_dict()["weight"])
    assert loaded_pt_meta["seq_len"] == 8

    assert mgr.list_checkpoints() == ["week5_rf", "week7_lstm"]
    assert mgr.load_metadata("week7_lstm")["stride"] == 4

    with pytest.raises(FileNotFoundError):
        mgr.load_checkpoint("non_existent_model")
