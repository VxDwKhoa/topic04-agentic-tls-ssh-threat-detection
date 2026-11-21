"""
Tier 1 E2E Tests: Dual-Environment Paths, Privacy Compliance (SR-01, CR-06),
Split Preservation (SR-08), Train-Only Transforms, Sequence Datasets, Base OOP & Packaging.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    DEFAULT_LOCAL_CHECKPOINT_DIR,
    EXPECTED_IDENTITY_COLUMNS,
    EXPECTED_L3_L4_FEATURE_COLUMNS,
    EXPECTED_LABEL_COLUMNS,
    HOST_TOKEN_PATTERN,
    IPV4_PATTERN,
    KAGGLE_CHECKPOINT_DIR,
    KAGGLE_DATASET_DIR,
    LOCAL_FULL_DATASET_DIR,
    SAMPLE_PARQUET_PATH,
    import_preprocess_module,
    import_privacy_validator,
    import_src_module,
    temporary_env_vars,
)


# ---------------------------------------------------------------------------
# F3: Dual-Environment Path & Checkpoint Resolution + Provenance (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_path_resolver_explicit_dataset_and_checkpoint_override(
    local_dataset_dir: Path,
    isolated_checkpoint_dir: Path,
):
    config_mod = import_src_module("src.utils.config")
    PathResolver = config_mod.PathResolver

    assert local_dataset_dir.exists(), f"Missing local dataset dir: {local_dataset_dir}"
    resolved = PathResolver.resolve(
        dataset_dir=str(local_dataset_dir),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )

    assert Path(resolved.dataset_dir).resolve() == local_dataset_dir.resolve()
    assert Path(resolved.checkpoint_dir).resolve() == isolated_checkpoint_dir.resolve()
    assert resolved.is_sample_mode is False


def test_path_resolver_env_var_topic04_data_dir_sample_mode(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
):
    config_mod = import_src_module("src.utils.config")
    PathResolver = config_mod.PathResolver

    with temporary_env_vars(
        {
            "TOPIC04_DATA_DIR": str(sample_parquet_path),
            "TOPIC04_CHECKPOINT_DIR": str(isolated_checkpoint_dir),
        }
    ):
        resolved = PathResolver.resolve()

    assert Path(resolved.dataset_dir).resolve() == sample_parquet_path.resolve()
    assert resolved.is_sample_mode is True


def test_path_resolver_kaggle_priority_and_local_fallback(monkeypatch):
    config_mod = import_src_module("src.utils.config")
    PathResolver = config_mod.PathResolver

    orig_exists = Path.exists

    def fake_kaggle_exists(path_obj: Path) -> bool:
        norm = str(path_obj).replace("\\", "/")
        if norm.rstrip("/") in (KAGGLE_DATASET_DIR, "/kaggle/working"):
            return True
        return orig_exists(path_obj)

    with temporary_env_vars({"TOPIC04_DATA_DIR": None, "TOPIC04_CHECKPOINT_DIR": None}):
        with monkeypatch.context() as mp:
            mp.setattr(Path, "exists", fake_kaggle_exists)
            resolved_kaggle = PathResolver.resolve()

    assert KAGGLE_DATASET_DIR in str(resolved_kaggle.dataset_dir).replace("\\", "/")
    assert resolved_kaggle.is_sample_mode is False

    # Now simulate neither Kaggle nor local full dataset existing -> sample fallback
    def fake_only_sample_exists(path_obj: Path) -> bool:
        norm = str(path_obj).replace("\\", "/")
        if norm.rstrip("/") in (
            KAGGLE_DATASET_DIR,
            str(LOCAL_FULL_DATASET_DIR).replace("\\", "/"),
            "/kaggle/working",
        ):
            return False
        return orig_exists(path_obj)

    with temporary_env_vars({"TOPIC04_DATA_DIR": None, "TOPIC04_CHECKPOINT_DIR": None}):
        with monkeypatch.context() as mp:
            mp.setattr(Path, "exists", fake_only_sample_exists)
            resolved_sample = PathResolver.resolve()

    assert Path(resolved_sample.dataset_dir).resolve() == SAMPLE_PARQUET_PATH.resolve()
    assert resolved_sample.is_sample_mode is True


def test_path_resolver_checkpoint_dir_kaggle_vs_local(monkeypatch):
    config_mod = import_src_module("src.utils.config")
    PathResolver = config_mod.PathResolver

    orig_exists = Path.exists

    def fake_kaggle_working_exists(path_obj: Path) -> bool:
        norm = str(path_obj).replace("\\", "/").rstrip("/")
        if norm in ("/kaggle/working", KAGGLE_DATASET_DIR):
            return True
        return orig_exists(path_obj)

    with temporary_env_vars({"TOPIC04_DATA_DIR": None, "TOPIC04_CHECKPOINT_DIR": None}):
        with monkeypatch.context() as mp:
            mp.setattr(Path, "exists", fake_kaggle_working_exists)
            resolved_k = PathResolver.resolve()

    assert KAGGLE_CHECKPOINT_DIR in str(resolved_k.checkpoint_dir).replace("\\", "/")

    # Local fallback checkpoint dir when /kaggle/working does not exist
    with temporary_env_vars({"TOPIC04_DATA_DIR": None, "TOPIC04_CHECKPOINT_DIR": None}):
        resolved_local = PathResolver.resolve()

    assert Path(resolved_local.checkpoint_dir).resolve() == DEFAULT_LOCAL_CHECKPOINT_DIR.resolve()


def test_experiment_config_defaults_and_cpu_device():
    config_mod = import_src_module("src.utils.config")
    ExperimentConfig = config_mod.ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.seed == 42
    assert cfg.seq_len == 8
    assert cfg.stride == 4
    assert cfg.device in ("cpu", "cuda")


def test_set_global_seed_determinism_and_version_provenance():
    import torch

    config_mod = import_src_module("src.utils.config")
    set_global_seed = config_mod.set_global_seed

    versions_1 = set_global_seed(42)
    r1 = random.random()
    n1 = float(np.random.rand())
    t1 = float(torch.rand(1).item())

    versions_2 = set_global_seed(42)
    r2 = random.random()
    n2 = float(np.random.rand())
    t2 = float(torch.rand(1).item())

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2
    assert isinstance(versions_1, dict) and versions_1 == versions_2
    assert any(k.lower() == "python" for k in versions_1)
    assert any("numpy" in k.lower() for k in versions_1)
    assert any("pandas" in k.lower() for k in versions_1)
    assert any("torch" in k.lower() for k in versions_1)


# ---------------------------------------------------------------------------
# F1: Strict No-L7 Privacy Validation (5 tests)
# ---------------------------------------------------------------------------


def test_l3_l4_feature_columns_exact_40_contract():
    prep_mod = import_preprocess_module()
    assert list(prep_mod.L3_L4_FEATURE_COLUMNS) == EXPECTED_L3_L4_FEATURE_COLUMNS
    assert len(prep_mod.L3_L4_FEATURE_COLUMNS) == 40
    assert len(set(prep_mod.L3_L4_FEATURE_COLUMNS)) == 40

    transforms_mod = import_src_module("src.features.transforms")
    assert list(transforms_mod.L3_L4_FEATURE_COLUMNS) == EXPECTED_L3_L4_FEATURE_COLUMNS


def test_sample_parquet_passes_privacy_validator_zero_violations(
    sample_df: pd.DataFrame,
    schema_allowlist_path: Path,
):
    pv = import_privacy_validator()
    audit = pv.validate_dataframe_privacy(
        sample_df,
        schema_path=str(schema_allowlist_path),
        dataset_name="nfuq_sample_1500",
    )
    assert audit["passed_no_l7_audit"] is True
    assert audit["violations"] == []
    assert audit["num_rows_checked"] == 1500
    assert audit["num_columns_checked"] == 52
    assert set(["SR-01", "CR-06"]).issubset(set(audit["compliance_anchors"]))


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_full_dataset_splits_pass_privacy_validator_when_present(
    local_dataset_dir: Path,
    schema_allowlist_path: Path,
):
    assert local_dataset_dir.exists(), f"Missing local dataset dir: {local_dataset_dir}"
    pv = import_privacy_validator()

    split_files = [
        ("train", local_dataset_dir / "train" / "flows_tabular.parquet"),
        ("val", local_dataset_dir / "val" / "flows_tabular.parquet"),
        ("test", local_dataset_dir / "test" / "flows_tabular.parquet"),
        ("unseen_holdout", local_dataset_dir / "test" / "unseen_attack_holdout.parquet"),
    ]
    for name, path in split_files:
        assert path.exists(), f"Missing parquet split: {path}"
        df = pd.read_parquet(path)
        audit = pv.validate_dataframe_privacy(df, schema_path=str(schema_allowlist_path), dataset_name=name)
        assert audit["passed_no_l7_audit"] is True, f"{name} failed privacy audit: {audit['violations']}"
        assert audit["violations"] == []


def test_host_tokens_are_hmac_pseudonymized_zero_ipv4(sample_df: pd.DataFrame):
    for col in ("src_host_token", "dst_host_token"):
        vals = sample_df[col].dropna().astype(str)
        assert len(vals) == len(sample_df)
        assert not vals.str.match(IPV4_PATTERN).any(), f"Raw IPv4 leaked in {col}"
        assert vals.str.match(HOST_TOKEN_PATTERN).all(), f"Unexpected token format in {col}"


def test_nfuq_dataloader_runs_privacy_audit_on_sample_load(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(sample_parquet_path),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    loader = loader_mod.NFUQDataLoader(paths=resolved, schema_path=schema_allowlist_path)
    splits = loader.load_tabular_splits()
    assert isinstance(splits, dict)
    for split_name, df in splits.items():
        if len(df) > 0:
            assert set(EXPECTED_L3_L4_FEATURE_COLUMNS).issubset(set(df.columns)), (
                f"Split {split_name} missing L3/L4 features"
            )


# ---------------------------------------------------------------------------
# F2: Pseudonymized Host Tokens & Split Boundaries (SR-08) (5 tests)
# ---------------------------------------------------------------------------


def test_nfuq_dataloader_returns_four_canonical_split_keys(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(sample_parquet_path),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    loader = loader_mod.NFUQDataLoader(paths=resolved, schema_path=schema_allowlist_path)
    splits = loader.load_tabular_splits()

    assert set(splits.keys()) == {"train", "val", "test", "unseen_holdout"}
    for key, val in splits.items():
        assert isinstance(val, pd.DataFrame), f"Expected DataFrame for split '{key}'"


def test_sample_mode_temporal_split_preserves_disjointness_and_benign_train(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(sample_parquet_path),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    loader = loader_mod.NFUQDataLoader(paths=resolved, schema_path=schema_allowlist_path)
    splits = loader.load_tabular_splits()

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    assert (train_df["label"] == 0).sum() > 0, "Sample train split must include benign flows for IF"

    # Train/Val/Test must be row-disjoint by session_id
    s_tr = set(train_df["session_id"])
    s_va = set(val_df["session_id"])
    s_te = set(test_df["session_id"])
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_full_split_class_and_row_invariants_via_dataloader(
    local_dataset_dir: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(local_dataset_dir),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    loader = loader_mod.NFUQDataLoader(paths=resolved, schema_path=schema_allowlist_path)
    splits = loader.load_tabular_splits(max_train_rows=5000)

    assert 0 < len(splits["train"]) <= 5000
    assert len(splits["val"]) == 91230
    assert len(splits["test"]) == 135108
    assert len(splits["unseen_holdout"]) == 635
    assert "Shellcode" not in set(splits["train"]["attack_family"])
    assert "Worms" not in set(splits["train"]["attack_family"])
    assert "Shellcode" not in set(splits["val"]["attack_family"])
    assert "Worms" not in set(splits["val"]["attack_family"])


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_unseen_attack_holdout_contains_only_unseen_families(local_dataset_dir: Path):
    unseen_path = local_dataset_dir / "test" / "unseen_attack_holdout.parquet"
    assert unseen_path.exists(), f"Missing unseen holdout parquet: {unseen_path}"
    df_unseen = pd.read_parquet(unseen_path)

    assert len(df_unseen) == 635
    assert set(df_unseen["label"].unique()) == {1}
    assert df_unseen["attack_family"].value_counts().to_dict() == {"Shellcode": 562, "Worms": 73}


def test_nfuq_dataloader_temporal_sequence_split_integrity(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(sample_parquet_path),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    loader = loader_mod.NFUQDataLoader(paths=resolved, schema_path=schema_allowlist_path)
    splits = loader.load_tabular_splits()
    for split_name in ("train", "val", "test"):
        df = splits[split_name]
        required = set(EXPECTED_IDENTITY_COLUMNS + EXPECTED_L3_L4_FEATURE_COLUMNS + ["label"])
        assert required.issubset(set(df.columns)), f"{split_name} missing required sequence columns"


# ---------------------------------------------------------------------------
# F4: Feature Sanitization, Train-Only Scaling & TargetLabelEncoder (5 tests)
# ---------------------------------------------------------------------------


def test_flow_feature_transformer_fit_transform_finite_40_cols(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    transformer = FlowFeatureTransformer()
    X_train = transformer.fit(synthetic_clean_splits["train"]).transform(synthetic_clean_splits["train"])
    X_val = transformer.transform(synthetic_clean_splits["val"])
    X_test = transformer.transform(synthetic_clean_splits["test"])
    X_unseen = transformer.transform(synthetic_clean_splits["unseen_holdout"])

    for name, arr, df in (
        ("train", X_train, synthetic_clean_splits["train"]),
        ("val", X_val, synthetic_clean_splits["val"]),
        ("test", X_test, synthetic_clean_splits["test"]),
        ("unseen", X_unseen, synthetic_clean_splits["unseen_holdout"]),
    ):
        assert isinstance(arr, np.ndarray), f"{name} must return np.ndarray"
        assert arr.shape == (len(df), 40), f"{name} unexpected shape {arr.shape}"
        assert np.isfinite(arr).all(), f"{name} contains non-finite values"


def test_flow_feature_transformer_train_mean_std_properties(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    transformer = FlowFeatureTransformer()
    X_train = transformer.fit(synthetic_clean_splits["train"]).transform(synthetic_clean_splits["train"])

    col_means = np.mean(X_train, axis=0)
    col_stds = np.std(X_train, axis=0)
    assert np.all(np.abs(col_means) < 1e-4)
    # Non-constant columns should have std close to 1.0
    non_const = col_stds > 1e-6
    assert non_const.sum() >= 15
    assert np.allclose(col_stds[non_const], 1.0, atol=1e-2)


def test_flow_feature_transformer_get_state_from_state_roundtrip(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    transformer = FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    state = transformer.get_state()
    assert isinstance(state, dict)
    # Ensure JSON serializability of state
    serialized = json.dumps(state)
    restored_state = json.loads(serialized)

    restored = FlowFeatureTransformer.from_state(restored_state)
    X_orig = transformer.transform(synthetic_clean_splits["test"])
    X_restored = restored.transform(synthetic_clean_splits["test"])
    assert np.allclose(X_orig, X_restored, atol=1e-6)


def test_target_label_encoder_fits_known_classes_and_maps_unseen_safely(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    TargetLabelEncoder = transforms_mod.TargetLabelEncoder

    enc = TargetLabelEncoder().fit(synthetic_clean_splits["train"]["attack_family"])
    y_train = enc.transform(synthetic_clean_splits["train"]["attack_family"])
    assert set(np.unique(y_train)) == {0, 1, 2}

    y_test = enc.transform(synthetic_clean_splits["test"]["attack_family"])
    unseen_mask = synthetic_clean_splits["test"]["attack_family"].isin(["Shellcode", "Worms"]).to_numpy()
    assert unseen_mask.any()
    assert np.all(y_test[unseen_mask] == -1)
    assert np.all(y_test[~unseen_mask] >= 0)


def test_tabular_flow_dataset_shapes_and_dtypes(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    ds = datasets_mod.TabularFlowDataset(synthetic_clean_splits["train"], transformer)

    assert len(ds) == len(synthetic_clean_splits["train"])
    x0, y0 = ds[0]
    assert tuple(x0.shape) == (40,)
    assert int(y0) in (0, 1)


# ---------------------------------------------------------------------------
# F5 & F6: Sequence Dataset, Base OOP Contracts, Checkpointing & Packaging (6 tests)
# ---------------------------------------------------------------------------


def test_temporal_sequence_dataset_3d_shape_and_explicit_metadata(synthetic_clean_splits):
    import torch

    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    ds = datasets_mod.TemporalSequenceDataset(
        synthetic_clean_splits["train"],
        transformer=transformer,
        seq_len=8,
        stride=4,
    )

    assert ds.seq_len == 8
    assert ds.stride == 4
    assert ds.num_sequences == len(ds)
    assert ds.num_sequences > 0
    assert ds.num_hosts == synthetic_clean_splits["train"]["src_host_token"].nunique()

    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor)
    assert tuple(x0.shape) == (8, 40), "Sequence tensor must be pure (seq_len, 40) with zero graph features"
    assert int(y0) in (0, 1)


def test_temporal_sequence_dataset_window_label_is_max_label():
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")
    from .conftest import make_synthetic_l3_l4_flows

    df_benign = make_synthetic_l3_l4_flows(
        num_rows=8,
        seed=10,
        attack_families=("Benign",),
        benign_ratio=1.0,
        start_order_id=1,
        num_hosts=1,
        host_prefix="single_host",
    )
    df_benign["src_host_token"] = "host_0000000000000001"
    # First 4 flows benign [0,0,0,0], next 4 have 1 attack [0,0,1,0]
    df_benign.loc[6, "label"] = 1
    df_benign.loc[6, "attack_family"] = "DoS"

    transformer = transforms_mod.FlowFeatureTransformer().fit(df_benign)
    ds = datasets_mod.TemporalSequenceDataset(df_benign, transformer=transformer, seq_len=4, stride=4)
    assert len(ds) == 2
    _, y_first = ds[0]
    _, y_second = ds[1]
    assert int(y_first) == 0
    assert int(y_second) == 1


def test_packaging_pyproject_and_requirements_exist_and_valid(repo_root: Path):
    pyproject = repo_root / "pyproject.toml"
    requirements = repo_root / "requirements.txt"
    assert pyproject.exists(), f"Missing {pyproject}"
    assert requirements.exists(), f"Missing {requirements}"

    req_text = requirements.read_text(encoding="utf-8").lower()
    py_text = pyproject.read_text(encoding="utf-8").lower()
    for pkg in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "joblib"):
        assert pkg in req_text, f"requirements.txt missing {pkg}"
    assert "[project]" in py_text or "[build-system]" in py_text


def test_base_oop_contracts_exist_and_expose_required_interfaces():
    base_mod = import_src_module("src.models.base")
    for cls_name in ("BaseThreatModel", "BaseTrainer", "BaseEvaluator", "CheckpointManager"):
        assert hasattr(base_mod, cls_name), f"src.models.base missing {cls_name}"

    BaseThreatModel = base_mod.BaseThreatModel
    for method in ("fit", "predict", "get_metadata"):
        assert hasattr(BaseThreatModel, method), f"BaseThreatModel missing {method}"

    CheckpointManager = base_mod.CheckpointManager
    for method in ("save_checkpoint", "load_checkpoint"):
        assert hasattr(CheckpointManager, method), f"CheckpointManager missing {method}"


def test_checkpoint_manager_saves_and_loads_artifacts_with_metadata(isolated_checkpoint_dir: Path):
    from sklearn.dummy import DummyClassifier

    base_mod = import_src_module("src.models.base")
    CheckpointManager = base_mod.CheckpointManager

    mgr = CheckpointManager(isolated_checkpoint_dir)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((6, 40)), np.array([0, 0, 0, 1, 1, 0]))
    meta = {
        "seed": 42,
        "versions": {"python": "3.13", "scikit-learn": "1.8.0"},
        "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
        "metrics": {"f1": 0.5},
    }
    mgr.save_checkpoint("dummy_unit_model", {"model": dummy}, meta)
    loaded_artifacts, loaded_meta = mgr.load_checkpoint("dummy_unit_model")

    assert "model" in loaded_artifacts
    assert loaded_meta["seed"] == 42
    assert loaded_meta["feature_columns"] == EXPECTED_L3_L4_FEATURE_COLUMNS


def test_checkpoint_manager_metadata_json_persisted_on_disk(isolated_checkpoint_dir: Path):
    from sklearn.dummy import DummyClassifier

    base_mod = import_src_module("src.models.base")
    CheckpointManager = base_mod.CheckpointManager

    mgr = CheckpointManager(isolated_checkpoint_dir)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((4, 40)), np.array([0, 1, 0, 1]))
    meta = {
        "seed": 42,
        "versions": {"python": "3.13"},
        "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
        "metrics": {"roc_auc": 0.75},
    }
    mgr.save_checkpoint("persisted_meta_test", {"model": dummy}, meta)

    json_files = list(isolated_checkpoint_dir.glob("*persisted_meta_test*_metadata.json"))
    assert json_files, f"Expected *_metadata.json in {isolated_checkpoint_dir}"
    on_disk = json.loads(json_files[0].read_text(encoding="utf-8"))
    for key in ("seed", "versions", "feature_columns", "metrics"):
        assert key in on_disk, f"Metadata JSON missing '{key}'"
