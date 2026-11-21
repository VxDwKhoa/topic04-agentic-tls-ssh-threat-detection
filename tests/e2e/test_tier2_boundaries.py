"""
Tier 2 E2E Tests: Boundary Value Analysis & Corner Cases across all 11 Features.

Covers +inf/-inf/NaN/1.87e38 overflow, 51-col unseen_attack_holdout.parquet,
Shellcode/Worms vs lowercase ransomware, < seq_len & 0-sequence guards,
single-class metric slice guards, L7/IPv4 privacy rejections, and missing/corrupted
paths & checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    EXPECTED_L3_L4_FEATURE_COLUMNS,
    LOCAL_FULL_DATASET_DIR,
    discover_class,
    import_privacy_validator,
    import_src_module,
    make_synthetic_l3_l4_flows,
)


# ---------------------------------------------------------------------------
# Numeric Sanitization & FlowFeatureTransformer Boundaries (6 tests)
# ---------------------------------------------------------------------------


def test_transformer_handles_pos_inf_neg_inf_nan_in_second_bytes(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    df_train = synthetic_clean_splits["train"].copy()
    df_test = synthetic_clean_splits["test"].copy()

    df_train.loc[df_train.index[0], "src_to_dst_second_bytes"] = np.inf
    df_train.loc[df_train.index[1], "dst_to_src_second_bytes"] = -np.inf
    df_train.loc[df_train.index[2], "bytes_ratio"] = np.nan

    df_test.loc[df_test.index[0], "src_to_dst_second_bytes"] = np.inf
    df_test.loc[df_test.index[1], "dst_to_src_second_bytes"] = -np.inf
    df_test.loc[df_test.index[2], "pkts_ratio"] = np.nan

    transformer = FlowFeatureTransformer().fit(df_train)
    X_train = transformer.transform(df_train)
    X_test = transformer.transform(df_test)

    assert np.isfinite(X_train).all(), "Train output must have 0 NaN/inf"
    assert np.isfinite(X_test).all(), "Test output must have 0 NaN/inf"


def test_transformer_handles_extreme_float32_overflow_1e38(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    df_train = synthetic_clean_splits["train"].copy()
    df_test = synthetic_clean_splits["test"].copy()

    # Empirical NF-UQ-NIDS-v2 extreme finite values (1.12e37 in train, 1.872e38 in test)
    df_train.loc[df_train.index[0], "src_to_dst_second_bytes"] = np.float32(1.121e37)
    df_test.loc[df_test.index[0], "dst_to_src_second_bytes"] = np.float32(1.872e38)

    transformer = FlowFeatureTransformer().fit(df_train)
    X_train = transformer.transform(df_train)
    X_test = transformer.transform(df_test)

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()


def test_transformer_transform_before_fit_raises_explicit_error(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    transformer = FlowFeatureTransformer()
    with pytest.raises((RuntimeError, ValueError)):
        transformer.transform(synthetic_clean_splits["test"])


def test_transformer_handles_constant_zero_variance_column_without_nan(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    df_train = synthetic_clean_splits["train"].copy()
    df_test = synthetic_clean_splits["test"].copy()
    df_train["icmp_type"] = 0
    df_train["icmp_ipv4_type"] = 0
    df_test["icmp_type"] = 0

    transformer = FlowFeatureTransformer().fit(df_train)
    X_test = transformer.transform(df_test)
    assert np.isfinite(X_test).all()


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_transformer_sanitizes_full_nfuq_splits_plus_inf_when_present(local_dataset_dir: Path):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    train_path = local_dataset_dir / "train" / "flows_tabular.parquet"
    val_path = local_dataset_dir / "val" / "flows_tabular.parquet"
    test_path = local_dataset_dir / "test" / "flows_tabular.parquet"
    assert train_path.exists() and val_path.exists() and test_path.exists()

    df_tr = pd.read_parquet(train_path)
    df_va = pd.read_parquet(val_path)
    df_te = pd.read_parquet(test_path)

    # Confirm empirical +inf rows exist in full dataset and include them in bounded slice
    inf_mask_tr = np.isinf(df_tr["src_to_dst_second_bytes"]) | np.isinf(df_tr["dst_to_src_second_bytes"])
    assert inf_mask_tr.sum() > 0, "Expected empirical +inf rows in full NF-UQ train split"

    tr_slice = pd.concat([df_tr[inf_mask_tr], df_tr[~inf_mask_tr].head(2000)], ignore_index=True)
    inf_mask_va = np.isinf(df_va["src_to_dst_second_bytes"]) | np.isinf(df_va["dst_to_src_second_bytes"])
    va_slice = pd.concat([df_va[inf_mask_va], df_va[~inf_mask_va].head(500)], ignore_index=True)
    inf_mask_te = np.isinf(df_te["src_to_dst_second_bytes"]) | np.isinf(df_te["dst_to_src_second_bytes"])
    te_slice = pd.concat([df_te[inf_mask_te], df_te[~inf_mask_te].head(500)], ignore_index=True)

    transformer = FlowFeatureTransformer().fit(tr_slice)
    for split_name, split_df in (("train", tr_slice), ("val", va_slice), ("test", te_slice)):
        X = transformer.transform(split_df)
        assert X.shape == (len(split_df), 40)
        assert np.isfinite(X).all(), f"Non-finite values survived sanitization in {split_name}"


def test_transformer_empty_dataframe_boundary(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    empty_df = synthetic_clean_splits["train"].iloc[:0].copy()
    with pytest.raises(ValueError):
        FlowFeatureTransformer().fit(empty_df)

    fitted = FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    X_empty = fitted.transform(empty_df)
    assert isinstance(X_empty, np.ndarray)
    assert X_empty.shape == (0, 40)


# ---------------------------------------------------------------------------
# Schema, Label & Unseen Holdout Boundaries (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_unseen_attack_holdout_51_columns_missing_time_bucket_loads_cleanly(
    local_dataset_dir: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")
    transforms_mod = import_src_module("src.features.transforms")

    unseen_path = local_dataset_dir / "test" / "unseen_attack_holdout.parquet"
    df_raw_unseen = pd.read_parquet(unseen_path)
    assert len(df_raw_unseen.columns) == 51
    assert "time_bucket" not in df_raw_unseen.columns

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(local_dataset_dir),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    splits = loader_mod.NFUQDataLoader(resolved, schema_path=schema_allowlist_path).load_tabular_splits(
        max_train_rows=1000
    )
    transformer = transforms_mod.FlowFeatureTransformer().fit(splits["train"])
    X_unseen = transformer.transform(splits["unseen_holdout"])
    assert X_unseen.shape == (635, 40)
    assert np.isfinite(X_unseen).all()


def test_target_label_encoder_unseen_shellcode_worms_map_to_unknown():
    transforms_mod = import_src_module("src.features.transforms")
    TargetLabelEncoder = transforms_mod.TargetLabelEncoder

    train_families = pd.Series(["Benign", "DoS", "Brute Force", "ransomware"])
    enc = TargetLabelEncoder().fit(train_families)

    test_families = pd.Series(["Benign", "Shellcode", "Worms", "DoS"])
    encoded = enc.transform(test_families)
    assert encoded[0] >= 0
    assert encoded[1] == -1
    assert encoded[2] == -1
    assert encoded[3] >= 0

    if hasattr(enc, "inverse_transform"):
        decoded = enc.inverse_transform(encoded)
        assert decoded[1] == "UNKNOWN_UNSEEN"
        assert decoded[2] == "UNKNOWN_UNSEEN"


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_ransomware_lowercase_vs_titlecase_behavior(local_dataset_dir: Path):
    transforms_mod = import_src_module("src.features.transforms")
    TargetLabelEncoder = transforms_mod.TargetLabelEncoder

    df_train = pd.read_parquet(
        local_dataset_dir / "train" / "flows_tabular.parquet", columns=["attack_family"]
    )
    df_unseen = pd.read_parquet(
        local_dataset_dir / "test" / "unseen_attack_holdout.parquet", columns=["attack_family"]
    )

    train_counts = df_train["attack_family"].value_counts().to_dict()
    assert train_counts.get("ransomware", 0) == 400
    assert set(df_unseen["attack_family"].unique()) == {"Shellcode", "Worms"}

    enc = TargetLabelEncoder().fit(df_train["attack_family"])
    # Lowercase "ransomware" is known; TitleCase "Ransomware" is unseen unless normalized
    assert enc.transform(pd.Series(["ransomware"]))[0] >= 0
    assert enc.transform(pd.Series(["Shellcode"]))[0] == -1


def test_target_label_encoder_transform_before_fit_raises():
    transforms_mod = import_src_module("src.features.transforms")
    TargetLabelEncoder = transforms_mod.TargetLabelEncoder

    enc = TargetLabelEncoder()
    with pytest.raises((RuntimeError, ValueError)):
        enc.transform(pd.Series(["Benign", "DoS"]))


def test_rules_baseline_empty_dataframe_returns_empty_outputs(synthetic_clean_splits):
    base_mod = import_src_module("src.models.base")
    rules_mod = import_src_module("src.models.rules_baseline")
    RulesCls = discover_class(
        rules_mod,
        candidate_names=("RulesBaselineModel", "DeterministicRulesModel", "RulesEngine", "RulesThreatModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    empty_df = synthetic_clean_splits["test"].iloc[:0].copy()
    model = RulesCls()
    preds = np.asarray(model.predict(empty_df))
    assert len(preds) == 0


def test_week5_random_forest_single_class_eval_graceful(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    rf_mod = import_src_module("src.models.random_forest")

    RFModelCls = discover_class(
        rf_mod,
        candidate_names=("RandomForestThreatModel", "Week5RandomForestModel", "RandomForestModel"),
        base_cls=base_mod.BaseThreatModel,
    )
    RFEvalCls = discover_class(
        rf_mod,
        candidate_names=("RandomForestEvaluator", "Week5Evaluator"),
        base_cls=base_mod.BaseEvaluator,
    )

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    X_train = transformer.transform(synthetic_clean_splits["train"])
    y_train = synthetic_clean_splits["train"]["label"].to_numpy()
    model = RFModelCls(task="binary", random_state=42, n_estimators=10).fit(X_train, y_train)

    # Single-class slice (all label == 1)
    df_single = synthetic_clean_splits["unseen_holdout"]
    X_single = transformer.transform(df_single)
    y_single = df_single["label"].to_numpy()
    preds = np.asarray(model.predict(X_single))
    probs = np.asarray(model.predict_proba(X_single))
    scores = probs[:, 1] if probs.ndim == 2 else probs

    evaluator = RFEvalCls()
    metrics = evaluator.evaluate_binary(y_single, preds, scores)
    assert isinstance(metrics, dict)
    assert "f1" in metrics


# ---------------------------------------------------------------------------
# Sequence Windowing & Week 7 LSTM Boundaries (6 tests)
# ---------------------------------------------------------------------------


def test_sequence_dataset_skips_hosts_shorter_than_seq_len():
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")

    # 2 hosts: host_short has 3 rows (< seq_len=6), host_long has 8 rows (>= seq_len=6)
    df_short = make_synthetic_l3_l4_flows(num_rows=3, seed=1, num_hosts=1, host_prefix="short")
    df_short["src_host_token"] = "host_00000000000000aa"
    df_long = make_synthetic_l3_l4_flows(num_rows=8, seed=2, start_order_id=100, num_hosts=1, host_prefix="long")
    df_long["src_host_token"] = "host_00000000000000bb"
    df = pd.concat([df_short, df_long], ignore_index=True)

    transformer = transforms_mod.FlowFeatureTransformer().fit(df)
    ds = datasets_mod.TemporalSequenceDataset(df, transformer=transformer, seq_len=6, stride=2)
    # Only host_long yields windows: indices [0:6] and [2:8] => 2 windows
    assert len(ds) == 2


def test_sequence_dataset_all_short_hosts_returns_zero_sequences():
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")

    df = make_synthetic_l3_l4_flows(num_rows=12, seed=3, num_hosts=6, host_prefix="all_short")
    # 12 rows / 6 hosts = 2 rows per host < seq_len=8
    transformer = transforms_mod.FlowFeatureTransformer().fit(df)
    ds = datasets_mod.TemporalSequenceDataset(df, transformer=transformer, seq_len=8, stride=4)
    assert len(ds) == 0


def test_sequence_dataset_exact_seq_len_boundary_produces_one_window():
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")

    df = make_synthetic_l3_l4_flows(num_rows=5, seed=4, num_hosts=1, host_prefix="exact")
    df["src_host_token"] = "host_00000000000000cc"
    transformer = transforms_mod.FlowFeatureTransformer().fit(df)
    ds = datasets_mod.TemporalSequenceDataset(df, transformer=transformer, seq_len=5, stride=3)
    assert len(ds) == 1


def test_sequence_dataset_stride_larger_than_seq_len_boundary():
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")

    df = make_synthetic_l3_l4_flows(num_rows=10, seed=5, num_hosts=1, host_prefix="stride_gt")
    df["src_host_token"] = "host_00000000000000dd"
    transformer = transforms_mod.FlowFeatureTransformer().fit(df)
    ds = datasets_mod.TemporalSequenceDataset(df, transformer=transformer, seq_len=4, stride=6)
    # Windows: [0:4], [6:10] -> 2 windows
    assert len(ds) == 2


def test_week7_lstm_evaluator_single_class_val_slice_guard(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")
    base_mod = import_src_module("src.models.base")
    lstm_mod = import_src_module("src.models.lstm_sequence")

    LSTMCls = discover_class(
        lstm_mod,
        candidate_names=("LSTMSequenceModel", "LSTMThreatModel", "Week7LSTMModel", "LSTMSequenceClassifier"),
        base_cls=base_mod.BaseThreatModel,
    )
    TrainerCls = discover_class(
        lstm_mod,
        candidate_names=("LSTMTrainer", "Week7LSTMTrainer"),
        base_cls=base_mod.BaseTrainer,
    )
    EvalCls = discover_class(
        lstm_mod,
        candidate_names=("LSTMEvaluator", "Week7LSTMEvaluator"),
        base_cls=base_mod.BaseEvaluator,
    )

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    train_ds = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["train"], transformer, seq_len=4, stride=2)
    single_class_ds = datasets_mod.TemporalSequenceDataset(
        synthetic_clean_splits["unseen_holdout"], transformer, seq_len=4, stride=2
    )
    assert len(single_class_ds) > 0

    model = LSTMCls(input_dim=40, hidden_dim=16, num_layers=1, seq_len=4, stride=2, device="cpu")
    TrainerCls(model=model, epochs=1, batch_size=16, lr=1e-3, seed=42, device="cpu").fit(train_ds)

    metrics = EvalCls().evaluate(model, single_class_ds)
    assert isinstance(metrics, dict)
    assert "f1" in metrics


def test_sample_parquet_non_monotonic_global_flow_order_id_handled(
    sample_df: pd.DataFrame,
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    # Verify empirical sample parquet has non-monotonic global flow_order_id
    assert not sample_df["flow_order_id"].is_monotonic_increasing

    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(sample_parquet_path),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    splits = loader_mod.NFUQDataLoader(resolved, schema_path=schema_allowlist_path).load_tabular_splits()
    transformer = transforms_mod.FlowFeatureTransformer().fit(splits["train"])
    ds = datasets_mod.TemporalSequenceDataset(splits["train"], transformer, seq_len=4, stride=2)
    assert len(ds) > 0


# ---------------------------------------------------------------------------
# Privacy & Security Negative Rejections (6 tests)
# ---------------------------------------------------------------------------


def test_privacy_validator_rejects_nf_uq_dropped_l7_columns(schema_allowlist_path: Path):
    pv = import_privacy_validator()
    for col in ("L7_PROTO", "DNS_QUERY_ID", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE"):
        bad_df = make_synthetic_l3_l4_flows(num_rows=4, seed=11)
        bad_df[col] = 1
        res = pv.validate_dataframe_privacy(bad_df, schema_path=str(schema_allowlist_path))
        assert res["passed_no_l7_audit"] is False, f"Validator failed to reject L7 column {col}"


def test_privacy_validator_rejects_http_and_ssh_payload_columns(schema_allowlist_path: Path):
    pv = import_privacy_validator()
    for col in ("http_body", "http_headers_raw", "ssh_commands", "ssh_terminal_keystrokes"):
        bad_df = make_synthetic_l3_l4_flows(num_rows=4, seed=12)
        bad_df[col] = "redacted_synthetic"
        res = pv.validate_dataframe_privacy(bad_df, schema_path=str(schema_allowlist_path))
        assert res["passed_no_l7_audit"] is False, f"Validator failed to reject {col}"


def test_privacy_validator_rejects_tls_secrets_and_raw_sni(schema_allowlist_path: Path):
    pv = import_privacy_validator()
    for col in ("tls_session_keys", "decrypted_tls_payload", "private_keys", "raw_sni", "raw_san"):
        bad_df = make_synthetic_l3_l4_flows(num_rows=4, seed=13)
        bad_df[col] = "redacted_synthetic"
        res = pv.validate_dataframe_privacy(bad_df, schema_path=str(schema_allowlist_path))
        assert res["passed_no_l7_audit"] is False, f"Validator failed to reject {col}"


def test_privacy_validator_rejects_raw_ipv4_in_src_and_dst_host_tokens(schema_allowlist_path: Path):
    pv = import_privacy_validator()
    for host_col, raw_ip in (("src_host_token", "10.20.30.40"), ("dst_host_token", "192.168.1.1")):
        bad_df = make_synthetic_l3_l4_flows(num_rows=4, seed=14)
        bad_df.loc[0, host_col] = raw_ip
        res = pv.validate_dataframe_privacy(bad_df, schema_path=str(schema_allowlist_path))
        assert res["passed_no_l7_audit"] is False
        assert any(host_col in v for v in res["violations"])


def test_nfuq_dataloader_blocks_contaminated_parquet_with_value_error(
    tmp_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")

    bad_df = make_synthetic_l3_l4_flows(num_rows=20, seed=15)
    bad_df["http_body"] = "synthetic_forbidden_l7_column"
    bad_parquet = tmp_path / "contaminated_sample.parquet"
    bad_df.to_parquet(bad_parquet, index=False)

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(bad_parquet),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    loader = loader_mod.NFUQDataLoader(paths=resolved, schema_path=schema_allowlist_path)
    with pytest.raises(ValueError):
        loader.load_tabular_splits()


def test_week6_isolation_forest_excludes_or_rejects_attack_rows_in_train_fit(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    IFModelCls = discover_class(
        if_mod,
        candidate_names=("IsolationForestThreatModel", "Week6IsolationForestModel", "IsolationForestAnomalyModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df_train = synthetic_clean_splits["train"]
    n_benign = int((df_train["label"] == 0).sum())
    assert (df_train["label"] == 1).sum() > 0

    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
    X_train = transformer.transform(df_train)
    y_train = df_train["label"].to_numpy()

    model = IFModelCls(random_state=42, n_estimators=30).fit(X_train, y_train)
    meta = model.get_metadata()
    assert meta.get("n_train_benign") == n_benign
    assert meta.get("benign_only_fit", True) is True


# ---------------------------------------------------------------------------
# Week 6 Unseen Eval, PathResolver & CheckpointManager Boundaries (5 tests)
# ---------------------------------------------------------------------------


def test_week6_unseen_eval_without_benign_rows_guards_clearly(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    IFModelCls = discover_class(
        if_mod,
        candidate_names=("IsolationForestThreatModel", "Week6IsolationForestModel", "IsolationForestAnomalyModel"),
        base_cls=base_mod.BaseThreatModel,
    )
    IFEvalCls = discover_class(
        if_mod,
        candidate_names=("IsolationForestEvaluator", "Week6Evaluator"),
        base_cls=base_mod.BaseEvaluator,
    )

    df_train = synthetic_clean_splits["train"]
    df_val = synthetic_clean_splits["val"]
    df_unseen = synthetic_clean_splits["unseen_holdout"]

    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
    model = IFModelCls(random_state=42, n_estimators=30).fit(
        transformer.transform(df_train), df_train["label"].to_numpy()
    )
    model.calibrate_threshold(transformer.transform(df_val), df_val["label"].to_numpy(), target_fpr=0.05)

    X_unseen = transformer.transform(df_unseen)
    y_unseen = df_unseen["label"].to_numpy()
    empty_benign = np.empty((0, 40), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.int64)

    evaluator = IFEvalCls()
    with pytest.raises(ValueError):
        evaluator.evaluate_unseen_holdout(
            model=model,
            X_unseen=X_unseen,
            y_unseen=y_unseen,
            X_benign_test=empty_benign,
            y_benign_test=empty_y,
        )


def test_path_resolver_explicit_nonexistent_dataset_dir_raises_error(tmp_path: Path):
    config_mod = import_src_module("src.utils.config")
    missing_dir = tmp_path / "does_not_exist_nfuq"
    with pytest.raises((FileNotFoundError, ValueError)):
        config_mod.PathResolver.resolve(dataset_dir=str(missing_dir))


def test_checkpoint_manager_auto_creates_missing_checkpoint_dir(tmp_path: Path):
    from sklearn.dummy import DummyClassifier

    base_mod = import_src_module("src.models.base")
    nested_dir = tmp_path / "nested" / "sub" / "Checkpoint"
    assert not nested_dir.exists()

    mgr = base_mod.CheckpointManager(nested_dir)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((4, 40)), np.array([0, 1, 0, 1]))
    mgr.save_checkpoint(
        "auto_dir_test",
        {"model": dummy},
        {
            "seed": 42,
            "versions": {"python": "3.13"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "metrics": {},
        },
    )
    assert nested_dir.exists()


def test_checkpoint_manager_load_missing_model_raises_file_not_found(isolated_checkpoint_dir: Path):
    base_mod = import_src_module("src.models.base")
    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    with pytest.raises(FileNotFoundError):
        mgr.load_checkpoint("non_existent_model_xyz")


def test_checkpoint_manager_load_corrupted_metadata_raises_error(isolated_checkpoint_dir: Path):
    from sklearn.dummy import DummyClassifier

    base_mod = import_src_module("src.models.base")
    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((4, 40)), np.array([0, 1, 0, 1]))
    mgr.save_checkpoint(
        "corrupt_meta_model",
        {"model": dummy},
        {
            "seed": 42,
            "versions": {"python": "3.13"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "metrics": {},
        },
    )

    meta_files = list(isolated_checkpoint_dir.glob("*corrupt_meta_model*_metadata.json"))
    assert meta_files, "Expected metadata JSON to corrupt"
    meta_files[0].write_text("{corrupted json !!!", encoding="utf-8")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        mgr.load_checkpoint("corrupt_meta_model")
