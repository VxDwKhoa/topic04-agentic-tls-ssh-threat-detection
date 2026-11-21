"""
Tier 3 E2E Tests: Pairwise & Cross-Feature Combinations, Anti-Leakage Invariance,
Host/Temporal Window Purity, Full Dataset Split Boundaries, and Baseline Coexistence.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from .conftest import (
    DEFAULT_LOCAL_CHECKPOINT_DIR,
    EXPECTED_L3_L4_FEATURE_COLUMNS,
    EXPERIMENTS_DIR,
    LOCAL_FULL_DATASET_DIR,
    NOTEBOOK_DIR,
    NOTEBOOK_FILENAMES,
    REPO_ROOT,
    discover_class,
    import_src_module,
    make_synthetic_l3_l4_flows,
)


def test_e2e_pipeline_week5_privacy_transform_rf_checkpoint_reload_parity(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    rules_mod = import_src_module("src.models.rules_baseline")
    rf_mod = import_src_module("src.models.random_forest")

    config_mod.set_global_seed(42)
    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(sample_parquet_path),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    splits = loader_mod.NFUQDataLoader(resolved, schema_path=schema_allowlist_path).load_tabular_splits()
    transformer = transforms_mod.FlowFeatureTransformer().fit(splits["train"])
    X_train = transformer.transform(splits["train"])
    X_test = transformer.transform(splits["test"])
    y_train = splits["train"]["label"].to_numpy()

    RulesCls = discover_class(
        rules_mod,
        candidate_names=("RulesBaselineModel", "DeterministicRulesModel", "RulesEngine", "RulesThreatModel"),
        base_cls=base_mod.BaseThreatModel,
    )
    RFModelCls = discover_class(
        rf_mod,
        candidate_names=("RandomForestThreatModel", "Week5RandomForestModel", "RandomForestModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    rules_preds = np.asarray(RulesCls().predict(splits["test"]))
    assert rules_preds.shape == (len(splits["test"]),)

    rf = RFModelCls(task="binary", random_state=42, n_estimators=20).fit(X_train, y_train)
    orig_probs = np.asarray(rf.predict_proba(X_test))

    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    mgr.save_checkpoint(
        "e2e_week5_rf",
        {"model": rf, "transformer_state": transformer.get_state()},
        {
            "seed": 42,
            "versions": {"python": "3.13"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "metrics": {"sample_mode": True},
        },
    )
    loaded_artifacts, _ = mgr.load_checkpoint("e2e_week5_rf")
    reloaded_transformer = transforms_mod.FlowFeatureTransformer.from_state(loaded_artifacts["transformer_state"])
    reloaded_rf = loaded_artifacts["model"]

    X_test_reloaded = reloaded_transformer.transform(splits["test"])
    reloaded_probs = np.asarray(reloaded_rf.predict_proba(X_test_reloaded))
    assert np.allclose(orig_probs, reloaded_probs, atol=1e-6)


def test_e2e_pipeline_week6_privacy_transform_if_calibrate_unseen_reload_parity(
    synthetic_clean_splits,
    isolated_checkpoint_dir: Path,
):
    config_mod = import_src_module("src.utils.config")
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    config_mod.set_global_seed(42)
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

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    model = IFModelCls(random_state=42, n_estimators=40).fit(
        transformer.transform(synthetic_clean_splits["train"]),
        synthetic_clean_splits["train"]["label"].to_numpy(),
    )
    calib = model.calibrate_threshold(
        transformer.transform(synthetic_clean_splits["val"]),
        synthetic_clean_splits["val"]["label"].to_numpy(),
        target_fpr=0.05,
    )

    df_test_benign = synthetic_clean_splits["test"][synthetic_clean_splits["test"]["label"] == 0]
    df_unseen = synthetic_clean_splits["unseen_holdout"]
    metrics = IFEvalCls().evaluate_unseen_holdout(
        model=model,
        X_unseen=transformer.transform(df_unseen),
        y_unseen=df_unseen["label"].to_numpy(),
        X_benign_test=transformer.transform(df_test_benign),
        y_benign_test=df_test_benign["label"].to_numpy(),
    )
    assert np.isfinite(metrics["aucpr"]) and np.isfinite(metrics["roc_auc"])

    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    mgr.save_checkpoint(
        "e2e_week6_if",
        {"model": model, "transformer_state": transformer.get_state()},
        {
            "seed": 42,
            "versions": {"python": "3.13"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "calibrated_threshold": float(calib["calibrated_threshold"] if isinstance(calib, dict) else calib),
            "metrics": metrics,
        },
    )
    loaded_artifacts, _ = mgr.load_checkpoint("e2e_week6_if")
    X_unseen_rel = transforms_mod.FlowFeatureTransformer.from_state(
        loaded_artifacts["transformer_state"]
    ).transform(df_unseen)
    assert np.allclose(
        np.asarray(model.score_samples(transformer.transform(df_unseen))),
        np.asarray(loaded_artifacts["model"].score_samples(X_unseen_rel)),
        atol=1e-6,
    )


def test_e2e_pipeline_week7_privacy_transform_sequence_lstm_checkpoint_reload_parity(
    synthetic_clean_splits,
    isolated_checkpoint_dir: Path,
):
    config_mod = import_src_module("src.utils.config")
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")
    base_mod = import_src_module("src.models.base")
    lstm_mod = import_src_module("src.models.lstm_sequence")

    config_mod.set_global_seed(42)
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
    test_ds = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["test"], transformer, seq_len=4, stride=2)

    model = LSTMCls(input_dim=40, hidden_dim=16, num_layers=1, seq_len=4, stride=2, device="cpu")
    TrainerCls(model=model, epochs=2, batch_size=16, lr=1e-3, seed=42, device="cpu").fit(train_ds)
    metrics = EvalCls().evaluate(model, test_ds)
    orig_probs = np.asarray(model.predict_proba(test_ds))

    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    mgr.save_checkpoint(
        "e2e_week7_lstm",
        {
            "state_dict": model.network.state_dict() if hasattr(model, "network") else model.state_dict(),
            "transformer_state": transformer.get_state(),
        },
        {
            "seed": 42,
            "versions": {"torch": "cpu"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "seq_len": 4,
            "stride": 2,
            "hidden_dim": 16,
            "num_layers": 1,
            "metrics": metrics,
        },
    )
    loaded_artifacts, loaded_meta = mgr.load_checkpoint("e2e_week7_lstm")
    reloaded = LSTMCls(
        input_dim=40,
        hidden_dim=loaded_meta.get("hidden_dim", 16),
        num_layers=loaded_meta.get("num_layers", 1),
        seq_len=loaded_meta["seq_len"],
        stride=loaded_meta["stride"],
        device="cpu",
    )
    net = reloaded.network if hasattr(reloaded, "network") else reloaded
    net.load_state_dict(loaded_artifacts["state_dict"])
    net.eval()
    assert np.allclose(orig_probs, np.asarray(reloaded.predict_proba(test_ds)), atol=1e-5)


def test_transformer_anti_leakage_invariance_under_extreme_test_outliers(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    FlowFeatureTransformer = transforms_mod.FlowFeatureTransformer

    df_train = synthetic_clean_splits["train"]
    df_test_normal = synthetic_clean_splits["test"].copy()
    df_test_extreme = synthetic_clean_splits["test"].copy()
    df_test_extreme["in_bytes"] = 10**12
    df_test_extreme["src_to_dst_second_bytes"] = np.float32(1.5e38)

    t1 = FlowFeatureTransformer().fit(df_train)
    X_train_before = t1.transform(df_train).copy()
    _ = t1.transform(df_test_normal)
    state_after_normal = t1.get_state()

    _ = t1.transform(df_test_extreme)
    state_after_extreme = t1.get_state()
    X_train_after = t1.transform(df_train)

    assert state_after_normal == state_after_extreme
    assert np.array_equal(X_train_before, X_train_after)


def test_temporal_sequence_window_host_purity_and_chronological_order():
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")

    df_a = make_synthetic_l3_l4_flows(num_rows=6, seed=101, num_hosts=1, host_prefix="host_a")
    df_a["src_host_token"] = "host_aaaaaaaaaaaaaaaa"
    df_a["dst_port"] = 443
    df_a["flow_order_id"] = [10, 20, 30, 40, 50, 60]
    df_a["in_bytes"] = [100, 200, 300, 400, 500, 600]

    df_b = make_synthetic_l3_l4_flows(num_rows=6, seed=102, num_hosts=1, host_prefix="host_b")
    df_b["src_host_token"] = "host_bbbbbbbbbbbbbbbb"
    df_b["dst_port"] = 22
    df_b["flow_order_id"] = [15, 25, 35, 45, 55, 65]
    df_b["in_bytes"] = [10000, 20000, 30000, 40000, 50000, 60000]

    # Intentionally scramble row order across hosts and timestamps
    df_mixed = pd.concat([df_a, df_b], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    transformer = transforms_mod.FlowFeatureTransformer().fit(df_mixed)
    ds = datasets_mod.TemporalSequenceDataset(df_mixed, transformer=transformer, seq_len=4, stride=2)
    assert len(ds) == 4  # 2 windows from host_a + 2 windows from host_b

    dst_port_idx = EXPECTED_L3_L4_FEATURE_COLUMNS.index("dst_port")
    in_bytes_idx = EXPECTED_L3_L4_FEATURE_COLUMNS.index("in_bytes")

    for i in range(len(ds)):
        x_win, _ = ds[i]
        win_np = x_win.numpy()
        # No cross-host mixing inside any window
        assert np.std(win_np[:, dst_port_idx]) < 1e-5, f"Window {i} mixed multiple hosts!"
        # Strictly increasing flow_order_id -> strictly increasing in_bytes
        assert np.all(np.diff(win_np[:, in_bytes_idx]) > 0), f"Window {i} broke chronological order!"


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_full_nfuq_split_boundaries_host_holdout_and_unseen_families(local_dataset_dir: Path):
    assert local_dataset_dir.exists()
    df_tr = pd.read_parquet(
        local_dataset_dir / "train" / "flows_tabular.parquet",
        columns=["session_id", "src_host_token", "attack_family"],
    )
    df_va = pd.read_parquet(
        local_dataset_dir / "val" / "flows_tabular.parquet",
        columns=["session_id", "src_host_token", "attack_family"],
    )
    df_te = pd.read_parquet(
        local_dataset_dir / "test" / "flows_tabular.parquet",
        columns=["session_id", "src_host_token", "attack_family"],
    )
    df_unseen = pd.read_parquet(
        local_dataset_dir / "test" / "unseen_attack_holdout.parquet",
        columns=["session_id", "src_host_token", "attack_family", "label"],
    )

    assert "Shellcode" not in set(df_tr["attack_family"])
    assert "Worms" not in set(df_tr["attack_family"])
    assert "Shellcode" not in set(df_va["attack_family"])
    assert "Worms" not in set(df_va["attack_family"])

    assert set(df_unseen["session_id"]).issubset(set(df_te["session_id"]))
    assert set(df_unseen["label"].unique()) == {1}

    seen_hosts = set(df_tr["src_host_token"]) | set(df_va["src_host_token"])
    test_only_hosts = set(df_te["src_host_token"]) - seen_hosts
    assert len(test_only_hosts) > 0, "Host holdout split must reserve unseen hosts exclusively in test"


def test_sample_parquet_split_and_sequence_viability_together(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
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

    tab_ds = datasets_mod.TabularFlowDataset(splits["train"], transformer)
    seq_ds = datasets_mod.TemporalSequenceDataset(splits["train"], transformer, seq_len=4, stride=2)
    assert len(tab_ds) == len(splits["train"])
    assert len(seq_ds) > 0


def test_rules_and_random_forest_share_identical_feature_space_no_l7(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    rules_mod = import_src_module("src.models.rules_baseline")
    rf_mod = import_src_module("src.models.random_forest")

    RulesCls = discover_class(
        rules_mod,
        candidate_names=("RulesBaselineModel", "DeterministicRulesModel", "RulesEngine", "RulesThreatModel"),
        base_cls=base_mod.BaseThreatModel,
    )
    RFModelCls = discover_class(
        rf_mod,
        candidate_names=("RandomForestThreatModel", "Week5RandomForestModel", "RandomForestModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df_l3_l4_only = synthetic_clean_splits["train"][EXPECTED_L3_L4_FEATURE_COLUMNS + ["label"]].copy()
    transformer = transforms_mod.FlowFeatureTransformer().fit(df_l3_l4_only)
    X_train = transformer.transform(df_l3_l4_only)
    y_train = df_l3_l4_only["label"].to_numpy()

    rules_preds = np.asarray(RulesCls().predict(df_l3_l4_only[EXPECTED_L3_L4_FEATURE_COLUMNS]))
    rf_preds = np.asarray(RFModelCls(task="binary", random_state=42, n_estimators=10).fit(X_train, y_train).predict(X_train))
    assert rules_preds.shape == rf_preds.shape == (len(df_l3_l4_only),)


def test_week6_calibrated_threshold_monotonicity_across_fpr_targets(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    IFModelCls = discover_class(
        if_mod,
        candidate_names=("IsolationForestThreatModel", "Week6IsolationForestModel", "IsolationForestAnomalyModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    X_val = transformer.transform(synthetic_clean_splits["val"])
    y_val = synthetic_clean_splits["val"]["label"].to_numpy()
    X_test = transformer.transform(synthetic_clean_splits["test"])

    model = IFModelCls(random_state=42, n_estimators=50).fit(
        transformer.transform(synthetic_clean_splits["train"]),
        synthetic_clean_splits["train"]["label"].to_numpy(),
    )

    model.calibrate_threshold(X_val, y_val, target_fpr=0.01)
    alerts_at_1pct = int(np.asarray(model.predict(X_test)).sum())

    model.calibrate_threshold(X_val, y_val, target_fpr=0.10)
    alerts_at_10pct = int(np.asarray(model.predict(X_test)).sum())

    assert alerts_at_10pct >= alerts_at_1pct


def test_notebooks_oop_class_contracts_match_src_module_contracts():
    # Ensure notebooks and src/ both exist and align on core OOP stages
    src_expected = [
        REPO_ROOT / "src" / "utils" / "config.py",
        REPO_ROOT / "src" / "features" / "data_loader.py",
        REPO_ROOT / "src" / "features" / "transforms.py",
        REPO_ROOT / "src" / "features" / "datasets.py",
        REPO_ROOT / "src" / "models" / "base.py",
        REPO_ROOT / "src" / "models" / "rules_baseline.py",
        REPO_ROOT / "src" / "models" / "random_forest.py",
        REPO_ROOT / "src" / "models" / "isolation_forest.py",
        REPO_ROOT / "src" / "models" / "lstm_sequence.py",
    ]
    for p in src_expected:
        assert p.exists(), f"Missing packaged src module: {p}"

    for nb_name in NOTEBOOK_FILENAMES:
        for folder in (NOTEBOOK_DIR, EXPERIMENTS_DIR):
            nb_path = folder / nb_name
            assert nb_path.exists(), f"Missing notebook: {nb_path}"
            nb = nbformat.read(str(nb_path), as_version=4)
            text = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
            for kw in ("PathResolver", "validate_dataframe_privacy", "Checkpoint"):
                assert kw in text, f"{nb_path} missing contract keyword '{kw}'"


def test_existing_baseline_privacy_and_loader_tests_coexist_no_regression():
    import importlib.util

    baseline_path = REPO_ROOT / "tests" / "test_privacy_and_loader.py"
    assert baseline_path.exists()
    spec = importlib.util.spec_from_file_location("baseline_privacy_and_loader", str(baseline_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    mod.test_privacy_validator_blocks_l7_and_raw_ip()
    mod.test_repo_sample_compliance_and_dual_model_loading()


def test_isolated_checkpoint_runs_leave_user_checkpoint_dir_untouched(isolated_checkpoint_dir: Path):
    base_mod = import_src_module("src.models.base")
    from sklearn.dummy import DummyClassifier

    before = set(p.name for p in DEFAULT_LOCAL_CHECKPOINT_DIR.iterdir()) if DEFAULT_LOCAL_CHECKPOINT_DIR.exists() else set()
    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((4, 40)), np.array([0, 1, 0, 1]))
    mgr.save_checkpoint(
        "isolation_guard_model",
        {"model": dummy},
        {
            "seed": 42,
            "versions": {"python": "3.13"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "metrics": {},
        },
    )
    after = set(p.name for p in DEFAULT_LOCAL_CHECKPOINT_DIR.iterdir()) if DEFAULT_LOCAL_CHECKPOINT_DIR.exists() else set()
    assert before == after
    assert any(isolated_checkpoint_dir.iterdir())
