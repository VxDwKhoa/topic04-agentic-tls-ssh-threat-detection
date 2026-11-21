"""
Tier 4 E2E Tests: Real-World Application Workloads & Headless Notebook Execution.

Exercises full end-to-end workflows:
1. Headless Week 5 Notebook execution on sample mode
2. Headless Week 6 Notebook execution on sample mode
3. Headless Week 7 Notebook execution on sample mode
4. Unseen-Attack Hunting scenario (Shellcode + Worms paired with benign Test flows)
5. Temporal Sequence Hunting scenario (pure PyTorch LSTM reproducibility & latency)
6. Kaggle Path Simulation & Checkpoint Priority Verification
7. Mirrored `experiments/*.ipynb` Headless Execution Smoke Test
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    DEFAULT_LOCAL_CHECKPOINT_DIR,
    EXPECTED_L3_L4_FEATURE_COLUMNS,
    EXPERIMENTS_DIR,
    KAGGLE_CHECKPOINT_DIR,
    KAGGLE_DATASET_DIR,
    LOCAL_FULL_DATASET_DIR,
    NOTEBOOK_DIR,
    NOTEBOOK_FILENAMES,
    discover_class,
    execute_notebook_headless,
    import_src_module,
    temporary_env_vars,
)


def _snapshot_files(directory: Path) -> Set[str]:
    if not directory.exists():
        return set()
    return {p.name for p in directory.iterdir() if p.is_file()}


def _assert_checkpoint_created(
    isolated_dir: Path,
    default_before: Set[str],
    required_suffixes: Iterable[str],
) -> None:
    iso_files = list(isolated_dir.glob("*"))
    default_after = _snapshot_files(DEFAULT_LOCAL_CHECKPOINT_DIR)
    new_default = [DEFAULT_LOCAL_CHECKPOINT_DIR / name for name in (default_after - default_before)]
    created = iso_files if iso_files else new_default
    assert created, f"No checkpoint artifacts created in {isolated_dir} (or {DEFAULT_LOCAL_CHECKPOINT_DIR})"
    names = [p.name.lower() for p in created]
    for suffix in required_suffixes:
        assert any(n.endswith(suffix) for n in names), f"Missing checkpoint artifact ending in '{suffix}' among {names}"


def test_scenario_1_headless_week5_notebook_execution_on_sample(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
):
    default_before = _snapshot_files(DEFAULT_LOCAL_CHECKPOINT_DIR)
    _, output_text = execute_notebook_headless(
        NOTEBOOK_DIR / "week5_rules_random_forest.ipynb",
        data_dir=sample_parquet_path,
        checkpoint_dir=isolated_checkpoint_dir,
        timeout=300,
    )
    lower = output_text.lower()
    for kw in ("roc", "pr", "f1", "macro", "confusion", "importance", "rules"):
        assert kw in lower, f"Week 5 notebook output missing expected metric/report keyword '{kw}'"
    _assert_checkpoint_created(isolated_checkpoint_dir, default_before, (".joblib", "_metadata.json"))


def test_scenario_2_headless_week6_notebook_execution_on_sample(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
):
    default_before = _snapshot_files(DEFAULT_LOCAL_CHECKPOINT_DIR)
    _, output_text = execute_notebook_headless(
        NOTEBOOK_DIR / "week6_isolation_forest_anomaly.ipynb",
        data_dir=sample_parquet_path,
        checkpoint_dir=isolated_checkpoint_dir,
        timeout=300,
    )
    lower = output_text.lower()
    for kw in ("threshold", "aucpr", "fpr", "reason"):
        assert kw in lower, f"Week 6 notebook output missing expected metric/calibration keyword '{kw}'"
    _assert_checkpoint_created(isolated_checkpoint_dir, default_before, (".joblib", "_metadata.json"))


def test_scenario_3_headless_week7_notebook_execution_on_sample(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
):
    default_before = _snapshot_files(DEFAULT_LOCAL_CHECKPOINT_DIR)
    _, output_text = execute_notebook_headless(
        NOTEBOOK_DIR / "week7_pytorch_lstm_temporal.ipynb",
        data_dir=sample_parquet_path,
        checkpoint_dir=isolated_checkpoint_dir,
        timeout=300,
    )
    lower = output_text.lower()
    for kw in ("seq_len", "stride", "roc", "f1", "latency"):
        assert kw in lower, f"Week 7 notebook output missing expected metric/metadata keyword '{kw}'"
    _assert_checkpoint_created(isolated_checkpoint_dir, default_before, (".pt", "_metadata.json"))


@pytest.mark.skipif(
    not LOCAL_FULL_DATASET_DIR.exists(),
    reason="Full local NF-UQ-NIDS-v2 dataset not present",
)
def test_scenario_4_unseen_attack_hunting_workflow_shellcode_worms(
    local_dataset_dir: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    config_mod.set_global_seed(42)
    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(local_dataset_dir),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    splits = loader_mod.NFUQDataLoader(resolved, schema_path=schema_allowlist_path).load_tabular_splits(
        max_train_rows=10000
    )

    df_train = splits["train"]
    df_val = splits["val"].head(5000)
    df_test_benign = splits["test"][splits["test"]["label"] == 0].head(1000)
    df_unseen = splits["unseen_holdout"]

    assert set(df_unseen["attack_family"].unique()) == {"Shellcode", "Worms"}
    assert len(df_test_benign) == 1000

    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
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

    model = IFModelCls(random_state=42, n_estimators=60).fit(
        transformer.transform(df_train), df_train["label"].to_numpy()
    )
    model.calibrate_threshold(transformer.transform(df_val), df_val["label"].to_numpy(), target_fpr=0.05)

    X_unseen = transformer.transform(df_unseen)
    hunting_out = model.predict_with_context(X_unseen, df_raw=df_unseen)
    assert len(hunting_out["reason_codes"]) == len(df_unseen)
    assert np.all(np.isfinite(hunting_out["confidence"]))

    metrics = IFEvalCls().evaluate_unseen_holdout(
        model=model,
        X_unseen=X_unseen,
        y_unseen=df_unseen["label"].to_numpy(),
        X_benign_test=transformer.transform(df_test_benign),
        y_benign_test=df_test_benign["label"].to_numpy(),
    )
    for k in ("calibrated_threshold", "aucpr", "roc_auc", "tpr_at_fpr"):
        assert k in metrics and np.isfinite(float(metrics[k]))


def test_scenario_5_temporal_sequence_hunting_reproducible_latency(
    sample_parquet_path: Path,
    isolated_checkpoint_dir: Path,
    schema_allowlist_path: Path,
):
    config_mod = import_src_module("src.utils.config")
    loader_mod = import_src_module("src.features.data_loader")
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

    resolved = config_mod.PathResolver.resolve(
        dataset_dir=str(sample_parquet_path),
        checkpoint_dir=str(isolated_checkpoint_dir),
    )
    splits = loader_mod.NFUQDataLoader(resolved, schema_path=schema_allowlist_path).load_tabular_splits()
    transformer = transforms_mod.FlowFeatureTransformer().fit(splits["train"])
    train_ds = datasets_mod.TemporalSequenceDataset(splits["train"], transformer, seq_len=4, stride=2)
    test_ds = datasets_mod.TemporalSequenceDataset(splits["test"], transformer, seq_len=4, stride=2)
    assert len(train_ds) > 0 and len(test_ds) > 0

    def _run_once():
        config_mod.set_global_seed(42)
        m = LSTMCls(input_dim=40, hidden_dim=16, num_layers=1, seq_len=4, stride=2, device="cpu")
        TrainerCls(model=m, epochs=2, batch_size=16, lr=1e-3, seed=42, device="cpu").fit(train_ds)
        ev = EvalCls().evaluate(m, test_ds)
        return np.asarray(m.predict_proba(test_ds)), ev

    probs_1, metrics_1 = _run_once()
    probs_2, _ = _run_once()

    assert np.allclose(probs_1, probs_2, atol=1e-5), "Seed=42 LSTM temporal predictions must be reproducible"
    assert metrics_1["seq_len"] == 4 and metrics_1["stride"] == 2
    latency_keys = [k for k in metrics_1 if "latency" in k or "events_per_sec" in k]
    assert latency_keys and np.isfinite(float(metrics_1[latency_keys[0]])) and float(metrics_1[latency_keys[0]]) > 0


def test_scenario_6_kaggle_path_simulation_and_checkpoint_fallback(
    monkeypatch,
    sample_parquet_path: Path,
    local_dataset_dir: Path,
):
    config_mod = import_src_module("src.utils.config")
    PathResolver = config_mod.PathResolver
    orig_exists = Path.exists

    def fake_kaggle_env(path_obj: Path) -> bool:
        norm = str(path_obj).replace("\\", "/").rstrip("/")
        if norm in (KAGGLE_DATASET_DIR, "/kaggle/working"):
            return True
        return orig_exists(path_obj)

    # 1. Default priority picks Kaggle when /kaggle paths exist
    with temporary_env_vars({"TOPIC04_DATA_DIR": None, "TOPIC04_CHECKPOINT_DIR": None}):
        with monkeypatch.context() as mp:
            mp.setattr(Path, "exists", fake_kaggle_env)
            k_paths = PathResolver.resolve()
    assert KAGGLE_DATASET_DIR in str(k_paths.dataset_dir).replace("\\", "/")
    assert KAGGLE_CHECKPOINT_DIR in str(k_paths.checkpoint_dir).replace("\\", "/")
    assert k_paths.is_sample_mode is False

    # 2. Explicit TOPIC04_DATA_DIR overrides Kaggle even inside Kaggle environment
    with temporary_env_vars({"TOPIC04_DATA_DIR": str(sample_parquet_path), "TOPIC04_CHECKPOINT_DIR": None}):
        with monkeypatch.context() as mp:
            mp.setattr(Path, "exists", fake_kaggle_env)
            override_paths = PathResolver.resolve()
    assert Path(override_paths.dataset_dir).resolve() == sample_parquet_path.resolve()
    assert override_paths.is_sample_mode is True

    # 3. Local full dataset selected when Kaggle absent and no env override (or sample fallback if absent)
    with temporary_env_vars({"TOPIC04_DATA_DIR": None, "TOPIC04_CHECKPOINT_DIR": None}):
        local_paths = PathResolver.resolve()
    if LOCAL_FULL_DATASET_DIR.exists():
        assert Path(local_paths.dataset_dir).resolve() == local_dataset_dir.resolve()
        assert local_paths.is_sample_mode is False
    else:
        assert Path(local_paths.dataset_dir).resolve() == sample_parquet_path.resolve()
        assert local_paths.is_sample_mode is True


def test_scenario_7_mirrored_experiments_notebooks_headless_execution(
    sample_parquet_path: Path,
    tmp_path: Path,
):
    default_before = _snapshot_files(DEFAULT_LOCAL_CHECKPOINT_DIR)
    for nb_name in NOTEBOOK_FILENAMES:
        exp_ckpt = tmp_path / f"ckpt_{Path(nb_name).stem}"
        _, out_text = execute_notebook_headless(
            EXPERIMENTS_DIR / nb_name,
            data_dir=sample_parquet_path,
            checkpoint_dir=exp_ckpt,
            timeout=300,
        )
        assert len(out_text.strip()) > 0, f"Mirrored notebook {nb_name} produced empty output"
        _assert_checkpoint_created(exp_ckpt, default_before, ("_metadata.json",))
