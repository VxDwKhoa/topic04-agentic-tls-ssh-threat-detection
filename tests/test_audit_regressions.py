"""
Regression tests for Topic 04 Critical & High fixes across `src/` and `Notebook/*.ipynb`:
- Cross-environment `.joblib` reload (no `__main__` pickle dependency) via `from_checkpoint`
- Week 7 `transformer_state` presence & canonical schema parity in metadata
- `IsolationForestThreatModel.predict_with_context` NaN safety & context reason codes (`src/` + notebook)
- `LSTMSequenceModel.fit(tensor, y)`, `_extract_labels(None / grad / Tensor)`, and `batch_size=256` default
- Notebook sample-mode unseen-attack non-leakage & full-column IPv4 / L7 pattern rejection
- Notebook `/kaggle/input` slug scan fallback
"""

from __future__ import annotations

import inspect
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.features.data_loader import L3_L4_FEATURE_COLUMNS, NFUQDataLoader, UNSEEN_ATTACK_FAMILIES
from src.features.datasets import TemporalSequenceDataset
from src.features.transforms import FlowFeatureTransformer
from src.models.base import CheckpointManager
from src.models.isolation_forest import IsolationForestThreatModel
from src.models.lstm_sequence import LSTMEvaluator, LSTMSequenceModel, LSTMTrainer
from src.models.random_forest import RandomForestThreatModel
from src.utils.config import PathResolver

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
NOTEBOOK_DIR = WORKSPACE_ROOT / "Notebook"
SAMPLE_PARQUET = REPO_ROOT / "data" / "public_loaders" / "sample_data" / "nfuq_sample_1500.parquet"

CANONICAL_TRANSFORMER_STATE_KEYS = {
    "feature_cols",
    "clip_quantiles",
    "output_dtype",
    "medians",
    "lower_bounds",
    "upper_bounds",
    "mean",
    "scale",
    "var",
    "n_samples_seen",
}


def _exec_notebook_cells(nb_path: Path, cell_indices: tuple[int, ...], extra_ns: Dict[str, Any] | None = None) -> Dict[str, Any]:
    nb = nbformat.read(str(nb_path), as_version=4)
    ns: Dict[str, Any] = dict(extra_ns or {})
    for idx in cell_indices:
        src = nb.cells[idx].source
        exec(compile(src, f"{nb_path.name}:cell_{idx}", "exec"), ns)
    return ns


def _assert_no_main_in_joblib(joblib_path: Path) -> None:
    raw = joblib_path.read_bytes()
    # joblib compress=3 uses zlib compression; scan raw stream for any __main__ pickle reference
    try:
        uncompressed = zlib.decompress(raw)
    except zlib.error:
        uncompressed = raw
    assert b"__main__" not in uncompressed, f"Found unpickleable __main__ reference in {joblib_path}"
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        assert not hasattr(main_mod, "RandomForestThreatModel")
        assert not hasattr(main_mod, "IsolationForestThreatModel")


def test_cross_env_notebook_checkpoints_reload_without_main(tmp_path: Path) -> None:
    from tests.e2e.conftest import execute_notebook_headless

    # 1. Execute Week 5 notebook headlessly on sample and verify cross-env reload via src/
    w5_ckpt = tmp_path / "w5_ckpt"
    execute_notebook_headless(
        NOTEBOOK_DIR / "week5_rules_random_forest.ipynb",
        data_dir=SAMPLE_PARQUET,
        checkpoint_dir=w5_ckpt,
        timeout=300,
    )
    for fname in ("week5_rf_binary.joblib", "week5_rf_multiclass.joblib"):
        _assert_no_main_in_joblib(w5_ckpt / fname)

    bin_payload = joblib.load(w5_ckpt / "week5_rf_binary.joblib")
    assert "estimator" in bin_payload and "model_state" in bin_payload and "transformer_state" in bin_payload
    rf_bin = RandomForestThreatModel.from_checkpoint(bin_payload)
    tf_w5 = FlowFeatureTransformer.from_state(bin_payload["transformer_state"])
    assert set(bin_payload["transformer_state"].keys()) == CANONICAL_TRANSFORMER_STATE_KEYS

    splits = NFUQDataLoader(PathResolver.resolve(prefer_sample=True)).load_tabular_splits()
    X_test_w5 = tf_w5.transform(splits["test"])
    probs_w5 = rf_bin.predict_proba(X_test_w5)
    assert probs_w5.shape == (len(splits["test"]), 2)

    mc_payload = joblib.load(w5_ckpt / "week5_rf_multiclass.joblib")
    rf_mc = RandomForestThreatModel.from_checkpoint(mc_payload)
    assert rf_mc.task == "multiclass"
    assert rf_mc.predict(X_test_w5).shape == (len(splits["test"]),)

    # 2. Execute Week 6 notebook headlessly on sample and verify cross-env reload via src/
    w6_ckpt = tmp_path / "w6_ckpt"
    execute_notebook_headless(
        NOTEBOOK_DIR / "week6_isolation_forest_anomaly.ipynb",
        data_dir=SAMPLE_PARQUET,
        checkpoint_dir=w6_ckpt,
        timeout=300,
    )
    _assert_no_main_in_joblib(w6_ckpt / "week6_isolation_forest.joblib")
    if_payload = joblib.load(w6_ckpt / "week6_isolation_forest.joblib")
    assert "estimator" in if_payload and "model_state" in if_payload and "transformer_state" in if_payload
    if_model = IsolationForestThreatModel.from_checkpoint(if_payload)
    tf_w6 = FlowFeatureTransformer.from_state(if_payload["transformer_state"])
    ctx = if_model.predict_with_context(tf_w6.transform(splits["test"]), df_raw=splits["test"])
    assert len(ctx["reason_codes"]) == len(splits["test"])

    # 3. Execute Week 7 notebook headlessly on sample and verify transformer_state + CPU state_dict
    w7_ckpt = tmp_path / "w7_ckpt"
    execute_notebook_headless(
        NOTEBOOK_DIR / "week7_pytorch_lstm_temporal.ipynb",
        data_dir=SAMPLE_PARQUET,
        checkpoint_dir=w7_ckpt,
        timeout=300,
    )
    pt_path = w7_ckpt / "week7_lstm_temporal.pt"
    meta_path = w7_ckpt / "week7_lstm_temporal_metadata.json"
    assert pt_path.is_file() and meta_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "transformer_state" in meta
    assert set(meta["transformer_state"].keys()) == CANONICAL_TRANSFORMER_STATE_KEYS
    for key in ("seq_len", "stride", "seed", "device", "metrics", "environment"):
        assert key in meta

    tf_w7 = FlowFeatureTransformer.from_state(meta["transformer_state"])
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    assert all(isinstance(v, torch.Tensor) and not v.is_cuda for v in state_dict.values())

    lstm_reloaded = LSTMSequenceModel(
        input_dim=40,
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        seq_len=meta["seq_len"],
        stride=meta["stride"],
        device="cpu",
    )
    lstm_reloaded.load_state_dict(state_dict)
    test_seq_ds = TemporalSequenceDataset(splits["test"], tf_w7, seq_len=meta["seq_len"], stride=meta["stride"])
    assert lstm_reloaded.predict_proba(test_seq_ds).shape == (len(test_seq_ds),)


def test_src_and_notebook_if_predict_with_context_nan_safety_and_reasons() -> None:
    splits = NFUQDataLoader(PathResolver.resolve(prefer_sample=True)).load_tabular_splits()
    tf = FlowFeatureTransformer().fit(splits["train"])
    X_tr = tf.transform(splits["train"])
    y_tr = splits["train"]["label"].to_numpy()
    X_val = tf.transform(splits["val"])
    y_val = splits["val"]["label"].to_numpy()

    # Construct df_raw containing NaNs and trigger conditions
    df_raw = splits["test"].iloc[:6].copy().reset_index(drop=True)
    df_raw.loc[0, ["is_tls_port", "is_ssh_port", "bytes_ratio"]] = [np.nan, np.nan, np.nan]
    df_raw.loc[1, ["is_tls_port", "is_ssh_port", "bytes_ratio"]] = [1, 0, 120.0]
    df_raw.loc[2, ["is_tls_port", "is_ssh_port", "bytes_ratio"]] = [0, 1, 10.0]
    X_probe = tf.transform(df_raw)

    # 1. src/ IsolationForestThreatModel
    src_model = IsolationForestThreatModel(n_estimators=20, random_state=42).fit(X_tr, y_tr)
    src_model.calibrate_threshold(X_val, y_val, target_fpr=0.05)
    # Force threshold very low so all 6 rows trigger anomaly reasons
    out_src = src_model.predict_with_context(X_probe, df_raw=df_raw, threshold=-10.0)
    assert len(out_src["reason_codes"]) == 6
    assert "CONTEXT_TLS_PORT_FLOW" in out_src["reason_codes"][1]
    assert "CONTEXT_HIGH_BYTES_ASYMMETRY" in out_src["reason_codes"][1]
    assert "CONTEXT_SSH_PORT_FLOW" in out_src["reason_codes"][2]

    # 2. Notebook week6 IsolationForestThreatModel (Cell index 10)
    nb_ns = _exec_notebook_cells(
        NOTEBOOK_DIR / "week6_isolation_forest_anomaly.ipynb",
        (10,),
        extra_ns={
            "np": np,
            "pd": pd,
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
        },
    )
    NbIFCls = nb_ns["IsolationForestThreatModel"]
    nb_model = NbIFCls(n_estimators=20, random_state=42).fit(X_tr, y_tr)
    nb_model.calibrate_threshold(X_val, y_val, target_fpr=0.05)
    out_nb = nb_model.predict_with_context(X_probe, df_raw=df_raw, threshold=-10.0)
    assert out_nb["reason_codes"] == out_src["reason_codes"]


def test_src_lstm_tensor_fit_extract_labels_and_default_batch_size() -> None:
    # 1. Default batch_size == 256
    assert inspect.signature(LSTMTrainer.__init__).parameters["batch_size"].default == 256

    # 2. LSTMSequenceModel.fit(torch.Tensor, torch.Tensor) works without dropping y
    rng = np.random.default_rng(42)
    X_t = torch.from_numpy(rng.normal(size=(16, 4, 40)).astype(np.float32))
    y_t = torch.tensor([i % 2 for i in range(16)], dtype=torch.float32)

    model = LSTMSequenceModel(input_dim=40, hidden_dim=8, num_layers=1, seq_len=4, stride=2, device="cpu")
    model.fit(X_t, y_t, epochs=1, batch_size=8, seed=42)
    assert model.is_fitted is True
    assert model.predict_proba(X_t).shape == (16,)

    # 3. _extract_labels raises ValueError when dataset.y is None
    splits = NFUQDataLoader(PathResolver.resolve(prefer_sample=True)).load_tabular_splits()
    tf = FlowFeatureTransformer().fit(splits["train"])
    unlabeled_ds = TemporalSequenceDataset(splits["train"], tf, seq_len=4, stride=2, label_col=None)
    assert unlabeled_ds.y is None
    with pytest.raises(ValueError, match="cannot be None"):
        LSTMTrainer._extract_labels(unlabeled_ds)
    with pytest.raises(ValueError, match="cannot be None"):
        LSTMEvaluator._extract_labels(unlabeled_ds)

    # 4. _extract_labels detaches tensors safely (e.g. requires_grad=True)
    class GradLabelDataset:
        def __init__(self) -> None:
            self.y = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)

        def __len__(self) -> int:
            return 3

    y_tr_ext = LSTMTrainer._extract_labels(GradLabelDataset())
    y_ev_ext = LSTMEvaluator._extract_labels(GradLabelDataset())
    assert np.array_equal(y_tr_ext, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.array_equal(y_ev_ext, np.array([0, 1, 0], dtype=np.int64))


@pytest.mark.parametrize(
    "nb_name",
    (
        "week5_rules_random_forest.ipynb",
        "week6_isolation_forest_anomaly.ipynb",
        "week7_pytorch_lstm_temporal.ipynb",
    ),
)
def test_notebook_sample_split_no_unseen_leakage_and_full_privacy(nb_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPIC04_DATA_DIR", str(SAMPLE_PARQUET))
    monkeypatch.setenv("TOPIC04_CHECKPOINT_DIR", str(tmp_path / "ckpt"))

    ns = _exec_notebook_cells(NOTEBOOK_DIR / nb_name, (2, 4))
    splits = ns["splits"]
    loader = ns["loader"]

    # 1. Sample-mode unseen attacks must NEVER leak into train or val
    train_fams = set(splits["train"]["attack_family"].astype(str).unique())
    val_fams = set(splits["val"]["attack_family"].astype(str).unique())
    unseen_fams = set(splits["unseen_holdout"]["attack_family"].astype(str).unique())
    assert train_fams.isdisjoint(UNSEEN_ATTACK_FAMILIES)
    assert val_fams.isdisjoint(UNSEEN_ATTACK_FAMILIES)
    assert len(splits["unseen_holdout"]) > 0
    assert unseen_fams.issubset(UNSEEN_ATTACK_FAMILIES)

    # 2. Full-column IPv4 detection past row 200 (row 300)
    df_bad_ip = pd.read_parquet(SAMPLE_PARQUET).iloc[:350].copy()
    df_bad_ip.loc[300, "src_host_token"] = "10.0.0.42"
    with pytest.raises(ValueError, match="Privacy Validator FAILED"):
        loader.validate_dataframe_privacy(df_bad_ip, "late_ipv4")

    # 3. Newly ported L7 patterns (payload, raw_sni, ssh_keystroke, dns_ttl_, L7_PROTO)
    for bad_col in ("payload_bytes", "raw_sni", "raw_san", "ssh_keystrokes", "dns_ttl_answer", "L7_PROTO"):
        df_bad_col = pd.read_parquet(SAMPLE_PARQUET).iloc[:10].copy()
        df_bad_col[bad_col] = "bad"
        with pytest.raises(ValueError, match="Privacy Validator FAILED"):
            loader.validate_dataframe_privacy(df_bad_col, f"bad_{bad_col}")

    # 4. Missing required L3/L4 column rejection
    df_missing = pd.read_parquet(SAMPLE_PARQUET).iloc[:10].drop(columns=[L3_L4_FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing required L3/L4"):
        loader.validate_dataframe_privacy(df_missing, "missing_col")


def test_notebook_kaggle_input_slug_scan_fallback(tmp_path: Path) -> None:
    nb_path = NOTEBOOK_DIR / "week5_rules_random_forest.ipynb"
    nb = nbformat.read(str(nb_path), as_version=4)
    # Compile Cell 2 class definitions without executing top-level CONFIG/PATHS resolution
    src_lines = nb.cells[2].source.splitlines()
    cut_idx = next(i for i, line in enumerate(src_lines) if line.startswith("CONFIG = "))
    ns: Dict[str, Any] = {}
    exec(compile("\n".join(src_lines[:cut_idx]), "cell2_defs", "exec"), ns)

    kaggle_input = tmp_path / "kaggle" / "input"
    slug_dataset = kaggle_input / "my-custom-slug" / "NF-UQ-NIDS-v2"
    (slug_dataset / "train").mkdir(parents=True)
    (slug_dataset / "train" / "flows_tabular.parquet").write_bytes(b"placeholder")

    found = ns["PathResolver"]._scan_kaggle_input(kaggle_input, max_depth=4)
    assert found == slug_dataset


def test_lstm_evaluator_empty_and_none_label_guards() -> None:
    splits = NFUQDataLoader(PathResolver.resolve(prefer_sample=True)).load_tabular_splits()
    tf = FlowFeatureTransformer().fit(splits["train"])

    empty_seq_ds = TemporalSequenceDataset(splits["train"].iloc[:1], tf, seq_len=8, stride=4)
    assert len(empty_seq_ds) == 0
    assert empty_seq_ds.y is not None

    unlabeled_ds = TemporalSequenceDataset(splits["train"], tf, seq_len=4, stride=2, label_col=None)
    assert len(unlabeled_ds) > 0 and unlabeled_ds.y is None

    empty_unlabeled_ds = TemporalSequenceDataset(splits["train"].iloc[:1], tf, seq_len=8, stride=4, label_col=None)
    assert len(empty_unlabeled_ds) == 0 and empty_unlabeled_ds.y is None

    # 1. src/ LSTMEvaluator: empty dataset returns clean JSON-safe dict; _safe_pr_auc guards empty; None labels raise ValueError
    assert LSTMEvaluator._safe_pr_auc(np.array([], dtype=np.int64), np.array([], dtype=np.float64)) == 0.0

    src_model = LSTMSequenceModel(input_dim=40, hidden_dim=8, num_layers=1, seq_len=8, stride=4, device="cpu")
    src_evaluator = LSTMEvaluator()
    empty_metrics = src_evaluator.evaluate(src_model, empty_seq_ds)
    assert empty_metrics["n_sequences"] == 0
    assert empty_metrics["roc_auc"] == 0.5
    assert empty_metrics["pr_auc"] == 0.0
    assert empty_metrics["confusion_matrix"] == [[0, 0], [0, 0]]
    assert empty_metrics["seq_len"] == 8
    assert empty_metrics["stride"] == 4
    json.dumps(empty_metrics, allow_nan=False)

    with pytest.raises(ValueError, match="cannot be None"):
        src_evaluator.evaluate(src_model, unlabeled_ds)
    with pytest.raises(ValueError, match="cannot be None"):
        src_evaluator.evaluate(src_model, empty_unlabeled_ds)

    # 2. Week 7 notebook Cell 12 LSTMEvaluator: raises ValueError (not TypeError) when ds.y is None
    nb_ns = _exec_notebook_cells(
        NOTEBOOK_DIR / "week7_pytorch_lstm_temporal.ipynb",
        (10, 12),
        extra_ns={
            "np": np,
            "torch": torch,
            "DataLoader": DataLoader,
            "TemporalSequenceDataset": TemporalSequenceDataset,
            "Path": Path,
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Mapping": Mapping,
            "Optional": Optional,
            "Tuple": Tuple,
        },
    )
    nb_model = nb_ns["LSTMSequenceModel"](input_dim=40, hidden_dim=8, num_layers=1, seq_len=8, stride=4, device="cpu")
    nb_evaluator = nb_ns["LSTMEvaluator"]

    nb_empty_metrics = nb_evaluator.evaluate(nb_model, empty_seq_ds)
    assert nb_empty_metrics["n_sequences"] == 0
    assert nb_empty_metrics["confusion_matrix"] == [[0, 0], [0, 0]]
    json.dumps(nb_empty_metrics, allow_nan=False)

    with pytest.raises(ValueError, match="cannot be None"):
        nb_evaluator.evaluate(nb_model, unlabeled_ds)
    with pytest.raises(ValueError, match="cannot be None"):
        nb_evaluator.evaluate(nb_model, empty_unlabeled_ds)
