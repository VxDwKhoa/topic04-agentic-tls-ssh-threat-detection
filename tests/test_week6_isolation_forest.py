"""Unit tests for Week 6 Isolation Forest Anomaly Detection & SR-04 Calibration."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.features.data_loader import NFUQDataLoader
from src.features.transforms import FlowFeatureTransformer
from src.models.base import CheckpointManager
from src.models.isolation_forest import (
    IsolationForestEvaluator,
    IsolationForestThreatModel,
    IsolationForestTrainer,
)
from src.utils.config import PathResolver, set_global_seed


def test_week6_isolation_forest_benign_fit_calibration_and_checkpoint(tmp_path: Path):
    set_global_seed(42)
    paths = PathResolver.resolve(checkpoint_dir=tmp_path, prefer_sample=True)
    splits = NFUQDataLoader(paths).load_tabular_splits()

    transformer = FlowFeatureTransformer().fit(splits["train"])
    X_train = transformer.transform(splits["train"])
    y_train = splits["train"]["label"].to_numpy()
    X_val = transformer.transform(splits["val"])
    y_val = splits["val"]["label"].to_numpy()
    X_test = transformer.transform(splits["test"])
    y_test = splits["test"]["label"].to_numpy()

    model = IsolationForestThreatModel(n_estimators=40, random_state=42)
    summary = IsolationForestTrainer(model, target_fpr=0.05).train(X_train, y_train, X_val, y_val)

    assert summary["benign_only_fit"] is True
    assert summary["n_train_benign"] == int((y_train == 0).sum())
    assert "calibration" in summary

    ctx = model.predict_with_context(X_test, df_raw=splits["test"])
    assert len(ctx["reason_codes"]) == len(X_test)
    assert np.all((ctx["confidence"] >= 0.0) & (ctx["confidence"] <= 1.0))

    evaluator = IsolationForestEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)
    for k in ("calibrated_threshold", "aucpr", "roc_auc", "tpr_at_fpr"):
        assert k in metrics and np.isfinite(float(metrics[k]))

    mgr = CheckpointManager(tmp_path)
    mgr.save_checkpoint(
        "week6_isolation_forest",
        {"model": model, "transformer_state": transformer.get_state()},
        {"seed": 42, "calibrated_threshold": model.calibrated_threshold_, "metrics": metrics},
    )
    loaded, meta = mgr.load_checkpoint("week6_isolation_forest")
    assert meta["calibrated_threshold"] == model.calibrated_threshold_
    assert np.allclose(model.score_samples(X_test), loaded["model"].score_samples(X_test), atol=1e-6)
