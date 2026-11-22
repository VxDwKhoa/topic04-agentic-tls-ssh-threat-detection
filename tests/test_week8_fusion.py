"""Unit tests for Week 8 Calibrated Multi-Model Fusion Engine."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.models.base import CheckpointManager
from src.models.fusion import COMPONENT_NAMES, CalibratedFusionModel, FusionEvaluator


def test_calibrated_fusion_simplex_weights_decomposable_and_checkpoint(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 400
    y = np.concatenate([np.zeros(200, dtype=np.int64), np.ones(200, dtype=np.int64)])

    scores = {
        "rule": np.clip(y * 0.4 + rng.normal(0.1, 0.15, n), 0.0, 1.0),
        "classifier": np.clip(y * 0.85 + rng.normal(0.05, 0.08, n), 0.0, 1.0),
        "anomaly": np.clip(y * 0.5 + rng.normal(0.2, 0.15, n), 0.0, 1.0),
        "sequence": np.clip(y * 0.9 + rng.normal(0.03, 0.06, n), 0.0, 1.0),
    }

    model = CalibratedFusionModel(min_anomaly_weight=0.05, min_rule_weight=0.02)
    model.fit(scores, y)

    w = model.weights
    assert set(w.keys()) == set(COMPONENT_NAMES)
    assert np.isclose(sum(w.values()), 1.0, atol=1e-6)
    assert w["anomaly"] >= 0.05 - 1e-6
    assert w["rule"] >= 0.02 - 1e-6
    assert all(v >= 0.0 for v in w.values())

    # Verify exact decomposability: sum(weighted_contributions) == s_total
    evidence = model.decompose_evidence(scores, top_n=10)
    assert len(evidence) == 10
    for rec in evidence:
        assert np.isclose(sum(rec["weighted_contributions"].values()), rec["s_total"], atol=1e-6)
        assert rec["dominant_signal"] in COMPONENT_NAMES

    # Verify Evaluator & Comparison table
    evaluator = FusionEvaluator()
    comp = evaluator.compare_all(model, scores, y)
    assert len(comp) == 5
    metrics = evaluator.evaluate(model, scores, y)
    assert metrics["f1"] > 0.95

    # Verify Checkpoint Save / Reload Parity
    mgr = CheckpointManager(tmp_path)
    mgr.save_checkpoint("week8_fusion_test", {"model_state": model.get_metadata()}, {"seed": 42})
    loaded, _ = mgr.load_checkpoint("week8_fusion_test")
    reloaded = CalibratedFusionModel.from_checkpoint(loaded)
    assert np.allclose(model.score_samples(scores), reloaded.score_samples(scores), atol=1e-6)
    assert np.array_equal(model.predict(scores), reloaded.predict(scores))
