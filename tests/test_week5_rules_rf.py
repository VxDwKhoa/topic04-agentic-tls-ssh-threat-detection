"""Unit tests for Week 5 Deterministic Rules Baseline and Random Forest (Binary & Multi-Class)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.features.data_loader import NFUQDataLoader
from src.features.transforms import FlowFeatureTransformer, TargetLabelEncoder
from src.models.base import CheckpointManager
from src.models.random_forest import RandomForestEvaluator, RandomForestThreatModel, RandomForestTrainer
from src.models.rules_baseline import RulesBaselineModel
from src.utils.config import PathResolver, set_global_seed
from data.public_loaders.preprocess_and_split_nfuq import L3_L4_FEATURE_COLUMNS


def test_week5_rules_and_rf_end_to_end_on_sample(tmp_path: Path):
    set_global_seed(42)
    paths = PathResolver.resolve(checkpoint_dir=tmp_path, prefer_sample=True)
    splits = NFUQDataLoader(paths).load_tabular_splits()

    transformer = FlowFeatureTransformer().fit(splits["train"])
    X_train = transformer.transform(splits["train"])
    X_val = transformer.transform(splits["val"])
    X_test = transformer.transform(splits["test"])
    y_train_bin = splits["train"]["label"].to_numpy()
    y_val_bin = splits["val"]["label"].to_numpy()
    y_test_bin = splits["test"]["label"].to_numpy()

    # 1. Deterministic Rules Baseline
    rules = RulesBaselineModel()
    rules_out = rules.predict_with_reasons(splits["test"])
    assert rules_out["predictions"].shape == (len(splits["test"]),)
    assert len(rules_out["reason_codes"]) == len(splits["test"])
    assert np.all((rules_out["scores"] >= 0.0) & (rules_out["scores"] <= 1.0))

    # 2. Random Forest Binary
    rf_bin = RandomForestThreatModel(task="binary", n_estimators=20, random_state=42)
    train_summary = RandomForestTrainer(rf_bin).train(X_train, y_train_bin, X_val, y_val_bin)
    assert train_summary["n_train_samples"] == len(y_train_bin)

    evaluator = RandomForestEvaluator()
    rules_metrics = evaluator.evaluate_binary(y_test_bin, rules_out["predictions"], rules_out["scores"])
    rf_bin_metrics = evaluator.evaluate(rf_bin, X_test, y_test_bin)
    comp = evaluator.compare_baselines(rules_metrics, rf_bin_metrics)
    assert len(comp) == 2

    fi = rf_bin.get_feature_importance(L3_L4_FEATURE_COLUMNS)
    assert len(fi) == 40
    assert np.isclose(sum(fi.values()), 1.0, atol=1e-4)

    # 3. Random Forest Multi-Class
    le = TargetLabelEncoder().fit(splits["train"]["attack_family"])
    y_train_mc = le.transform(splits["train"]["attack_family"])
    y_test_mc = le.transform(splits["test"]["attack_family"])

    rf_mc = RandomForestThreatModel(task="multiclass", n_estimators=20, random_state=42).fit(X_train, y_train_mc)
    mc_metrics = evaluator.evaluate(rf_mc, X_test, y_test_mc, class_names=le.classes_)
    assert "macro_f1" in mc_metrics and "micro_f1" in mc_metrics and "per_class_recall" in mc_metrics

    # 4. Checkpoint Save / Reload Parity
    mgr = CheckpointManager(tmp_path)
    mgr.save_checkpoint(
        "week5_rf_binary",
        {"model": rf_bin, "transformer_state": transformer.get_state()},
        {"seed": 42, "metrics": rf_bin_metrics},
    )
    loaded, meta = mgr.load_checkpoint("week5_rf_binary")
    assert meta["seed"] == 42
    assert np.allclose(rf_bin.predict_proba(X_test), loaded["model"].predict_proba(X_test), atol=1e-6)
