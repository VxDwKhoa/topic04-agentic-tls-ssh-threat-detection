"""
Tier 1 E2E Tests: Week 5 Rules + Random Forest, Week 6 Isolation Forest (SR-04),
and Week 7 Pure PyTorch LSTM Temporal Model.

All `src.*` imports are performed lazily inside test functions for collection safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .conftest import (
    EXPECTED_L3_L4_FEATURE_COLUMNS,
    discover_class,
    import_src_module,
    make_synthetic_l3_l4_flows,
)


def _extract_reason_codes_output(model: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """Call canonical prediction/explanation methods on Rules or IF models."""
    if hasattr(model, "predict_with_reasons"):
        out = model.predict_with_reasons(df)
        if isinstance(out, dict):
            return out
    if hasattr(model, "predict_with_explanations"):
        out = model.predict_with_explanations(df)
        if isinstance(out, dict):
            return out
    preds = np.asarray(model.predict(df))
    scores = (
        np.asarray(model.predict_proba(df))
        if hasattr(model, "predict_proba")
        else np.asarray(model.score_samples(df))
    )
    reasons = model.explain(df) if hasattr(model, "explain") else getattr(model, "last_reason_codes_", None)
    return {"predictions": preds, "scores": scores, "reason_codes": reasons}


# ---------------------------------------------------------------------------
# F7: Week 5 Deterministic Rules Baseline (5 tests)
# ---------------------------------------------------------------------------


def test_rules_baseline_returns_bounded_scores_binary_preds_and_reason_codes(
    synthetic_clean_splits,
):
    base_mod = import_src_module("src.models.base")
    rules_mod = import_src_module("src.models.rules_baseline")
    RulesCls = discover_class(
        rules_mod,
        candidate_names=("RulesBaselineModel", "DeterministicRulesModel", "RulesEngine", "RulesThreatModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    model = RulesCls()
    df_test = synthetic_clean_splits["test"]
    preds = np.asarray(model.predict(df_test))
    scores = np.asarray(
        model.predict_proba(df_test) if hasattr(model, "predict_proba") else model.score_samples(df_test)
    )
    if scores.ndim == 2 and scores.shape[1] == 2:
        scores = scores[:, 1]

    assert preds.shape == (len(df_test),)
    assert set(np.unique(preds)).issubset({0, 1})
    assert scores.shape == (len(df_test),)
    assert np.isfinite(scores).all()
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    details = _extract_reason_codes_output(model, df_test)
    reasons = details.get("reason_codes")
    assert reasons is not None and len(reasons) == len(df_test)


def test_rules_baseline_is_deterministic_across_repeated_calls(synthetic_clean_splits):
    base_mod = import_src_module("src.models.base")
    rules_mod = import_src_module("src.models.rules_baseline")
    RulesCls = discover_class(
        rules_mod,
        candidate_names=("RulesBaselineModel", "DeterministicRulesModel", "RulesEngine", "RulesThreatModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    model = RulesCls()
    df_val = synthetic_clean_splits["val"]
    p1 = np.asarray(model.predict(df_val))
    p2 = np.asarray(model.predict(df_val))
    assert np.array_equal(p1, p2)

    s1 = np.asarray(model.predict_proba(df_val) if hasattr(model, "predict_proba") else model.score_samples(df_val))
    s2 = np.asarray(model.predict_proba(df_val) if hasattr(model, "predict_proba") else model.score_samples(df_val))
    assert np.allclose(s1, s2, atol=0.0)


def test_rules_baseline_flags_suspicious_l3_l4_patterns_with_non_empty_reasons():
    base_mod = import_src_module("src.models.base")
    rules_mod = import_src_module("src.models.rules_baseline")
    RulesCls = discover_class(
        rules_mod,
        candidate_names=("RulesBaselineModel", "DeterministicRulesModel", "RulesEngine", "RulesThreatModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df = make_synthetic_l3_l4_flows(num_rows=6, seed=7, attack_families=("Benign",), benign_ratio=1.0)
    # Row 0..2: quiet benign TLS flow; Row 3..5: aggressive SYN flood / SSH brute-force / high retransmit
    df.loc[3:, "dst_port"] = 22
    df.loc[3:, "is_ssh_port"] = 1
    df.loc[3:, "is_tls_port"] = 0
    df.loc[3:, "in_pkts"] = 500
    df.loc[3:, "out_pkts"] = 1
    df.loc[3:, "in_bytes"] = 60000
    df.loc[3:, "out_bytes"] = 60
    df.loc[3:, "flow_duration_ms"] = 1
    df.loc[3:, "tcp_flags"] = 2
    df.loc[3:, "client_tcp_flags"] = 2
    df.loc[3:, "retransmitted_in_pkts"] = 100
    df.loc[3:, "bytes_ratio"] = 1000.0
    df.loc[3:, "pkts_ratio"] = 500.0

    model = RulesCls()
    details = _extract_reason_codes_output(model, df)
    preds = np.asarray(details["predictions"])
    reasons = details["reason_codes"]

    assert preds[3:].sum() >= 1, "Suspicious L3/L4 burst rows should trigger at least one rule"
    for idx in range(3, 6):
        if preds[idx] == 1:
            assert len(reasons[idx]) > 0, f"Flagged row {idx} must emit non-empty reason_codes"


def test_rules_baseline_operates_without_l7_columns(synthetic_clean_splits):
    base_mod = import_src_module("src.models.base")
    rules_mod = import_src_module("src.models.rules_baseline")
    RulesCls = discover_class(
        rules_mod,
        candidate_names=("RulesBaselineModel", "DeterministicRulesModel", "RulesEngine", "RulesThreatModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df_only_l3_l4 = synthetic_clean_splits["test"][EXPECTED_L3_L4_FEATURE_COLUMNS].copy()
    model = RulesCls()
    preds = np.asarray(model.predict(df_only_l3_l4))
    assert preds.shape == (len(df_only_l3_l4),)


def test_rules_baseline_supports_optional_l5_l6_hooks_safely(synthetic_clean_splits):
    base_mod = import_src_module("src.models.base")
    rules_mod = import_src_module("src.models.rules_baseline")
    RulesCls = discover_class(
        rules_mod,
        candidate_names=("RulesBaselineModel", "DeterministicRulesModel", "RulesEngine", "RulesThreatModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df_with_l5_l6 = synthetic_clean_splits["val"].copy()
    df_with_l5_l6["ja4"] = "t13d1516h2_8daaf6152771_e5627efa2ab1"
    df_with_l5_l6["ech_observed"] = 0

    model = RulesCls()
    preds = np.asarray(model.predict(df_with_l5_l6))
    assert preds.shape == (len(df_with_l5_l6),)
    meta = model.get_metadata()
    assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# F8: Week 5 Random Forest (Binary + Multi-Class) (6 tests)
# ---------------------------------------------------------------------------


def test_week5_random_forest_binary_fit_predict_and_metrics(synthetic_clean_splits):
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
    X_test = transformer.transform(synthetic_clean_splits["test"])
    y_test = synthetic_clean_splits["test"]["label"].to_numpy()

    model = RFModelCls(task="binary", random_state=42, n_estimators=20)
    model.fit(X_train, y_train)
    probs = np.asarray(model.predict_proba(X_test))
    preds = np.asarray(model.predict(X_test))
    assert preds.shape == (len(X_test),)
    assert probs.shape[0] == len(X_test)

    evaluator = RFEvalCls()
    metrics = evaluator.evaluate_binary(y_test, preds, probs[:, 1] if probs.ndim == 2 else probs)
    for k in ("roc_auc", "pr_auc", "f1"):
        assert k in metrics, f"Binary RF metrics missing '{k}'"
        assert 0.0 <= float(metrics[k]) <= 1.0


def test_week5_random_forest_multiclass_attack_family_metrics(synthetic_clean_splits):
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
    label_enc = transforms_mod.TargetLabelEncoder().fit(synthetic_clean_splits["train"]["attack_family"])

    X_train = transformer.transform(synthetic_clean_splits["train"])
    y_train = label_enc.transform(synthetic_clean_splits["train"]["attack_family"])
    X_val = transformer.transform(synthetic_clean_splits["val"])
    y_val = label_enc.transform(synthetic_clean_splits["val"]["attack_family"])

    model = RFModelCls(task="multiclass", random_state=42, n_estimators=20)
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_val))

    evaluator = RFEvalCls()
    metrics = evaluator.evaluate_multiclass(y_val, preds, class_names=label_enc.classes_)
    for k in ("macro_f1", "micro_f1", "per_class_recall", "confusion_matrix"):
        assert k in metrics, f"Multi-class RF metrics missing '{k}'"
    assert 0.0 <= float(metrics["macro_f1"]) <= 1.0
    assert 0.0 <= float(metrics["micro_f1"]) <= 1.0


def test_week5_random_forest_feature_importance_40_l3_l4_features(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    rf_mod = import_src_module("src.models.random_forest")

    RFModelCls = discover_class(
        rf_mod,
        candidate_names=("RandomForestThreatModel", "Week5RandomForestModel", "RandomForestModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    X_train = transformer.transform(synthetic_clean_splits["train"])
    y_train = synthetic_clean_splits["train"]["label"].to_numpy()

    model = RFModelCls(task="binary", random_state=42, n_estimators=20)
    model.fit(X_train, y_train)
    fi = model.get_feature_importance(EXPECTED_L3_L4_FEATURE_COLUMNS)

    assert len(fi) == 40
    if isinstance(fi, dict):
        assert set(fi.keys()) == set(EXPECTED_L3_L4_FEATURE_COLUMNS)
        vals = np.array(list(fi.values()), dtype=float)
    elif isinstance(fi, pd.DataFrame):
        assert len(fi) == 40
        vals = fi.select_dtypes(include=[np.number]).iloc[:, 0].to_numpy(dtype=float)
    else:
        vals = np.asarray(fi, dtype=float)
    assert np.all(vals >= 0.0)
    assert np.isclose(vals.sum(), 1.0, atol=1e-4)


def test_week5_rules_vs_random_forest_comparison_output(synthetic_clean_splits):
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
    RFEvalCls = discover_class(
        rf_mod,
        candidate_names=("RandomForestEvaluator", "Week5Evaluator"),
        base_cls=base_mod.BaseEvaluator,
    )

    df_train = synthetic_clean_splits["train"]
    df_test = synthetic_clean_splits["test"]
    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
    X_train = transformer.transform(df_train)
    X_test = transformer.transform(df_test)
    y_train = df_train["label"].to_numpy()
    y_test = df_test["label"].to_numpy()

    rules_model = RulesCls()
    rules_preds = np.asarray(rules_model.predict(df_test))
    rules_scores = np.asarray(
        rules_model.predict_proba(df_test)
        if hasattr(rules_model, "predict_proba")
        else rules_model.score_samples(df_test)
    )
    if rules_scores.ndim == 2:
        rules_scores = rules_scores[:, 1]

    rf_model = RFModelCls(task="binary", random_state=42, n_estimators=20).fit(X_train, y_train)
    rf_preds = np.asarray(rf_model.predict(X_test))
    rf_probs = np.asarray(rf_model.predict_proba(X_test))
    rf_scores = rf_probs[:, 1] if rf_probs.ndim == 2 else rf_probs

    evaluator = RFEvalCls()
    rules_metrics = evaluator.evaluate_binary(y_test, rules_preds, rules_scores)
    rf_metrics = evaluator.evaluate_binary(y_test, rf_preds, rf_scores)
    comparison = evaluator.compare_baselines(rules_metrics, rf_metrics)

    assert isinstance(comparison, (dict, pd.DataFrame))
    if isinstance(comparison, pd.DataFrame):
        assert len(comparison) >= 2
    else:
        assert len(comparison) >= 2


def test_week5_random_forest_checkpoint_save_load_parity(
    synthetic_clean_splits,
    isolated_checkpoint_dir: Path,
):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    rf_mod = import_src_module("src.models.random_forest")

    RFModelCls = discover_class(
        rf_mod,
        candidate_names=("RandomForestThreatModel", "Week5RandomForestModel", "RandomForestModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    X_train = transformer.transform(synthetic_clean_splits["train"])
    y_train = synthetic_clean_splits["train"]["label"].to_numpy()
    X_test = transformer.transform(synthetic_clean_splits["test"])

    model = RFModelCls(task="binary", random_state=42, n_estimators=15).fit(X_train, y_train)
    orig_probs = np.asarray(model.predict_proba(X_test))
    orig_preds = np.asarray(model.predict(X_test))

    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    mgr.save_checkpoint(
        "week5_rf_binary",
        {"model": model, "transformer_state": transformer.get_state()},
        {
            "seed": 42,
            "versions": {"scikit-learn": "1.8.0"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "metrics": {"f1": 0.9},
        },
    )
    loaded_artifacts, loaded_meta = mgr.load_checkpoint("week5_rf_binary")
    reloaded = loaded_artifacts["model"]

    assert np.allclose(orig_probs, np.asarray(reloaded.predict_proba(X_test)), atol=1e-6)
    assert np.array_equal(orig_preds, np.asarray(reloaded.predict(X_test)))
    assert loaded_meta["seed"] == 42


def test_week5_random_forest_handles_unseen_classes_in_test_eval(synthetic_clean_splits):
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
    label_enc = transforms_mod.TargetLabelEncoder().fit(synthetic_clean_splits["train"]["attack_family"])

    X_train = transformer.transform(synthetic_clean_splits["train"])
    y_train = label_enc.transform(synthetic_clean_splits["train"]["attack_family"])
    X_test = transformer.transform(synthetic_clean_splits["test"])
    y_test = label_enc.transform(synthetic_clean_splits["test"]["attack_family"])
    assert (y_test == -1).any(), "Synthetic test must include unseen -1 classes"

    model = RFModelCls(task="multiclass", random_state=42, n_estimators=20).fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test))

    evaluator = RFEvalCls()
    metrics = evaluator.evaluate_multiclass(y_test, preds, class_names=label_enc.classes_)
    assert "macro_f1" in metrics and np.isfinite(metrics["macro_f1"])


# ---------------------------------------------------------------------------
# F9: Week 6 Isolation Forest + SR-04 Calibration + Unseen Eval (5 tests)
# ---------------------------------------------------------------------------


def test_week6_isolation_forest_fits_strictly_on_benign_train_rows(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    IFModelCls = discover_class(
        if_mod,
        candidate_names=("IsolationForestThreatModel", "Week6IsolationForestModel", "IsolationForestAnomalyModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df_train = synthetic_clean_splits["train"]
    benign_count = int((df_train["label"] == 0).sum())
    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
    X_train = transformer.transform(df_train)
    y_train = df_train["label"].to_numpy()

    model = IFModelCls(random_state=42, n_estimators=50)
    model.fit(X_train, y_train)
    meta = model.get_metadata()

    assert meta.get("benign_only_fit") is True or meta.get("n_train_benign") == benign_count


def test_week6_isolation_forest_val_threshold_calibration_sr04(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    IFModelCls = discover_class(
        if_mod,
        candidate_names=("IsolationForestThreatModel", "Week6IsolationForestModel", "IsolationForestAnomalyModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df_train = synthetic_clean_splits["train"]
    df_val = synthetic_clean_splits["val"]
    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
    X_train = transformer.transform(df_train)
    y_train = df_train["label"].to_numpy()
    X_val = transformer.transform(df_val)
    y_val = df_val["label"].to_numpy()

    model = IFModelCls(random_state=42, n_estimators=50).fit(X_train, y_train)
    calib = model.calibrate_threshold(X_val, y_val, target_fpr=0.05)

    threshold = calib["calibrated_threshold"] if isinstance(calib, dict) else float(calib)
    assert np.isfinite(threshold)
    preds_val = np.asarray(model.predict(X_val))
    benign_fpr = float(preds_val[y_val == 0].mean())
    assert 0.0 <= benign_fpr <= 0.15


def test_week6_isolation_forest_emits_reason_codes_and_confidence(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    IFModelCls = discover_class(
        if_mod,
        candidate_names=("IsolationForestThreatModel", "Week6IsolationForestModel", "IsolationForestAnomalyModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df_train = synthetic_clean_splits["train"]
    df_val = synthetic_clean_splits["val"]
    df_test = synthetic_clean_splits["test"]

    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
    model = IFModelCls(random_state=42, n_estimators=50).fit(
        transformer.transform(df_train), df_train["label"].to_numpy()
    )
    model.calibrate_threshold(transformer.transform(df_val), df_val["label"].to_numpy(), target_fpr=0.05)

    X_test = transformer.transform(df_test)
    out = model.predict_with_context(X_test, df_raw=df_test)
    assert "predictions" in out
    assert "anomaly_scores" in out
    assert "reason_codes" in out
    assert "confidence" in out
    assert len(out["reason_codes"]) == len(df_test)
    conf = np.asarray(out["confidence"], dtype=float)
    assert np.all((conf >= 0.0) & (conf <= 1.0))


def test_week6_isolation_forest_eval_paired_unseen_attack_holdout(synthetic_clean_splits):
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
    df_test = synthetic_clean_splits["test"]
    df_unseen = synthetic_clean_splits["unseen_holdout"]

    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
    model = IFModelCls(random_state=42, n_estimators=50).fit(
        transformer.transform(df_train), df_train["label"].to_numpy()
    )
    model.calibrate_threshold(transformer.transform(df_val), df_val["label"].to_numpy(), target_fpr=0.05)

    df_benign_test = df_test[df_test["label"] == 0].copy()
    evaluator = IFEvalCls()
    metrics = evaluator.evaluate_unseen_holdout(
        model=model,
        X_unseen=transformer.transform(df_unseen),
        y_unseen=df_unseen["label"].to_numpy(),
        X_benign_test=transformer.transform(df_benign_test),
        y_benign_test=df_benign_test["label"].to_numpy(),
    )

    for key in ("calibrated_threshold", "aucpr", "roc_auc", "tpr_at_fpr"):
        assert key in metrics, f"Week 6 unseen eval missing '{key}'"
        assert np.isfinite(float(metrics[key]))


def test_week6_isolation_forest_checkpoint_save_load_parity(
    synthetic_clean_splits,
    isolated_checkpoint_dir: Path,
):
    transforms_mod = import_src_module("src.features.transforms")
    base_mod = import_src_module("src.models.base")
    if_mod = import_src_module("src.models.isolation_forest")

    IFModelCls = discover_class(
        if_mod,
        candidate_names=("IsolationForestThreatModel", "Week6IsolationForestModel", "IsolationForestAnomalyModel"),
        base_cls=base_mod.BaseThreatModel,
    )

    df_train = synthetic_clean_splits["train"]
    df_val = synthetic_clean_splits["val"]
    df_test = synthetic_clean_splits["test"]

    transformer = transforms_mod.FlowFeatureTransformer().fit(df_train)
    model = IFModelCls(random_state=42, n_estimators=40).fit(
        transformer.transform(df_train), df_train["label"].to_numpy()
    )
    calib = model.calibrate_threshold(transformer.transform(df_val), df_val["label"].to_numpy(), target_fpr=0.05)

    X_test = transformer.transform(df_test)
    orig_scores = np.asarray(model.score_samples(X_test))
    orig_preds = np.asarray(model.predict(X_test))

    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    mgr.save_checkpoint(
        "week6_if_anomaly",
        {"model": model, "transformer_state": transformer.get_state()},
        {
            "seed": 42,
            "versions": {"scikit-learn": "1.8.0"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "calibrated_threshold": float(calib["calibrated_threshold"] if isinstance(calib, dict) else calib),
            "target_fpr": 0.05,
            "metrics": {"aucpr": 0.8},
        },
    )
    loaded_artifacts, loaded_meta = mgr.load_checkpoint("week6_if_anomaly")
    reloaded = loaded_artifacts["model"]

    assert np.allclose(orig_scores, np.asarray(reloaded.score_samples(X_test)), atol=1e-6)
    assert np.array_equal(orig_preds, np.asarray(reloaded.predict(X_test)))
    assert "calibrated_threshold" in loaded_meta


# ---------------------------------------------------------------------------
# F10: Week 7 Pure PyTorch LSTM Temporal Model (6 tests)
# ---------------------------------------------------------------------------


def test_week7_lstm_model_is_pure_nn_lstm_no_graph_features():
    import torch.nn as nn

    base_mod = import_src_module("src.models.base")
    lstm_mod = import_src_module("src.models.lstm_sequence")

    LSTMCls = discover_class(
        lstm_mod,
        candidate_names=("LSTMSequenceModel", "LSTMThreatModel", "Week7LSTMModel", "LSTMSequenceClassifier"),
        base_cls=base_mod.BaseThreatModel,
    )

    model = LSTMCls(input_dim=40, hidden_dim=32, num_layers=1, seq_len=8, stride=4)
    net = model.network if hasattr(model, "network") else model
    assert isinstance(net, nn.Module), "Week 7 model must wrap or subclass torch.nn.Module"

    submodules = list(net.modules())
    assert any(isinstance(m, nn.LSTM) for m in submodules), "Week 7 model must use torch.nn.LSTM"
    lstm_layer = next(m for m in submodules if isinstance(m, nn.LSTM))
    assert lstm_layer.input_size == 40, "Pure L3/L4 LSTM must take exactly 40 features (no graph features)"

    module_names = " ".join(type(m).__name__.lower() for m in submodules)
    for forbidden in ("gcn", "gat", "graph", "gnn", "sage"):
        assert forbidden not in module_names, f"Graph layer '{forbidden}' forbidden in Week 7 pure LSTM"


def test_week7_lstm_trainer_trains_on_3d_sequences_cpu_reproducible(synthetic_clean_splits):
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

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    train_ds = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["train"], transformer, seq_len=4, stride=2)
    val_ds = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["val"], transformer, seq_len=4, stride=2)

    model = LSTMCls(input_dim=40, hidden_dim=16, num_layers=1, seq_len=4, stride=2, device="cpu")
    trainer = TrainerCls(model=model, epochs=2, batch_size=16, lr=1e-3, seed=42, device="cpu")
    history = trainer.fit(train_ds, val_ds)

    assert isinstance(history, dict)
    probs = np.asarray(model.predict_proba(val_ds))
    assert probs.shape[0] == len(val_ds)
    assert np.isfinite(probs).all()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_week7_lstm_evaluator_outputs_sequence_metrics_and_latency(synthetic_clean_splits):
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
    test_ds = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["test"], transformer, seq_len=4, stride=2)

    model = LSTMCls(input_dim=40, hidden_dim=16, num_layers=1, seq_len=4, stride=2, device="cpu")
    TrainerCls(model=model, epochs=2, batch_size=16, lr=1e-3, seed=42, device="cpu").fit(train_ds)

    evaluator = EvalCls()
    metrics = evaluator.evaluate(model, test_ds)
    for k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "seq_len", "stride"):
        assert k in metrics, f"Week 7 LSTM metrics missing '{k}'"
    assert any("latency" in k or "events_per_sec" in k for k in metrics), "Week 7 metrics must report inference latency"
    assert metrics["seq_len"] == 4
    assert metrics["stride"] == 2


def test_week7_lstm_checkpoint_saves_pt_state_dict_and_seq_metadata(
    synthetic_clean_splits,
    isolated_checkpoint_dir: Path,
):
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

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    train_ds = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["train"], transformer, seq_len=4, stride=2)

    model = LSTMCls(input_dim=40, hidden_dim=16, num_layers=1, seq_len=4, stride=2, device="cpu")
    TrainerCls(model=model, epochs=1, batch_size=16, lr=1e-3, seed=42, device="cpu").fit(train_ds)

    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    mgr.save_checkpoint(
        "week7_lstm_seq",
        {"state_dict": model.network.state_dict() if hasattr(model, "network") else model.state_dict()},
        {
            "seed": 42,
            "versions": {"torch": "cpu"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "seq_len": 4,
            "stride": 2,
            "metrics": {"f1": 0.8},
        },
    )
    pt_files = list(isolated_checkpoint_dir.glob("*week7_lstm_seq*.pt"))
    meta_files = list(isolated_checkpoint_dir.glob("*week7_lstm_seq*_metadata.json"))
    assert pt_files, "Expected .pt state_dict file for Week 7 LSTM"
    assert meta_files, "Expected _metadata.json file for Week 7 LSTM"


def test_week7_lstm_checkpoint_reload_inference_parity(
    synthetic_clean_splits,
    isolated_checkpoint_dir: Path,
):
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

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    train_ds = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["train"], transformer, seq_len=4, stride=2)
    test_ds = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["test"], transformer, seq_len=4, stride=2)

    model = LSTMCls(input_dim=40, hidden_dim=16, num_layers=1, seq_len=4, stride=2, device="cpu")
    TrainerCls(model=model, epochs=2, batch_size=16, lr=1e-3, seed=42, device="cpu").fit(train_ds)
    orig_probs = np.asarray(model.predict_proba(test_ds))

    mgr = base_mod.CheckpointManager(isolated_checkpoint_dir)
    mgr.save_checkpoint(
        "week7_lstm_parity",
        {"state_dict": model.network.state_dict() if hasattr(model, "network") else model.state_dict()},
        {
            "seed": 42,
            "versions": {"torch": "cpu"},
            "feature_columns": EXPECTED_L3_L4_FEATURE_COLUMNS,
            "seq_len": 4,
            "stride": 2,
            "hidden_dim": 16,
            "num_layers": 1,
            "metrics": {"f1": 0.8},
        },
    )
    loaded_artifacts, loaded_meta = mgr.load_checkpoint("week7_lstm_parity")
    reloaded = LSTMCls(
        input_dim=40,
        hidden_dim=loaded_meta.get("hidden_dim", 16),
        num_layers=loaded_meta.get("num_layers", 1),
        seq_len=loaded_meta["seq_len"],
        stride=loaded_meta["stride"],
        device="cpu",
    )
    target_net = reloaded.network if hasattr(reloaded, "network") else reloaded
    target_net.load_state_dict(loaded_artifacts["state_dict"])
    target_net.eval()

    reloaded_probs = np.asarray(reloaded.predict_proba(test_ds))
    assert np.allclose(orig_probs, reloaded_probs, atol=1e-5)


def test_week7_lstm_explicit_seq_len_stride_reflected_in_metadata(synthetic_clean_splits):
    transforms_mod = import_src_module("src.features.transforms")
    datasets_mod = import_src_module("src.features.datasets")
    base_mod = import_src_module("src.models.base")
    lstm_mod = import_src_module("src.models.lstm_sequence")

    LSTMCls = discover_class(
        lstm_mod,
        candidate_names=("LSTMSequenceModel", "LSTMThreatModel", "Week7LSTMModel", "LSTMSequenceClassifier"),
        base_cls=base_mod.BaseThreatModel,
    )

    transformer = transforms_mod.FlowFeatureTransformer().fit(synthetic_clean_splits["train"])
    ds_4_2 = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["train"], transformer, seq_len=4, stride=2)
    ds_8_4 = datasets_mod.TemporalSequenceDataset(synthetic_clean_splits["train"], transformer, seq_len=8, stride=4)

    assert len(ds_4_2) > len(ds_8_4)
    model = LSTMCls(input_dim=40, hidden_dim=16, num_layers=1, seq_len=8, stride=4, device="cpu")
    meta = model.get_metadata()
    assert meta["seq_len"] == 8
    assert meta["stride"] == 4
