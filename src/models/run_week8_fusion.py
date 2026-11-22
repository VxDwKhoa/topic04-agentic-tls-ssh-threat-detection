r"""
Week 8 Calibrated Multi-Model Fusion Runner.
Loads the trained Kaggle checkpoints from `D:\Crypto\DoAn\Notebook\Checkpoint\`,
calibrates fusion weights (w_r, w_c, w_a, w_s) strictly on the Validation split,
evaluates on Test & paired Unseen Attack Holdout, and saves `week8_calibrated_fusion`
checkpoint + decomposable evidence sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.data_loader import NFUQDataLoader
from src.models.base import CheckpointManager
from src.models.fusion import CalibratedFusionModel, FusionEvaluator, MultiModelInferenceBundle
from src.utils.config import PathResolver, set_global_seed


def run_week8_fusion(
    checkpoint_root: str = r"D:\Crypto\DoAn\Notebook\Checkpoint",
    output_dir: str = r"D:\Crypto\DoAn\Notebook\Checkpoint\week8",
) -> None:
    versions = set_global_seed(42)
    paths = PathResolver.resolve(checkpoint_dir=output_dir)
    print(f"[INFO] Dataset mode: {paths.mode} | Path: {paths.dataset_dir}")

    # 1. Load Splits with strict No-L7 Privacy Validation
    splits = NFUQDataLoader(paths).load_tabular_splits()
    df_val = splits["val"]
    df_test = splits["test"]
    df_unseen = splits["unseen_holdout"]
    df_test_benign = df_test[df_test["label"] == 0].reset_index(drop=True)

    # 2. Load trained Week 5, Week 6, Week 7 checkpoints
    bundle = MultiModelInferenceBundle.from_checkpoint_root(checkpoint_root, device="cpu")
    print("[INFO] Loaded trained checkpoints: Rules + RF Binary (W5) + Isolation Forest (W6) + PyTorch LSTM (W7)")

    # 3. Compute constituent [0, 1] scores on Validation & calibrate convex weights
    val_scores, _ = bundle.compute_component_scores(df_val)
    y_val = df_val["label"].to_numpy(dtype=np.int64)

    fusion_model = CalibratedFusionModel(min_anomaly_weight=0.05, min_rule_weight=0.02)
    fusion_model.fit(val_scores, y_val)
    print("\n=== CALIBRATED FUSION WEIGHTS & THRESHOLD (VALIDATION) ===")
    print(json.dumps(fusion_model.calibration_summary_, indent=2))

    # 4. Evaluate on Full Test (`135,108` flows)
    test_scores, test_ctx = bundle.compute_component_scores(df_test)
    y_test = df_test["label"].to_numpy(dtype=np.int64)

    evaluator = FusionEvaluator()
    comp_test = evaluator.compare_all(fusion_model, test_scores, y_test)
    test_metrics = evaluator.evaluate(fusion_model, test_scores, y_test)
    print("\n=== FULL TEST COMPARISON: SINGLE MODELS vs CALIBRATED FUSION ===")
    print(comp_test.to_string(index=False))

    # 5. Evaluate on Paired Unseen Attack Holdout (`Shellcode`, `Worms` + Benign Test)
    df_unseen_pair = pd.concat([df_test_benign, df_unseen], axis=0, ignore_index=True)
    unseen_scores, _ = bundle.compute_component_scores(df_unseen_pair)
    y_unseen_pair = df_unseen_pair["label"].to_numpy(dtype=np.int64)

    comp_unseen = evaluator.compare_all(fusion_model, unseen_scores, y_unseen_pair)
    unseen_metrics = evaluator.evaluate(fusion_model, unseen_scores, y_unseen_pair)
    print("\n=== PAIRED UNSEEN ATTACK HOLDOUT COMPARISON ===")
    print(comp_unseen.to_string(index=False))

    # 6. Demonstrate Decomposable Evidence Payload for Week 9 AI Agent
    flagged_idx = np.flatnonzero(fusion_model.predict(test_scores) == 1)[:3]
    sample_sub_scores = {k: v[flagged_idx] for k, v in test_scores.items()}
    sample_rule_r = [test_ctx["rule_reasons"][i] for i in flagged_idx]
    sample_anom_r = [test_ctx["anomaly_reasons"][i] for i in flagged_idx]
    decomposed = fusion_model.decompose_evidence(sample_sub_scores, sample_rule_r, sample_anom_r)
    print("\n=== SAMPLE DECOMPOSABLE EVIDENCE FOR WEEK 9 AI AGENT ===")
    print(json.dumps(decomposed[0], indent=2))

    # 7. Save Checkpoint (`week8_calibrated_fusion.joblib` + `_metadata.json`)
    mgr = CheckpointManager(output_dir)
    saved = mgr.save_checkpoint(
        "week8_calibrated_fusion",
        {"model_state": fusion_model.get_metadata()},
        {
            "seed": 42,
            "versions": versions,
            "weights": fusion_model.weights,
            "threshold": fusion_model.threshold,
            "val_calibration": fusion_model.calibration_summary_,
            "test_metrics": test_metrics,
            "unseen_holdout_metrics": unseen_metrics,
        },
        fmt="joblib",
    )
    print(f"\n[SAVED] Fusion checkpoint: {saved['artifact_path']} & {saved['metadata_path']}")


if __name__ == "__main__":
    run_week8_fusion()
