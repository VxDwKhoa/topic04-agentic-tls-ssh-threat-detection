"""
Week 6 Object-Oriented Isolation Forest Anomaly Detection & SR-04 Threshold Calibration.
Topic 04 — Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response.
Adheres strictly to:
- Section 10: Isolation Forest trained on benign-only baseline (`label == 0`).
- SR-04: Anomaly score alone cannot be labeled "attack" without calibrated threshold,
  reason codes, and calibrated confidence.
- Section 14 & 18: Evaluation on `unseen_attack_holdout.parquet` paired with benign test rows
  (calibrated_threshold, AUCPR, ROC-AUC, TPR at fixed FPR).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.base import BaseEvaluator, BaseThreatModel, BaseTrainer


class IsolationForestThreatModel(BaseThreatModel):
    """
    Benign-only Isolation Forest anomaly detector with Validation-set FPR threshold
    calibration (SR-04), calibrated confidence scores, and contextual reason codes.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Any = "auto",
        contamination: Any = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(model_name="week6_isolation_forest", task_type="anomaly")
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = float(max_features)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)

        self.estimator = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.benign_only_fit: bool = True
        self.n_train_benign: int = 0
        self.calibrated_threshold_: Optional[float] = None
        self.target_fpr_: float = 0.05
        self.score_scale_: float = 1.0
        self.val_benign_score_mean_: float = 0.0
        self.val_benign_score_std_: float = 1.0

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        if hasattr(X, "X"):
            arr = np.asarray(X.X, dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Input features contain NaN or Inf. Run FlowFeatureTransformer first.")
        return arr

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "IsolationForestThreatModel":
        """
        Fit strictly on benign (`label == 0`) training rows.
        If `y` is provided, filters `X[y == 0]`. If `y` is None, assumes `X` is already benign-only.
        """
        X_arr = self._to_numpy(X)
        if y is not None:
            y_arr = np.asarray(y)
            if len(y_arr) != len(X_arr):
                raise ValueError(f"Length mismatch: X={len(X_arr)}, y={len(y_arr)}")
            benign_mask = y_arr == 0
            if not np.any(benign_mask):
                raise ValueError("IsolationForestThreatModel.fit() requires at least one benign (label == 0) row.")
            X_benign = X_arr[benign_mask]
        else:
            X_benign = X_arr

        if len(X_benign) == 0:
            raise ValueError("Cannot fit IsolationForest on empty array.")

        self.estimator.fit(X_benign)
        self.n_train_benign = int(len(X_benign))
        self.benign_only_fit = True
        self.is_fitted = True

        # Default provisional threshold from train benign 95th percentile until calibrated on val
        train_scores = self.score_samples(X_benign)
        self.calibrated_threshold_ = float(np.quantile(train_scores, 0.95))
        std = float(np.std(train_scores))
        self.score_scale_ = std if std > 1e-6 else 1.0
        return self

    def score_samples(self, X: Any, **kwargs: Any) -> np.ndarray:
        """
        Return continuous anomaly scores where HIGHER = MORE ANOMALOUS.
        (Negates scikit-learn's `decision_function` where lower = more anomalous.)
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestThreatModel must be fitted before scoring.")
        X_arr = self._to_numpy(X)
        raw = -self.estimator.decision_function(X_arr)
        return np.asarray(raw, dtype=np.float64)

    def calibrate_threshold(
        self,
        X_val: Any,
        y_val: np.ndarray,
        target_fpr: float = 0.05,
    ) -> Dict[str, Any]:
        """
        SR-04 Validation-set threshold calibration:
        Sets `calibrated_threshold_` at the `(1 - target_fpr)` quantile of benign validation scores
        so false positive rate on benign traffic is controlled near `target_fpr`.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibrate_threshold().")
        if not (0.0 < target_fpr < 1.0):
            raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}.")

        y_arr = np.asarray(y_val)
        scores = self.score_samples(X_val)
        benign_mask = y_arr == 0
        if not np.any(benign_mask):
            raise ValueError("Validation split must contain benign (label == 0) rows for FPR calibration.")

        benign_scores = scores[benign_mask]
        threshold = float(np.quantile(benign_scores, 1.0 - target_fpr))
        self.calibrated_threshold_ = threshold
        self.target_fpr_ = float(target_fpr)
        self.val_benign_score_mean_ = float(np.mean(benign_scores))
        std = float(np.std(benign_scores))
        self.val_benign_score_std_ = std if std > 1e-6 else 1.0
        self.score_scale_ = self.val_benign_score_std_

        preds_val = (scores >= threshold).astype(np.int64)
        achieved_fpr = float(preds_val[benign_mask].mean())
        achieved_tpr = float(preds_val[~benign_mask].mean()) if np.any(~benign_mask) else 0.0

        return {
            "calibrated_threshold": threshold,
            "target_fpr": float(target_fpr),
            "achieved_val_fpr": achieved_fpr,
            "achieved_val_tpr": achieved_tpr,
            "n_val_benign": int(benign_mask.sum()),
            "n_val_attack": int((~benign_mask).sum()),
        }

    def predict(self, X: Any, threshold: Optional[float] = None, **kwargs: Any) -> np.ndarray:
        """Return binary anomaly flags {0, 1} using calibrated threshold."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        thr = float(threshold) if threshold is not None else float(self.calibrated_threshold_)
        scores = self.score_samples(X)
        return (scores >= thr).astype(np.int64)

    def _compute_confidence(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """Map distance from calibrated threshold to [0.0, 1.0] via numerically stable sigmoid."""
        z = np.clip((scores - threshold) / max(self.score_scale_, 1e-6), -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float64)

    def predict_with_context(
        self,
        X: Any,
        df_raw: Optional[pd.DataFrame] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        SR-04 compliant prediction output:
        Emits predictions, continuous anomaly_scores, calibrated confidence in [0, 1],
        and contextual reason_codes for every sample.
        """
        thr = float(threshold) if threshold is not None else float(self.calibrated_threshold_)
        scores = self.score_samples(X)
        preds = (scores >= thr).astype(np.int64)
        confidence = self._compute_confidence(scores, thr)

        n = len(scores)
        reason_codes: List[List[str]] = []

        n_raw = len(df_raw) if df_raw is not None else 0
        if df_raw is not None and n_raw > 0:
            tls_arr: Optional[np.ndarray] = (
                pd.to_numeric(df_raw["is_tls_port"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                if "is_tls_port" in df_raw.columns
                else None
            )
            ssh_arr: Optional[np.ndarray] = (
                pd.to_numeric(df_raw["is_ssh_port"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                if "is_ssh_port" in df_raw.columns
                else None
            )
            ratio_arr: Optional[np.ndarray] = (
                pd.to_numeric(df_raw["bytes_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                if "bytes_ratio" in df_raw.columns
                else None
            )
        else:
            tls_arr = None
            ssh_arr = None
            ratio_arr = None

        for i in range(n):
            reasons: List[str] = []
            if preds[i] == 1:
                reasons.append(f"ANOMALY_SCORE_ABOVE_CALIBRATED_FPR_{self.target_fpr_:.3f}")
                if scores[i] >= thr + 2.0 * self.score_scale_:
                    reasons.append("EXTREME_ISOLATION_DEPTH_OUTLIER")
                if df_raw is not None and i < n_raw:
                    if tls_arr is not None and tls_arr[i] == 1.0:
                        reasons.append("CONTEXT_TLS_PORT_FLOW")
                    elif ssh_arr is not None and ssh_arr[i] == 1.0:
                        reasons.append("CONTEXT_SSH_PORT_FLOW")
                    if ratio_arr is not None and ratio_arr[i] > 50.0:
                        reasons.append("CONTEXT_HIGH_BYTES_ASYMMETRY")
            else:
                reasons.append("WITHIN_BENIGN_CALIBRATED_BASELINE")
            reason_codes.append(reasons)

        return {
            "predictions": preds,
            "anomaly_scores": scores,
            "confidence": confidence,
            "reason_codes": reason_codes,
            "calibrated_threshold": thr,
            "target_fpr": self.target_fpr_,
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "n_estimators": self.n_estimators,
            "max_samples": str(self.max_samples),
            "contamination": str(self.contamination),
            "max_features": self.max_features,
            "random_state": self.random_state,
            "benign_only_fit": self.benign_only_fit,
            "n_train_benign": self.n_train_benign,
            "calibrated_threshold": self.calibrated_threshold_,
            "target_fpr": self.target_fpr_,
            "score_scale": self.score_scale_,
            "val_benign_score_mean": self.val_benign_score_mean_,
            "val_benign_score_std": self.val_benign_score_std_,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_checkpoint(cls, payload: Any) -> "IsolationForestThreatModel":
        """Reconstruct IsolationForestThreatModel from either a wrapper instance or a dict payload."""
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            if "model" in payload and isinstance(payload["model"], cls):
                return payload["model"]
            if "estimator" in payload:
                est = payload["estimator"]
                state = payload.get("model_state") or payload.get("state") or {}
                obj = cls(
                    n_estimators=int(state.get("n_estimators", getattr(est, "n_estimators", 200))),
                    max_samples=getattr(est, "max_samples", state.get("max_samples", "auto")),
                    contamination=getattr(est, "contamination", state.get("contamination", "auto")),
                    max_features=float(state.get("max_features", getattr(est, "max_features", 1.0))),
                    random_state=int(state.get("random_state", getattr(est, "random_state", 42) or 42)),
                    n_jobs=int(getattr(est, "n_jobs", state.get("n_jobs", -1)) or -1),
                )
                obj.estimator = est
                obj.benign_only_fit = bool(state.get("benign_only_fit", True))
                obj.n_train_benign = int(state.get("n_train_benign", 0))
                cal_thr = state.get("calibrated_threshold")
                obj.calibrated_threshold_ = float(cal_thr) if cal_thr is not None else None
                obj.target_fpr_ = float(state.get("target_fpr", 0.05))
                obj.score_scale_ = float(state.get("score_scale", 1.0))
                obj.val_benign_score_mean_ = float(state.get("val_benign_score_mean", 0.0))
                obj.val_benign_score_std_ = float(state.get("val_benign_score_std", 1.0))
                obj.is_fitted = bool(state.get("is_fitted", hasattr(est, "estimators_")))
                return obj
        raise TypeError(
            f"Unsupported IsolationForest checkpoint payload type: {type(payload).__name__}"
        )


class IsolationForestTrainer(BaseTrainer):
    """Trainer orchestrator for Week 6 Isolation Forest (benign fit + val calibration)."""

    def __init__(self, model: IsolationForestThreatModel, target_fpr: float = 0.05) -> None:
        self.model = model
        self.target_fpr = float(target_fpr)

    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        result: Dict[str, Any] = {
            "model_name": self.model.model_name,
            "fit_seconds": float(fit_seconds),
            "n_train_benign": self.model.n_train_benign,
            "benign_only_fit": True,
        }
        if X_val is not None and y_val is not None:
            calib = self.model.calibrate_threshold(X_val, y_val, target_fpr=self.target_fpr)
            result["calibration"] = calib
        return result


class IsolationForestEvaluator(BaseEvaluator):
    """
    Evaluator for Week 6 Isolation Forest covering both full Test split and
    paired Unseen-Attack Holdout (`unseen_attack_holdout.parquet` + benign test flows).
    """

    @staticmethod
    def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def _safe_aucpr(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float(np.mean(y_true == 1))
        return float(average_precision_score(y_true, y_score))

    def _compute_metrics(
        self,
        model: IsolationForestThreatModel,
        X: np.ndarray,
        y: np.ndarray,
        fpr_targets: Sequence[float] = (0.01, 0.05, 0.10),
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        scores = model.score_samples(X)
        preds = model.predict(X)
        elapsed = time.perf_counter() - t0

        yt = np.asarray(y, dtype=np.int64)
        cm = confusion_matrix(yt, preds, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        achieved_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        achieved_tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Detection rate (TPR) at fixed FPR operating points on the benign subset
        tpr_at_fixed_fpr: Dict[str, float] = {}
        benign_mask = yt == 0
        attack_mask = yt == 1
        if np.any(benign_mask) and np.any(attack_mask):
            b_scores = scores[benign_mask]
            a_scores = scores[attack_mask]
            for target in fpr_targets:
                q_thr = float(np.quantile(b_scores, 1.0 - target))
                tpr_at_fixed_fpr[f"tpr_at_fpr_{target:.2f}"] = float((a_scores >= q_thr).mean())
        else:
            for target in fpr_targets:
                tpr_at_fixed_fpr[f"tpr_at_fpr_{target:.2f}"] = achieved_tpr

        return {
            "calibrated_threshold": float(model.calibrated_threshold_),
            "target_fpr": float(model.target_fpr_),
            "achieved_fpr": achieved_fpr,
            "tpr_at_fpr": achieved_tpr,
            "tpr_at_fixed_fpr": tpr_at_fixed_fpr,
            "aucpr": self._safe_aucpr(yt, scores),
            "roc_auc": self._safe_roc_auc(yt, scores),
            "f1": float(f1_score(yt, preds, zero_division=0)),
            "precision": float(precision_score(yt, preds, zero_division=0)),
            "recall": float(recall_score(yt, preds, zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "n_samples": int(len(yt)),
            "inference_latency_ms_per_flow": float((elapsed / max(1, len(yt))) * 1000.0),
        }

    def evaluate(self, model: BaseThreatModel, X: Any, y: np.ndarray) -> Dict[str, Any]:
        return self._compute_metrics(model, X, y)

    def evaluate_unseen_holdout(
        self,
        model: IsolationForestThreatModel,
        X_unseen: np.ndarray,
        y_unseen: np.ndarray,
        X_benign_test: np.ndarray,
        y_benign_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly detection on unseen attacks paired with benign test flows
        so AUCPR, ROC-AUC, and TPR@FPR are mathematically well-defined.
        """
        X_b = np.asarray(X_benign_test, dtype=np.float32)
        y_b = np.asarray(y_benign_test, dtype=np.int64)
        X_u = np.asarray(X_unseen, dtype=np.float32)
        y_u = np.asarray(y_unseen, dtype=np.int64)
        if len(X_b) == 0 or len(y_b) == 0 or not np.any(y_b == 0):
            raise ValueError(
                "evaluate_unseen_holdout() requires non-empty benign (label == 0) test rows "
                "paired with unseen attacks so ROC-AUC, AUCPR, and fixed-FPR metrics are well-defined."
            )
        if len(X_u) == 0 or len(y_u) == 0:
            raise ValueError("evaluate_unseen_holdout() requires non-empty unseen attack rows.")
        X_pair = np.vstack([X_b, X_u])
        y_pair = np.concatenate([y_b, y_u])
        metrics = self._compute_metrics(model, X_pair, y_pair)
        metrics["n_benign_paired"] = int(len(y_b))
        metrics["n_unseen_attacks"] = int(len(y_u))
        return metrics
