"""
Week 5 Object-Oriented Random Forest Baseline (Binary + Multi-Class) & Evaluator.
Topic 04 — Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response.
Adheres to Section 10, Section 17 (E2), Section 18, Section 19, Section 20, and Section 23.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.base import BaseEvaluator, BaseThreatModel, BaseTrainer


class RandomForestThreatModel(BaseThreatModel):
    """
    OOP Random Forest classifier supporting both:
    - `task="binary"` (`label`: 0 Benign vs 1 Attack)
    - `task="multiclass"` (`attack_family`: known Train attack families)
    """

    def __init__(
        self,
        task: str = "binary",
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced_subsample",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if task not in ("binary", "multiclass"):
            raise ValueError(f"Unsupported task '{task}'. Expected 'binary' or 'multiclass'.")
        super().__init__(model_name=f"week5_rf_{task}", task_type=task)
        self.task = task
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)

        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        if hasattr(X, "X"):
            arr = np.asarray(X.X, dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Input features contain NaN or Inf. Run FlowFeatureTransformer first.")
        return arr

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "RandomForestThreatModel":
        if y is None:
            if hasattr(X, "y") and X.y is not None:
                y_arr = np.asarray(X.y)
            else:
                raise ValueError("RandomForestThreatModel.fit() requires target labels `y`.")
        else:
            y_arr = np.asarray(y)

        X_arr = self._to_numpy(X)
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(X_arr)} rows, y has {len(y_arr)} rows.")

        # Exclude any unseen-encoded (-1) labels if passed during supervised training
        valid_mask = y_arr >= 0
        if not np.all(valid_mask):
            X_arr = X_arr[valid_mask]
            y_arr = y_arr[valid_mask]

        self.estimator.fit(X_arr, y_arr)
        self.n_features_in_ = int(X_arr.shape[1])
        self.classes_ = np.asarray(self.estimator.classes_)
        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RandomForestThreatModel must be fitted before calling predict().")
        return np.asarray(self.estimator.predict(self._to_numpy(X)))

    def predict_proba(self, X: Any, **kwargs: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RandomForestThreatModel must be fitted before calling predict_proba().")
        return np.asarray(self.estimator.predict_proba(self._to_numpy(X)), dtype=np.float64)

    def get_feature_importance(self, feature_names: Sequence[str]) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before querying feature importance.")
        importances = np.asarray(self.estimator.feature_importances_, dtype=np.float64)
        if len(feature_names) != len(importances):
            raise ValueError(
                f"Expected {len(importances)} feature names, got {len(feature_names)}."
            )
        total = float(importances.sum())
        if total > 0:
            importances = importances / total
        else:
            importances = np.full(len(importances), 1.0 / len(importances), dtype=np.float64)
        pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return {str(k): float(v) for k, v in pairs}

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "task_type": self.task_type,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": str(self.class_weight),
            "random_state": self.random_state,
            "n_features_in": self.n_features_in_,
            "num_classes": int(len(self.classes_)) if self.classes_ is not None else 0,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_checkpoint(cls, payload: Any) -> "RandomForestThreatModel":
        """Reconstruct RandomForestThreatModel from either a wrapper instance or a dict payload."""
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            if "model" in payload and isinstance(payload["model"], cls):
                return payload["model"]
            if "estimator" in payload:
                est = payload["estimator"]
                state = payload.get("model_state") or payload.get("state") or {}
                task = str(state.get("task", state.get("task_type", "binary")))
                if task not in ("binary", "multiclass"):
                    n_cls = len(getattr(est, "classes_", []))
                    task = "multiclass" if n_cls > 2 else "binary"
                obj = cls(
                    task=task,
                    n_estimators=int(state.get("n_estimators", getattr(est, "n_estimators", 100))),
                    max_depth=state.get("max_depth", getattr(est, "max_depth", 20)),
                    min_samples_split=int(state.get("min_samples_split", getattr(est, "min_samples_split", 2))),
                    min_samples_leaf=int(state.get("min_samples_leaf", getattr(est, "min_samples_leaf", 1))),
                    class_weight=getattr(est, "class_weight", "balanced_subsample"),
                    random_state=int(state.get("random_state", getattr(est, "random_state", 42) or 42)),
                    n_jobs=int(getattr(est, "n_jobs", state.get("n_jobs", -1)) or -1),
                )
                obj.estimator = est
                n_feat = state.get("n_features_in")
                obj.n_features_in_ = int(n_feat) if n_feat is not None else getattr(est, "n_features_in_", None)
                obj.classes_ = np.asarray(est.classes_) if hasattr(est, "classes_") else None
                obj.is_fitted = bool(state.get("is_fitted", hasattr(est, "classes_")))
                return obj
        raise TypeError(
            f"Unsupported RandomForest checkpoint payload type: {type(payload).__name__}"
        )


class RandomForestTrainer(BaseTrainer):
    """Trainer orchestrator for Week 5 Random Forest models (binary & multi-class)."""

    def __init__(self, model: RandomForestThreatModel) -> None:
        self.model = model

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
            "task": self.model.task,
            "fit_seconds": float(fit_seconds),
            "n_train_samples": int(len(y_train)),
        }
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            y_val_arr = np.asarray(y_val)
            valid = y_val_arr >= 0
            if np.any(valid):
                avg = "binary" if self.model.task == "binary" else "macro"
                result["val_f1"] = float(f1_score(y_val_arr[valid], val_preds[valid], average=avg, zero_division=0))
        return result


class RandomForestEvaluator(BaseEvaluator):
    """
    Evaluator for Week 5 Rules Baseline and Random Forest (binary & multi-class).
    Safely handles single-class edge slices and unseen (-1) test labels.
    """

    @staticmethod
    def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        unique = np.unique(y_true)
        if len(unique) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        unique = np.unique(y_true)
        if len(unique) < 2:
            return float(np.mean(y_true == 1))
        return float(average_precision_score(y_true, y_score))

    def evaluate_binary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        yt = np.asarray(y_true, dtype=np.int64)
        yp = np.asarray(y_pred, dtype=np.int64)
        ys = np.asarray(y_score, dtype=np.float64) if y_score is not None else yp.astype(np.float64)
        if ys.ndim == 2 and ys.shape[1] >= 2:
            ys = ys[:, 1]

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "roc_auc": self._safe_roc_auc(yt, ys),
            "pr_auc": self._safe_pr_auc(yt, ys),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "fpr": fpr,
            "confusion_matrix": cm.tolist(),
            "n_samples": int(len(yt)),
        }

    def evaluate_multiclass(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        yt = np.asarray(y_true, dtype=np.int64)
        yp = np.asarray(y_pred, dtype=np.int64)

        # Evaluate multi-class metrics on known classes (y_true >= 0), while recording unseen count
        unseen_mask = yt < 0
        n_unseen = int(np.sum(unseen_mask))
        known_mask = ~unseen_mask

        yt_k = yt[known_mask]
        yp_k = yp[known_mask]

        if len(yt_k) == 0:
            return {
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "per_class_recall": {},
                "confusion_matrix": [],
                "n_known_samples": 0,
                "n_unseen_samples": n_unseen,
            }

        present_labels = sorted(set(yt_k.tolist()) | set(yp_k.tolist()))
        recalls = recall_score(yt_k, yp_k, labels=present_labels, average=None, zero_division=0)

        per_class_recall: Dict[str, float] = {}
        for idx, lbl in enumerate(present_labels):
            name = (
                str(class_names[lbl])
                if class_names is not None and 0 <= lbl < len(class_names)
                else f"class_{lbl}"
            )
            per_class_recall[name] = float(recalls[idx])

        cm = confusion_matrix(yt_k, yp_k, labels=present_labels)
        return {
            "macro_f1": float(f1_score(yt_k, yp_k, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(yt_k, yp_k, average="micro", zero_division=0)),
            "per_class_recall": per_class_recall,
            "confusion_matrix": cm.tolist(),
            "present_labels": present_labels,
            "n_known_samples": int(len(yt_k)),
            "n_unseen_samples": n_unseen,
        }

    def compare_baselines(
        self,
        rules_metrics: Dict[str, Any],
        rf_metrics: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build a side-by-side comparison DataFrame of Rules baseline vs Random Forest."""
        keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "fpr"]
        rows = [
            {"model": "RulesBaseline", **{k: rules_metrics.get(k) for k in keys}},
            {"model": "RandomForest_Binary", **{k: rf_metrics.get(k) for k in keys}},
        ]
        return pd.DataFrame(rows)

    def evaluate(
        self,
        model: BaseThreatModel,
        X: Any,
        y: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        preds = model.predict(X)
        elapsed = time.perf_counter() - t0
        latency_ms = (elapsed / max(1, len(preds))) * 1000.0

        if getattr(model, "task_type", "binary") == "multiclass":
            res = self.evaluate_multiclass(y, preds, class_names=class_names)
        else:
            probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            res = self.evaluate_binary(y, preds, y_score=probs)

        res["inference_latency_ms_per_flow"] = float(latency_ms)
        return res
