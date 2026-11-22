"""
Week 8 Calibrated Multi-Model Fusion Engine (Decomposable Evidence Scoring).
Topic 04 — Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response.

Adheres strictly to Section 10 (Phase 7 — Suggested Fusion Formulation) & SR-04:
    s_total = w_r * s_rule + w_c * s_classifier + w_a * s_anomaly + w_s * s_sequence
where:
- Weights (w_r, w_c, w_a, w_s >= 0, sum == 1.0) and decision threshold are calibrated strictly
  on the Validation split (NEVER invented at runtime by the AI Agent).
- Final score `s_total` remains 100% decomposable into per-component evidence scores,
  weighted contributions, and reason codes for downstream Week 9 Threat-Hunting Agent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.transforms import FlowFeatureTransformer
from src.models.base import BaseEvaluator, BaseThreatModel, CheckpointManager
from src.models.isolation_forest import IsolationForestThreatModel
from src.models.lstm_sequence import LSTMSequenceModel
from src.models.random_forest import RandomForestThreatModel
from src.models.rules_baseline import RulesBaselineModel


COMPONENT_NAMES: Tuple[str, ...] = ("rule", "classifier", "anomaly", "sequence")


def discover_checkpoint_file(root_dir: Union[str, Path], filename: str) -> Path:
    """Recursively locate a checkpoint artifact (supports flat Checkpoint/ or week5/results/Checkpoint/)."""
    root = Path(root_dir)
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find checkpoint file '{filename}' under {root}")
    return matches[0]


def align_lstm_sequence_scores_to_flows(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    lstm_model: LSTMSequenceModel,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Maps Pure PyTorch LSTM sequence probabilities back to individual flows in `df`
    without leaking across hosts:
    - Groups flows strictly by `src_host_token` ordered by `flow_order_id`.
    - For hosts with >= `seq_len` flows, slides `(seq_len, stride)` windows, scores them with
      `lstm_model`, and aggregates overlapping window probabilities onto each flow via `max`.
    - Flows belonging to hosts shorter than `seq_len` receive a neutral baseline score `0.0`
      (flagged via availability mask if needed; never fabricated from other hosts).
    """
    n = len(df)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    seq_len = int(lstm_model.seq_len)
    stride = int(lstm_model.stride)

    work_df = pd.DataFrame({
        "_orig_idx": np.arange(n, dtype=np.int64),
        "src_host_token": df["src_host_token"].to_numpy(),
        "flow_order_id": df["flow_order_id"].to_numpy(),
    })
    work_df.sort_values(["src_host_token", "flow_order_id"], kind="mergesort", inplace=True)
    work_df.reset_index(drop=True, inplace=True)
    sorted_orig_idx = work_df["_orig_idx"].to_numpy()

    windows: List[np.ndarray] = []
    window_flow_indices: List[np.ndarray] = []

    for _, pos_arr in work_df.groupby("src_host_token", sort=False).indices.items():
        if len(pos_arr) < seq_len:
            continue
        orig_positions = sorted_orig_idx[pos_arr]
        host_x = X_scaled[orig_positions]
        for start in range(0, len(orig_positions) - seq_len + 1, stride):
            windows.append(host_x[start:start + seq_len])
            window_flow_indices.append(orig_positions[start:start + seq_len])

    flow_scores = np.zeros(n, dtype=np.float64)
    if not windows:
        return flow_scores

    X_seq = np.stack(windows, axis=0).astype(np.float32)
    seq_probs = lstm_model.predict_proba(X_seq, batch_size=batch_size)

    for w_idx, prob in enumerate(seq_probs):
        idxs = window_flow_indices[w_idx]
        p = float(prob)
        flow_scores[idxs] = np.maximum(flow_scores[idxs], p)

    return np.clip(flow_scores, 0.0, 1.0)


class CalibratedFusionModel(BaseThreatModel):
    """
    Decomposable multi-model fusion detector implementing Section 10:
        s_total = w_r * s_rule + w_c * s_classifier + w_a * s_anomaly + w_s * s_sequence
    with convex weights (w_i >= 0, sum(w) == 1.0) and decision threshold calibrated strictly
    on the Validation split.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        min_anomaly_weight: float = 0.05,
        min_rule_weight: float = 0.02,
    ) -> None:
        super().__init__(model_name="week8_calibrated_fusion", task_type="binary")
        self.min_anomaly_weight = float(min_anomaly_weight)
        self.min_rule_weight = float(min_rule_weight)
        default_w = {"rule": 0.05, "classifier": 0.45, "anomaly": 0.10, "sequence": 0.40}
        self.weights: Dict[str, float] = self._normalize_weights(weights or default_w)
        self.threshold: float = float(threshold)
        self.calibration_summary_: Dict[str, Any] = {}
        self.is_fitted = weights is not None

    @staticmethod
    def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
        vec = np.array([max(0.0, float(w.get(k, 0.0))) for k in COMPONENT_NAMES], dtype=np.float64)
        total = float(vec.sum())
        if total <= 1e-12:
            vec = np.full(len(COMPONENT_NAMES), 1.0 / len(COMPONENT_NAMES), dtype=np.float64)
        else:
            vec = vec / total
        return {k: float(vec[i]) for i, k in enumerate(COMPONENT_NAMES)}

    @staticmethod
    def _scores_matrix(component_scores: Union[Dict[str, np.ndarray], np.ndarray]) -> np.ndarray:
        if isinstance(component_scores, dict):
            cols = [np.clip(np.asarray(component_scores[k], dtype=np.float64), 0.0, 1.0) for k in COMPONENT_NAMES]
            return np.column_stack(cols)
        arr = np.asarray(component_scores, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(COMPONENT_NAMES):
            raise ValueError(f"Expected (N, {len(COMPONENT_NAMES)}) component score matrix, got {arr.shape}.")
        return np.clip(arr, 0.0, 1.0)

    def fit(
        self,
        X: Union[Dict[str, np.ndarray], np.ndarray],
        y: Optional[np.ndarray] = None,
        target_fpr: Optional[float] = None,
        **kwargs: Any,
    ) -> "CalibratedFusionModel":
        """
        Calibrate convex fusion weights (w_r, w_c, w_a, w_s) and decision threshold strictly
        on the Validation split by minimizing regularized binary log-loss under simplex constraints
        with configurable floor weights on rule & anomaly channels (preserving zero-day sensitivity).
        """
        if y is None:
            raise ValueError("CalibratedFusionModel.fit() requires validation labels `y`.")

        S = self._scores_matrix(X)
        yt = np.asarray(y, dtype=np.int64)
        if len(S) != len(yt) or len(yt) == 0:
            raise ValueError(f"Invalid calibration inputs: S={len(S)}, y={len(yt)}")

        eps = 1e-6
        bounds = [
            (self.min_rule_weight, 1.0),
            (0.0, 1.0),
            (self.min_anomaly_weight, 1.0),
            (0.0, 1.0),
        ]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        init_w = np.array([
            max(self.min_rule_weight, 0.05),
            0.45,
            max(self.min_anomaly_weight, 0.10),
            max(0.0, 1.0 - max(self.min_rule_weight, 0.05) - 0.45 - max(self.min_anomaly_weight, 0.10)),
        ], dtype=np.float64)
        init_w = init_w / init_w.sum()

        def objective(w: np.ndarray) -> float:
            p = np.clip(S @ w, eps, 1.0 - eps)
            # Small L2 regularization toward balanced diversity prevents degenerate collapse
            return float(log_loss(yt, p, labels=[0, 1]) + 1e-3 * np.sum(w ** 2))

        opt = minimize(objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        best_w = opt.x if opt.success else init_w
        self.weights = self._normalize_weights({k: best_w[i] for i, k in enumerate(COMPONENT_NAMES)})

        fused_val = self.score_samples(S)
        if target_fpr is not None and np.any(yt == 0):
            self.threshold = float(np.quantile(fused_val[yt == 0], 1.0 - float(target_fpr)))
        else:
            # Sweep thresholds on Validation to maximize F1
            cand_thresholds = np.linspace(0.15, 0.85, 71)
            best_f1, best_thr = -1.0, 0.5
            for thr in cand_thresholds:
                score_f1 = float(f1_score(yt, (fused_val >= thr).astype(np.int64), zero_division=0))
                if score_f1 > best_f1:
                    best_f1, best_thr = score_f1, float(thr)
            self.threshold = best_thr

        preds_val = (fused_val >= self.threshold).astype(np.int64)
        self.calibration_summary_ = {
            "weights": dict(self.weights),
            "calibrated_threshold": self.threshold,
            "val_f1": float(f1_score(yt, preds_val, zero_division=0)),
            "val_fpr": float(preds_val[yt == 0].mean()) if np.any(yt == 0) else 0.0,
            "val_roc_auc": float(roc_auc_score(yt, fused_val)) if len(np.unique(yt)) > 1 else 0.5,
            "val_pr_auc": float(average_precision_score(yt, fused_val)) if len(np.unique(yt)) > 1 else float(yt.mean()),
            "optimizer_success": bool(opt.success),
            "n_val_samples": int(len(yt)),
        }
        self.is_fitted = True
        return self

    def score_samples(self, X: Union[Dict[str, np.ndarray], np.ndarray], **kwargs: Any) -> np.ndarray:
        S = self._scores_matrix(X)
        w = np.array([self.weights[k] for k in COMPONENT_NAMES], dtype=np.float64)
        return np.clip(S @ w, 0.0, 1.0)

    def predict_proba(self, X: Union[Dict[str, np.ndarray], np.ndarray], **kwargs: Any) -> np.ndarray:
        s = self.score_samples(X)
        return np.column_stack([1.0 - s, s])

    def predict(self, X: Union[Dict[str, np.ndarray], np.ndarray], threshold: Optional[float] = None, **kwargs: Any) -> np.ndarray:
        thr = float(threshold) if threshold is not None else self.threshold
        return (self.score_samples(X) >= thr).astype(np.int64)

    def decompose_evidence(
        self,
        component_scores: Dict[str, np.ndarray],
        rule_reasons: Optional[List[List[str]]] = None,
        anomaly_reasons: Optional[List[List[str]]] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decompose `s_total` into auditable per-flow evidence payloads for the Week 9
        Threat-Hunting Agent & Evidence Graph.
        """
        S = self._scores_matrix(component_scores)
        w_vec = np.array([self.weights[k] for k in COMPONENT_NAMES], dtype=np.float64)
        totals = np.clip(S @ w_vec, 0.0, 1.0)
        preds = (totals >= self.threshold).astype(np.int64)

        limit = len(totals) if top_n is None else min(len(totals), int(top_n))
        records: List[Dict[str, Any]] = []
        for i in range(limit):
            contribs = {k: float(S[i, j] * w_vec[j]) for j, k in enumerate(COMPONENT_NAMES)}
            raw_scores = {k: float(S[i, j]) for j, k in enumerate(COMPONENT_NAMES)}
            dominant = max(contribs.items(), key=lambda kv: kv[1])[0]
            reasons: List[str] = []
            if rule_reasons is not None and i < len(rule_reasons):
                reasons.extend(rule_reasons[i])
            if anomaly_reasons is not None and i < len(anomaly_reasons):
                reasons.extend(anomaly_reasons[i])
            records.append({
                "s_total": float(totals[i]),
                "prediction": int(preds[i]),
                "threshold": self.threshold,
                "dominant_signal": dominant,
                "component_scores": raw_scores,
                "weighted_contributions": contribs,
                "reason_codes": reasons,
            })
        return records

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "weights": dict(self.weights),
            "threshold": self.threshold,
            "min_anomaly_weight": self.min_anomaly_weight,
            "min_rule_weight": self.min_rule_weight,
            "calibration_summary": self.calibration_summary_,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_checkpoint(cls, payload: Dict[str, Any]) -> "CalibratedFusionModel":
        state = payload.get("model_state", payload)
        obj = cls(
            weights=state["weights"],
            threshold=float(state["threshold"]),
            min_anomaly_weight=float(state.get("min_anomaly_weight", 0.05)),
            min_rule_weight=float(state.get("min_rule_weight", 0.02)),
        )
        obj.calibration_summary_ = state.get("calibration_summary", {})
        obj.is_fitted = True
        return obj


class MultiModelInferenceBundle:
    """
    Loads the trained Week 5 (Rules + RF Binary), Week 6 (Isolation Forest), and Week 7
    (Pure PyTorch LSTM) checkpoints from disk and produces aligned [0, 1] component scores
    for CalibratedFusionModel.
    """

    def __init__(
        self,
        rules_model: RulesBaselineModel,
        rf_model: RandomForestThreatModel,
        if_model: IsolationForestThreatModel,
        lstm_model: LSTMSequenceModel,
        transformer: FlowFeatureTransformer,
    ) -> None:
        self.rules_model = rules_model
        self.rf_model = rf_model
        self.if_model = if_model
        self.lstm_model = lstm_model
        self.transformer = transformer

    @classmethod
    def from_checkpoint_root(
        cls,
        checkpoint_root: Union[str, Path],
        device: Optional[str] = None,
    ) -> "MultiModelInferenceBundle":
        root = Path(checkpoint_root)

        # 1. Week 5 Random Forest Binary
        rf_path = discover_checkpoint_file(root, "week5_rf_binary.joblib")
        rf_payload = joblib.load(rf_path)
        rf_model = RandomForestThreatModel.from_checkpoint(rf_payload)
        transformer = FlowFeatureTransformer.from_state(rf_payload["transformer_state"])

        # 2. Week 6 Isolation Forest
        if_path = discover_checkpoint_file(root, "week6_isolation_forest.joblib")
        if_payload = joblib.load(if_path)
        if_model = IsolationForestThreatModel.from_checkpoint(if_payload)

        # 3. Week 7 Pure PyTorch LSTM
        lstm_pt_path = discover_checkpoint_file(root, "week7_lstm_temporal.pt")
        lstm_meta_path = discover_checkpoint_file(root, "week7_lstm_temporal_metadata.json")
        with open(lstm_meta_path, "r", encoding="utf-8") as f:
            lstm_meta = json.load(f)

        resolved_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        lstm_model = LSTMSequenceModel(
            input_dim=len(lstm_meta["feature_columns"]),
            hidden_dim=int(lstm_meta["hidden_dim"]),
            num_layers=int(lstm_meta["num_layers"]),
            seq_len=int(lstm_meta["seq_len"]),
            stride=int(lstm_meta["stride"]),
            device=resolved_device,
        )
        try:
            pt_obj = torch.load(lstm_pt_path, map_location=resolved_device, weights_only=True)
        except TypeError:
            pt_obj = torch.load(lstm_pt_path, map_location=resolved_device)
        state_dict = pt_obj["state_dict"] if isinstance(pt_obj, dict) and "state_dict" in pt_obj else pt_obj
        lstm_model.load_state_dict(state_dict)

        return cls(
            rules_model=RulesBaselineModel(threshold=0.45),
            rf_model=rf_model,
            if_model=if_model,
            lstm_model=lstm_model,
            transformer=transformer,
        )

    def compute_component_scores(
        self,
        df: pd.DataFrame,
        lstm_batch_size: int = 512,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Compute all 4 normalized [0, 1] threat scores for each flow in `df`:
        - `rule`: RulesBaselineModel deterministic score in [0, 1].
        - `classifier`: RandomForest binary attack probability in [0, 1].
        - `anomaly`: IsolationForest calibrated confidence in [0, 1] (SR-04).
        - `sequence`: Pure PyTorch LSTM host-temporal sequence probability aligned to flows in [0, 1].
        """
        X_scaled = self.transformer.transform(df)

        rules_out = self.rules_model.predict_with_reasons(df)
        s_rule = np.asarray(rules_out["scores"], dtype=np.float64)

        rf_probs = self.rf_model.predict_proba(X_scaled)
        s_classifier = np.asarray(rf_probs[:, 1] if rf_probs.ndim == 2 else rf_probs, dtype=np.float64)

        if_ctx = self.if_model.predict_with_context(X_scaled, df_raw=df)
        s_anomaly = np.asarray(if_ctx["confidence"], dtype=np.float64)

        s_sequence = align_lstm_sequence_scores_to_flows(df, X_scaled, self.lstm_model, batch_size=lstm_batch_size)

        scores = {
            "rule": np.clip(s_rule, 0.0, 1.0),
            "classifier": np.clip(s_classifier, 0.0, 1.0),
            "anomaly": np.clip(s_anomaly, 0.0, 1.0),
            "sequence": np.clip(s_sequence, 0.0, 1.0),
        }
        context = {
            "rule_reasons": rules_out["reason_codes"],
            "anomaly_reasons": if_ctx["reason_codes"],
        }
        return scores, context


class FusionEvaluator(BaseEvaluator):
    """Evaluates CalibratedFusionModel against every individual constituent model."""

    @staticmethod
    def _metrics_for_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
        yt = np.asarray(y_true, dtype=np.int64)
        ys = np.asarray(y_score, dtype=np.float64)
        yp = (ys >= threshold).astype(np.int64)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        return {
            "roc_auc": float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else 0.5,
            "pr_auc": float(average_precision_score(yt, ys)) if len(np.unique(yt)) > 1 else float(yt.mean()),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "confusion_matrix": cm.tolist(),
        }

    def evaluate(
        self,
        model: CalibratedFusionModel,
        component_scores: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        fused_scores = model.score_samples(component_scores)
        elapsed = time.perf_counter() - t0

        res = self._metrics_for_scores(y_true, fused_scores, model.threshold)
        res["weights"] = dict(model.weights)
        res["threshold"] = model.threshold
        res["fusion_overhead_ms_per_flow"] = float((elapsed / max(1, len(y_true))) * 1000.0)
        return res

    def compare_all(
        self,
        model: CalibratedFusionModel,
        component_scores: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        default_thresholds = {"rule": 0.45, "classifier": 0.50, "anomaly": 0.50, "sequence": 0.50}
        for name in COMPONENT_NAMES:
            m = self._metrics_for_scores(y_true, component_scores[name], default_thresholds[name])
            rows.append({"Model": f"Single_{name}", **{k: m[k] for k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "fpr")}})
        fused_m = self.evaluate(model, component_scores, y_true)
        rows.append({"Model": "Calibrated_Fusion", **{k: fused_m[k] for k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "fpr")}})
        return pd.DataFrame(rows)
