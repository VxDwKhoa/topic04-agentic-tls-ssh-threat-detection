"""
Week 5 Deterministic Rules Baseline (L3/L4 + optional L5/L6 hooks, strictly NO L7).
Topic 04 — Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response.
Adheres to SR-01, SR-03, CR-01..CR-06, Section 10, Section 19, Section 20, and Section 23.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.base import BaseThreatModel


class RulesBaselineModel(BaseThreatModel):
    """
    Deterministic, interpretable L3/L4 network-flow and optional L5/L6 cryptographic policy
    rule engine. Emits bounded threat scores in [0.0, 1.0], binary predictions {0, 1}, and
    per-row human-readable reason codes without reading any raw Layer-7 payload.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        syn_flood_pkt_threshold: int = 100,
        high_retransmit_pkts: int = 20,
        extreme_bytes_ratio: float = 100.0,
        extreme_pkts_ratio: float = 50.0,
        high_throughput_bps: float = 1e8,
    ) -> None:
        super().__init__(model_name="week5_rules_baseline", task_type="binary")
        self.threshold = float(threshold)
        self.syn_flood_pkt_threshold = int(syn_flood_pkt_threshold)
        self.high_retransmit_pkts = int(high_retransmit_pkts)
        self.extreme_bytes_ratio = float(extreme_bytes_ratio)
        self.extreme_pkts_ratio = float(extreme_pkts_ratio)
        self.high_throughput_bps = float(high_throughput_bps)
        self.is_fitted = True  # Deterministic policy engine requires no stochastic fitting
        self.last_reason_codes_: Optional[List[List[str]]] = None

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "RulesBaselineModel":
        """No-op fit for interface compatibility with BaseThreatModel."""
        self.is_fitted = True
        return self

    @staticmethod
    def _col_array(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
            return vals.to_numpy(dtype=np.float64)
        return np.full(len(df), default, dtype=np.float64)

    def _evaluate_rules(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"RulesBaselineModel expects a pandas DataFrame with metadata columns, got {type(df).__name__}."
            )

        n = len(df)
        if n == 0:
            return np.zeros((0,), dtype=np.float64), []

        scores = np.zeros(n, dtype=np.float64)
        reasons: List[List[str]] = [[] for _ in range(n)]

        in_pkts = self._col_array(df, "in_pkts")
        out_pkts = self._col_array(df, "out_pkts")
        in_bytes = self._col_array(df, "in_bytes")
        out_bytes = self._col_array(df, "out_bytes")
        duration_ms = self._col_array(df, "flow_duration_ms")
        tcp_flags = self._col_array(df, "tcp_flags").astype(np.int64)
        client_flags = self._col_array(df, "client_tcp_flags").astype(np.int64)
        retrans_in = self._col_array(df, "retransmitted_in_pkts")
        retrans_out = self._col_array(df, "retransmitted_out_pkts")
        bytes_ratio = self._col_array(df, "bytes_ratio", default=1.0)
        pkts_ratio = self._col_array(df, "pkts_ratio", default=1.0)
        dst_port = self._col_array(df, "dst_port").astype(np.int64)
        is_ssh = self._col_array(df, "is_ssh_port")
        src_tp = self._col_array(df, "src_to_dst_avg_throughput")

        # Rule 1: SYN-only / Unanswered scan or SYN flood burst (L3/L4)
        r1 = ((tcp_flags == 2) | (client_flags == 2)) & ((out_pkts == 0) | (in_pkts >= self.syn_flood_pkt_threshold))
        scores += np.where(r1, 0.45, 0.0)
        for idx in np.flatnonzero(r1):
            reasons[idx].append("RULE_L4_SYN_SCAN_OR_FLOOD")

        # Rule 2: SSH brute-force / short high-churn connection storm (L4 + SSH port)
        r2 = ((is_ssh == 1) | (dst_port == 22) | (dst_port == 2222)) & (
            (in_pkts >= 50) | (retrans_in >= self.high_retransmit_pkts) | ((duration_ms <= 50) & (in_pkts >= 10))
        )
        scores += np.where(r2, 0.50, 0.0)
        for idx in np.flatnonzero(r2):
            reasons[idx].append("RULE_SSH_BRUTEFORCE_CHURN")

        # Rule 3: Severe TCP retransmission burst (DoS / packet drop anomaly)
        r3 = (retrans_in + retrans_out) >= self.high_retransmit_pkts
        scores += np.where(r3, 0.35, 0.0)
        for idx in np.flatnonzero(r3):
            reasons[idx].append("RULE_L4_HIGH_RETRANSMISSION")

        # Rule 4: Extreme flow asymmetry (exfiltration / amplification / one-way flood)
        r4 = (bytes_ratio >= self.extreme_bytes_ratio) | (pkts_ratio >= self.extreme_pkts_ratio) | (
            (in_bytes >= 50000) & (out_bytes <= 100)
        )
        scores += np.where(r4, 0.40, 0.0)
        for idx in np.flatnonzero(r4):
            reasons[idx].append("RULE_L3_EXTREME_FLOW_ASYMMETRY")

        # Rule 5: High-throughput burst anomaly
        r5 = (src_tp >= self.high_throughput_bps) & (duration_ms <= 5000)
        scores += np.where(r5, 0.30, 0.0)
        for idx in np.flatnonzero(r5):
            reasons[idx].append("RULE_L4_HIGH_THROUGHPUT_BURST")

        # Optional L5/L6 Cryptographic Policy Hooks (for Zeek Lab / TLS-SSH datasets, CR-01..CR-04)
        # Note: CR-03 strictly treats ECH (`ech_observed`) as a benign privacy feature, NEVER a rule violation.
        if "tls_version" in df.columns:
            tls_ver = df["tls_version"].astype(str).str.upper()
            r_tls = tls_ver.isin({"SSLV2", "SSLV3", "TLSV10", "TLSV11", "TLS1.0", "TLS1.1"}).to_numpy()
            scores += np.where(r_tls, 0.60, 0.0)
            for idx in np.flatnonzero(r_tls):
                reasons[idx].append("RULE_CR01_DEPRECATED_TLS_VERSION")

        if "cert_status" in df.columns:
            cert_st = df["cert_status"].astype(str).str.lower()
            r_cert = cert_st.isin({"expired", "self_signed", "revoked", "untrusted"}).to_numpy()
            scores += np.where(r_cert, 0.55, 0.0)
            for idx in np.flatnonzero(r_cert):
                reasons[idx].append("RULE_CR02_CERT_TRUST_ANOMALY")

        if "ssh_kex_class" in df.columns:
            ssh_kex = df["ssh_kex_class"].astype(str).str.lower()
            r_kex = ssh_kex.str.contains("legacy|sha1|group1|weak", na=False).to_numpy()
            scores += np.where(r_kex, 0.55, 0.0)
            for idx in np.flatnonzero(r_kex):
                reasons[idx].append("RULE_CR04_WEAK_SSH_KEX")

        scores = np.clip(scores, 0.0, 1.0)
        self.last_reason_codes_ = reasons
        return scores, reasons

    def score_samples(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Return bounded deterministic threat scores in [0.0, 1.0]."""
        scores, _ = self._evaluate_rules(X)
        return scores

    def predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Return 2-column probability-like scores [1 - score, score] in [0.0, 1.0]."""
        scores, _ = self._evaluate_rules(X)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Return deterministic binary predictions {0, 1}."""
        scores, _ = self._evaluate_rules(X)
        return (scores >= self.threshold).astype(np.int64)

    def explain(self, X: pd.DataFrame) -> List[List[str]]:
        """Return per-row rule reason codes."""
        _, reasons = self._evaluate_rules(X)
        return reasons

    def predict_with_reasons(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Return predictions, bounded scores, and per-row reason codes."""
        scores, reasons = self._evaluate_rules(X)
        preds = (scores >= self.threshold).astype(np.int64)
        return {
            "predictions": preds,
            "scores": scores,
            "reason_codes": reasons,
            "threshold": self.threshold,
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "deterministic": True,
            "threshold": self.threshold,
            "syn_flood_pkt_threshold": self.syn_flood_pkt_threshold,
            "high_retransmit_pkts": self.high_retransmit_pkts,
            "extreme_bytes_ratio": self.extreme_bytes_ratio,
            "extreme_pkts_ratio": self.extreme_pkts_ratio,
            "high_throughput_bps": self.high_throughput_bps,
            "l7_payload_used": False,
            "ech_treated_as_privacy_feature": True,
        }
