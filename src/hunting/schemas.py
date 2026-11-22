"""
Typed data contracts for Week 9 Threat-Hunting Correlation, Incident Evidence Graphs,
and Evidence-Grounded MITRE ATT&CK Mappings.

Topic 04 — Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response.
Adheres strictly to SR-01, SR-02, SR-05, CR-01..CR-06, and Section 15.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Mapping, Optional, Sequence, Tuple

BENIGN_BASELINE_REASON: str = "WITHIN_BENIGN_CALIBRATED_BASELINE"

GROUND_TRUTH_COLUMNS: FrozenSet[str] = frozenset(
    {
        "label",
        "attack_family",
        "attack_label",
        "attack_layers",
        "attack_mapping",
        "ground_truth_source",
        "scenario_id",
    }
)

L5_L6_OBSERVABLE_FIELDS: Tuple[str, ...] = (
    "security_proto",
    "tls_version",
    "tls_cipher_class",
    "cert_status",
    "cert_spki_token",
    "cert_issuer_class",
    "cert_validity_days",
    "ja4",
    "ja3_optional",
    "hassh",
    "hassh_or_ja4ssh",
    "ech_observed",
    "ssh_kex_class",
    "ssh_cipher_class",
    "ssh_mac_class",
    "baseline_fp_prevalence",
    "new_edge_score",
    "session_rate_5m",
    "fanout_5m",
)

DEPRECATED_TLS_VERSIONS: FrozenSet[str] = frozenset(
    {"SSLV2", "SSLV3", "TLSV10", "TLSV11", "TLS1.0", "TLS1.1", "TLSV1.0", "TLSV1.1"}
)
UNTRUSTED_CERT_STATUSES: FrozenSet[str] = frozenset(
    {"expired", "self_signed", "revoked", "untrusted"}
)
WEAK_SSH_KEX_MARKERS: Tuple[str, ...] = ("legacy", "sha1", "group1", "weak")

NodeType = Literal[
    "incident",
    "host",
    "session",
    "fingerprint",
    "certificate",
    "analytic_signal",
    "attack_technique",
]


def sanitize_json_value(value: Any) -> Any:
    """Recursively convert values into JSON-safe Python primitives (no NaN/Inf)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if hasattr(value, "item") and callable(value.item):
        return sanitize_json_value(value.item())
    if isinstance(value, Mapping):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json_value(v) for v in value]
    return str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan", "nat"}:
        return True
    return False


def _opt_str(row: Mapping[str, Any], key: str) -> Optional[str]:
    val = row.get(key)
    if _is_missing(val):
        return None
    return str(val).strip()


def _opt_float(row: Mapping[str, Any], key: str) -> Optional[float]:
    val = row.get(key)
    if _is_missing(val):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _opt_bool(row: Mapping[str, Any], key: str) -> Optional[bool]:
    val = row.get(key)
    if _is_missing(val):
        return None
    if isinstance(val, bool):
        return val
    if hasattr(val, "item") and callable(val.item):
        val = val.item()
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(int(val))
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _req_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    f = _opt_float(row, key)
    return int(round(f)) if f is not None else int(default)


def _req_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    f = _opt_float(row, key)
    return float(f) if f is not None else float(default)


@dataclass(frozen=True)
class FlowEvidence:
    """
    Privacy-preserving, GT-free flow evidence unit combining L3/L4 observables,
    optional L5/L6 Zeek metadata, and Week 8 decomposed fusion evidence.
    """

    event_id: str
    session_id: str
    flow_order_id: int
    window_id: int
    src_host_token: str
    dst_host_token: str

    # L3/L4 observables
    dst_port: int = 0
    proto: int = 6
    in_bytes: int = 0
    out_bytes: int = 0
    in_pkts: int = 0
    out_pkts: int = 0
    flow_duration_ms: int = 0
    tcp_flags: int = 0
    retransmitted_pkts: int = 0
    bytes_ratio: float = 1.0
    pkts_ratio: float = 1.0
    src_to_dst_avg_throughput: float = 0.0
    is_tls_port: bool = False
    is_ssh_port: bool = False

    # Optional L5/L6 Zeek metadata (None when absent, e.g. NF-UQ-NIDS-v2)
    security_proto: Optional[str] = None
    tls_version: Optional[str] = None
    tls_cipher_class: Optional[str] = None
    cert_status: Optional[str] = None
    cert_spki_token: Optional[str] = None
    cert_issuer_class: Optional[str] = None
    cert_validity_days: Optional[float] = None
    ja4: Optional[str] = None
    ja3_optional: Optional[str] = None
    hassh: Optional[str] = None
    ech_observed: Optional[bool] = None
    ssh_kex_class: Optional[str] = None
    ssh_cipher_class: Optional[str] = None
    ssh_mac_class: Optional[str] = None
    baseline_fp_prevalence: Optional[float] = None
    new_edge_score: Optional[float] = None
    session_rate_5m: Optional[float] = None
    fanout_5m: Optional[float] = None
    ts_start: Optional[str] = None

    # Week 8 decomposable fusion payload
    s_total: float = 0.0
    prediction: int = 0
    threshold: float = 0.45
    dominant_signal: str = "rule"
    component_scores: Dict[str, float] = field(
        default_factory=lambda: {"rule": 0.0, "classifier": 0.0, "anomaly": 0.0, "sequence": 0.0}
    )
    weighted_contributions: Dict[str, float] = field(
        default_factory=lambda: {"rule": 0.0, "classifier": 0.0, "anomaly": 0.0, "sequence": 0.0}
    )
    reason_codes: List[str] = field(default_factory=list)
    active_reason_codes: List[str] = field(default_factory=list)

    def present_l5_l6_fields(self) -> List[str]:
        fields: List[str] = []
        check_map = {
            "security_proto": self.security_proto,
            "tls_version": self.tls_version,
            "tls_cipher_class": self.tls_cipher_class,
            "cert_status": self.cert_status,
            "cert_spki_token": self.cert_spki_token,
            "cert_issuer_class": self.cert_issuer_class,
            "cert_validity_days": self.cert_validity_days,
            "ja4": self.ja4,
            "ja3_optional": self.ja3_optional,
            "hassh": self.hassh,
            "ech_observed": self.ech_observed,
            "ssh_kex_class": self.ssh_kex_class,
            "ssh_cipher_class": self.ssh_cipher_class,
            "ssh_mac_class": self.ssh_mac_class,
            "baseline_fp_prevalence": self.baseline_fp_prevalence,
            "new_edge_score": self.new_edge_score,
            "session_rate_5m": self.session_rate_5m,
            "fanout_5m": self.fanout_5m,
        }
        for name, val in check_map.items():
            if val is not None:
                fields.append(name)
        return fields

    @classmethod
    def from_row_and_fusion(
        cls,
        row: Mapping[str, Any],
        fusion_record: Optional[Mapping[str, Any]] = None,
        default_index: int = 0,
    ) -> "FlowEvidence":
        """
        Construct FlowEvidence from a telemetry row and optional Week 8 fusion record.
        Explicitly strips/ignores ground-truth columns (`label`, `attack_family`, etc.).
        """
        safe_row = {k: v for k, v in row.items() if str(k).lower() not in GROUND_TRUTH_COLUMNS}

        src_host = _opt_str(safe_row, "src_host_token") or f"host_src_{default_index}"
        dst_host = _opt_str(safe_row, "dst_host_token") or f"host_dst_{default_index}"

        flow_order_id = _req_int(safe_row, "flow_order_id", default=default_index)
        window_id = _req_int(
            safe_row,
            "window_id",
            default=_req_int(safe_row, "time_bucket", default=0),
        )
        raw_session = _opt_str(safe_row, "session_id")
        session_id = raw_session if raw_session is not None else f"sess_{src_host}_{window_id}_{flow_order_id}"
        raw_event = _opt_str(safe_row, "event_id")
        event_id = raw_event if raw_event is not None else f"evt_{session_id}_{flow_order_id}"

        dst_port = _req_int(safe_row, "dst_port", default=0)
        tls_ver = _opt_str(safe_row, "tls_version")
        ja4 = _opt_str(safe_row, "ja4")
        ja3 = _opt_str(safe_row, "ja3_optional")
        cert_status = _opt_str(safe_row, "cert_status")
        cert_spki = _opt_str(safe_row, "cert_spki_token")
        sec_proto = _opt_str(safe_row, "security_proto")
        hassh = _opt_str(safe_row, "hassh") or _opt_str(safe_row, "hassh_or_ja4ssh")
        ssh_kex = _opt_str(safe_row, "ssh_kex_class")

        is_tls_raw = _opt_bool(safe_row, "is_tls_port")
        is_tls_port = (
            is_tls_raw
            if is_tls_raw is not None
            else (
                dst_port in {443, 8443, 465, 993, 995, 636, 853}
                or (sec_proto is not None and "tls" in sec_proto.lower())
                or tls_ver is not None
                or ja4 is not None
            )
        )
        is_ssh_raw = _opt_bool(safe_row, "is_ssh_port")
        is_ssh_port = (
            is_ssh_raw
            if is_ssh_raw is not None
            else (
                dst_port in {22, 2222}
                or (sec_proto is not None and "ssh" in sec_proto.lower())
                or hassh is not None
                or ssh_kex is not None
            )
        )

        retrans_in = _req_int(safe_row, "retransmitted_in_pkts", default=0)
        retrans_out = _req_int(safe_row, "retransmitted_out_pkts", default=0)
        retrans_total = _req_int(safe_row, "retransmitted_pkts", default=retrans_in + retrans_out)

        # Extract fusion evidence from explicit fusion_record or pre-scored row columns
        f_rec: Mapping[str, Any] = fusion_record if fusion_record is not None else safe_row
        comp_raw = f_rec.get("component_scores")
        if isinstance(comp_raw, Mapping):
            comp_scores = {
                k: _req_float(comp_raw, k, default=0.0)
                for k in ("rule", "classifier", "anomaly", "sequence")
            }
        else:
            comp_scores = {
                "rule": _req_float(f_rec, "s_rule", default=0.0),
                "classifier": _req_float(f_rec, "s_classifier", default=0.0),
                "anomaly": _req_float(f_rec, "s_anomaly", default=0.0),
                "sequence": _req_float(f_rec, "s_sequence", default=0.0),
            }

        contrib_raw = f_rec.get("weighted_contributions")
        if isinstance(contrib_raw, Mapping):
            weighted_contribs = {
                k: _req_float(contrib_raw, k, default=0.0)
                for k in ("rule", "classifier", "anomaly", "sequence")
            }
        else:
            weighted_contribs = {
                k: _req_float(f_rec, f"w_{k}", default=comp_scores[k])
                for k in ("rule", "classifier", "anomaly", "sequence")
            }

        threshold = _req_float(f_rec, "threshold", default=0.45)
        if isinstance(contrib_raw, Mapping):
            default_s_total = float(sum(weighted_contribs.values()))
        else:
            default_s_total = float(max(comp_scores.values()))
        if math.isnan(default_s_total) or math.isinf(default_s_total):
            default_s_total = 0.0
        s_total = _req_float(f_rec, "s_total", default=default_s_total)
        if math.isnan(s_total) or math.isinf(s_total):
            s_total = 0.0
        s_total = max(0.0, min(1.0, s_total))
        prediction = _req_int(f_rec, "prediction", default=int(s_total >= threshold))
        dominant_signal = _opt_str(f_rec, "dominant_signal")
        if dominant_signal is None:
            dominant_signal = max(weighted_contribs.items(), key=lambda kv: kv[1])[0]

        raw_reasons = f_rec.get("reason_codes", [])
        if isinstance(raw_reasons, str):
            reason_codes = [r.strip() for r in raw_reasons.split(",") if r.strip()]
        elif isinstance(raw_reasons, Sequence):
            reason_codes = [str(r).strip() for r in raw_reasons if not _is_missing(r)]
        else:
            reason_codes = []

        # Deduplicate preserving order; filter WITHIN_BENIGN_CALIBRATED_BASELINE from active attack signals
        dedup_reasons: List[str] = []
        for code in reason_codes:
            if code not in dedup_reasons:
                dedup_reasons.append(code)
        active_reasons = [code for code in dedup_reasons if code != BENIGN_BASELINE_REASON]

        return cls(
            event_id=event_id,
            session_id=session_id,
            flow_order_id=flow_order_id,
            window_id=window_id,
            src_host_token=src_host,
            dst_host_token=dst_host,
            dst_port=dst_port,
            proto=_req_int(safe_row, "proto", default=6),
            in_bytes=_req_int(safe_row, "in_bytes", default=_req_int(safe_row, "orig_bytes", 0)),
            out_bytes=_req_int(safe_row, "out_bytes", default=_req_int(safe_row, "resp_bytes", 0)),
            in_pkts=_req_int(safe_row, "in_pkts", default=_req_int(safe_row, "orig_pkts", 0)),
            out_pkts=_req_int(safe_row, "out_pkts", default=_req_int(safe_row, "resp_pkts", 0)),
            flow_duration_ms=_req_int(safe_row, "flow_duration_ms", default=_req_int(safe_row, "duration", 0)),
            tcp_flags=_req_int(safe_row, "tcp_flags", default=0),
            retransmitted_pkts=retrans_total,
            bytes_ratio=_req_float(safe_row, "bytes_ratio", default=1.0),
            pkts_ratio=_req_float(safe_row, "pkts_ratio", default=1.0),
            src_to_dst_avg_throughput=_req_float(safe_row, "src_to_dst_avg_throughput", default=0.0),
            is_tls_port=is_tls_port,
            is_ssh_port=is_ssh_port,
            security_proto=sec_proto,
            tls_version=tls_ver,
            tls_cipher_class=_opt_str(safe_row, "tls_cipher_class"),
            cert_status=cert_status,
            cert_spki_token=cert_spki,
            cert_issuer_class=_opt_str(safe_row, "cert_issuer_class"),
            cert_validity_days=_opt_float(safe_row, "cert_validity_days"),
            ja4=ja4,
            ja3_optional=ja3,
            hassh=hassh,
            ech_observed=_opt_bool(safe_row, "ech_observed"),
            ssh_kex_class=ssh_kex,
            ssh_cipher_class=_opt_str(safe_row, "ssh_cipher_class"),
            ssh_mac_class=_opt_str(safe_row, "ssh_mac_class"),
            baseline_fp_prevalence=_opt_float(safe_row, "baseline_fp_prevalence"),
            new_edge_score=_opt_float(safe_row, "new_edge_score"),
            session_rate_5m=_opt_float(safe_row, "session_rate_5m"),
            fanout_5m=_opt_float(safe_row, "fanout_5m"),
            ts_start=_opt_str(safe_row, "ts_start"),
            s_total=s_total,
            prediction=prediction,
            threshold=threshold,
            dominant_signal=dominant_signal,
            component_scores=comp_scores,
            weighted_contributions=weighted_contribs,
            reason_codes=dedup_reasons,
            active_reason_codes=active_reasons,
        )

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "event_id": self.event_id,
                "session_id": self.session_id,
                "flow_order_id": self.flow_order_id,
                "window_id": self.window_id,
                "src_host_token": self.src_host_token,
                "dst_host_token": self.dst_host_token,
                "dst_port": self.dst_port,
                "proto": self.proto,
                "in_bytes": self.in_bytes,
                "out_bytes": self.out_bytes,
                "in_pkts": self.in_pkts,
                "out_pkts": self.out_pkts,
                "flow_duration_ms": self.flow_duration_ms,
                "tcp_flags": self.tcp_flags,
                "retransmitted_pkts": self.retransmitted_pkts,
                "bytes_ratio": self.bytes_ratio,
                "pkts_ratio": self.pkts_ratio,
                "src_to_dst_avg_throughput": self.src_to_dst_avg_throughput,
                "is_tls_port": self.is_tls_port,
                "is_ssh_port": self.is_ssh_port,
                "security_proto": self.security_proto,
                "tls_version": self.tls_version,
                "tls_cipher_class": self.tls_cipher_class,
                "cert_status": self.cert_status,
                "cert_spki_token": self.cert_spki_token,
                "cert_issuer_class": self.cert_issuer_class,
                "cert_validity_days": self.cert_validity_days,
                "ja4": self.ja4,
                "ja3_optional": self.ja3_optional,
                "hassh": self.hassh,
                "ech_observed": self.ech_observed,
                "ssh_kex_class": self.ssh_kex_class,
                "ssh_cipher_class": self.ssh_cipher_class,
                "ssh_mac_class": self.ssh_mac_class,
                "baseline_fp_prevalence": self.baseline_fp_prevalence,
                "new_edge_score": self.new_edge_score,
                "session_rate_5m": self.session_rate_5m,
                "fanout_5m": self.fanout_5m,
                "ts_start": self.ts_start,
                "s_total": self.s_total,
                "prediction": self.prediction,
                "threshold": self.threshold,
                "dominant_signal": self.dominant_signal,
                "component_scores": self.component_scores,
                "weighted_contributions": self.weighted_contributions,
                "reason_codes": self.reason_codes,
                "active_reason_codes": self.active_reason_codes,
            }
        )


@dataclass(frozen=True)
class AttackMapping:
    """Evidence-grounded MITRE ATT&CK technique mapping (SR-05, Section 15 Stage 5)."""

    technique_id: str
    technique_name: str
    tactic: str
    confidence: float
    rationale: str
    observed_signals: List[str]
    event_ids: List[str]
    session_ids: List[str]
    supporting_event_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.technique_id.strip():
            raise ValueError("AttackMapping requires a non-empty technique_id.")
        if not self.rationale.strip():
            raise ValueError(f"AttackMapping {self.technique_id} requires an explicit rationale.")
        if not self.observed_signals:
            raise ValueError(f"AttackMapping {self.technique_id} requires observed_signals (SR-05).")
        if not self.event_ids and self.supporting_event_ids:
            object.__setattr__(self, "event_ids", list(self.supporting_event_ids))
        if not self.supporting_event_ids and self.event_ids:
            object.__setattr__(self, "supporting_event_ids", list(self.event_ids))
        if not self.event_ids or not self.session_ids or not self.supporting_event_ids:
            raise ValueError(
                f"AttackMapping {self.technique_id} requires supporting event_ids and session_ids (SR-05)."
            )

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "technique_id": self.technique_id,
                "technique_name": self.technique_name,
                "tactic": self.tactic,
                "confidence": float(max(0.0, min(1.0, self.confidence))),
                "rationale": self.rationale,
                "observed_signals": list(self.observed_signals),
                "event_ids": list(self.event_ids),
                "session_ids": list(self.session_ids),
                "supporting_event_ids": list(self.supporting_event_ids),
            }
        )


@dataclass(frozen=True)
class IncidentCluster:
    """Correlated group of flows belonging to a candidate incident."""

    incident_id: str
    primary_host_token: str
    window_id: int
    target_host_tokens: List[str]
    session_ids: List[str]
    event_ids: List[str]
    flows: List[FlowEvidence]
    correlation_keys: List[str]
    max_s_total: float
    mean_s_total: float
    alerted_flow_count: int
    total_flow_count: int
    dominant_signal: str
    dominant_signal_counts: Dict[str, int]
    component_scores_max: Dict[str, float]
    weighted_contributions_max: Dict[str, float]
    all_reason_codes: List[str]
    active_reason_codes: List[str]
    l5_l6_availability: Dict[str, Any]
    metrics_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "incident_id": self.incident_id,
                "primary_host_token": self.primary_host_token,
                "window_id": self.window_id,
                "target_host_tokens": self.target_host_tokens,
                "session_ids": self.session_ids,
                "event_ids": self.event_ids,
                "correlation_keys": self.correlation_keys,
                "max_s_total": self.max_s_total,
                "mean_s_total": self.mean_s_total,
                "alerted_flow_count": self.alerted_flow_count,
                "total_flow_count": self.total_flow_count,
                "dominant_signal": self.dominant_signal,
                "dominant_signal_counts": self.dominant_signal_counts,
                "component_scores_max": self.component_scores_max,
                "weighted_contributions_max": self.weighted_contributions_max,
                "all_reason_codes": self.all_reason_codes,
                "active_reason_codes": self.active_reason_codes,
                "l5_l6_availability": self.l5_l6_availability,
                "metrics_summary": self.metrics_summary,
                "flows": [f.to_dict() for f in self.flows],
            }
        )


@dataclass(frozen=True)
class GraphNode:
    """Typed node inside an IncidentEvidenceGraph."""

    node_id: str
    node_type: NodeType
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "node_id": self.node_id,
                "node_type": self.node_type,
                "label": self.label,
                "attributes": self.attributes,
            }
        )


@dataclass(frozen=True)
class GraphEdge:
    """Directed, deterministic edge inside an IncidentEvidenceGraph."""

    edge_id: str
    source_id: str
    target_id: str
    relation: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "edge_id": self.edge_id,
                "source_id": self.source_id,
                "target_id": self.target_id,
                "relation": self.relation,
                "attributes": self.attributes,
            }
        )


@dataclass(frozen=True)
class IncidentEvidenceGraph:
    """Deterministic, JSON-serializable incident evidence graph."""

    incident_id: str
    window_id: int
    primary_host_token: str
    target_host_tokens: List[str]
    session_ids: List[str]
    event_ids: List[str]
    created_at_utc: str
    detector_versions: Dict[str, str]
    fusion_summary: Dict[str, Any]
    l5_l6_observables: Dict[str, Any]
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    attack_mappings: List[AttackMapping]
    unmapped_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "incident_id": self.incident_id,
                "window_id": self.window_id,
                "primary_host_token": self.primary_host_token,
                "target_host_tokens": self.target_host_tokens,
                "session_ids": self.session_ids,
                "event_ids": self.event_ids,
                "created_at_utc": self.created_at_utc,
                "detector_versions": self.detector_versions,
                "fusion_summary": self.fusion_summary,
                "l5_l6_observables": self.l5_l6_observables,
                "nodes": [n.to_dict() for n in self.nodes],
                "edges": [e.to_dict() for e in self.edges],
                "attack_mappings": [m.to_dict() for m in self.attack_mappings],
                "unmapped_rationale": self.unmapped_rationale,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def flow_has_tls(flow: FlowEvidence) -> bool:
    return bool(
        flow.is_tls_port
        or flow.dst_port in {443, 8443}
        or flow.tls_version is not None
        or flow.ja4 is not None
    )


def flow_is_syn_only_probe(flow: FlowEvidence) -> bool:
    return bool(
        flow.out_pkts == 0
        and (
            flow.tcp_flags == 2
            or "RULE_L4_SYN_SCAN_OR_FLOOD" in flow.active_reason_codes
        )
    )


def flow_is_established_tls(flow: FlowEvidence) -> bool:
    if not flow_has_tls(flow):
        return False
    if flow_is_syn_only_probe(flow):
        return False
    if flow.out_pkts < 1 or flow.out_bytes <= 0:
        return False
    return bool(
        flow.in_pkts >= 2
        or flow.tls_version is not None
        or flow.cert_status is not None
        or flow.ja4 is not None
    )


def flow_has_ssh(flow: FlowEvidence) -> bool:
    return bool(
        flow.is_ssh_port
        or flow.dst_port in {22, 2222}
        or flow.hassh is not None
        or flow.ssh_kex_class is not None
    )


def flow_is_established_ssh(flow: FlowEvidence) -> bool:
    if not flow_has_ssh(flow):
        return False
    if flow_is_syn_only_probe(flow):
        return False
    return bool(flow.out_pkts >= 1 and flow.out_bytes > 0)


def flow_is_ssh_attempt(flow: FlowEvidence) -> bool:
    if not flow_has_ssh(flow):
        return False
    if flow.out_pkts == 0 and flow.in_pkts <= 1 and (
        flow.tcp_flags == 2 or "RULE_L4_SYN_SCAN_OR_FLOOD" in flow.active_reason_codes
    ):
        return False
    if (
        flow.out_pkts == 0
        and (flow.tcp_flags == 2 or "RULE_L4_SYN_SCAN_OR_FLOOD" in flow.active_reason_codes)
        and "RULE_SSH_BRUTEFORCE_CHURN" not in flow.active_reason_codes
    ):
        return False
    return bool(flow.in_pkts >= 2 or flow.out_pkts >= 1)


def flow_is_cr01(flow: FlowEvidence) -> bool:
    return bool(
        "RULE_CR01_DEPRECATED_TLS_VERSION" in flow.active_reason_codes
        or (flow.tls_version is not None and flow.tls_version.strip().upper() in DEPRECATED_TLS_VERSIONS)
    )


def flow_is_cr02(flow: FlowEvidence) -> bool:
    return bool(
        "RULE_CR02_CERT_TRUST_ANOMALY" in flow.active_reason_codes
        or (flow.cert_status is not None and flow.cert_status.strip().lower() in UNTRUSTED_CERT_STATUSES)
    )


def flow_is_cr04(flow: FlowEvidence) -> bool:
    return bool(
        "RULE_CR04_WEAK_SSH_KEX" in flow.active_reason_codes
        or (
            flow.ssh_kex_class is not None
            and any(w in flow.ssh_kex_class.strip().lower() for w in WEAK_SSH_KEX_MARKERS)
        )
    )
