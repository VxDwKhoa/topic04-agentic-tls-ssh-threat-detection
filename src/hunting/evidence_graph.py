"""
Deterministic Incident Evidence Graph Builder for Topic 04 Week 9.

Constructs serializable, privacy-preserving graphs linking incidents, pseudonymized
hosts (`src_host_token`, `dst_host_token`), sessions/flows, optional L5/L6
fingerprints and certificates, Week 8 analytic signals, and grounded ATT&CK mappings.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Sequence

from src.hunting.schemas import (
    AttackMapping,
    GraphEdge,
    GraphNode,
    IncidentCluster,
    IncidentEvidenceGraph,
)


class EvidenceGraphBuilder:
    """Builds deterministic, JSON-serializable IncidentEvidenceGraph objects."""

    DEFAULT_DETECTOR_VERSIONS: Dict[str, str] = {
        "week5_rules_baseline": "1.0.0",
        "week5_rf_binary": "1.0.0",
        "week6_isolation_forest": "1.0.0",
        "week7_lstm_temporal": "1.0.0",
        "week8_calibrated_fusion": "1.0.0",
        "week9_evidence_graph": "1.0.0",
    }

    def __init__(
        self,
        default_detector_versions: Optional[Dict[str, str]] = None,
        deterministic_timestamp: str = "1970-01-01T00:00:00Z",
    ) -> None:
        self.default_detector_versions = dict(default_detector_versions or self.DEFAULT_DETECTOR_VERSIONS)
        self.deterministic_timestamp = str(deterministic_timestamp)

    @staticmethod
    def _short_hash(prefix: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    @staticmethod
    def _edge_id(source_id: str, relation: str, target_id: str) -> str:
        digest = hashlib.sha256(f"{source_id}|{relation}|{target_id}".encode("utf-8")).hexdigest()[:16]
        return f"edge_{digest}"

    def build(
        self,
        cluster: IncidentCluster,
        attack_mappings: Sequence[AttackMapping] = (),
        unmapped_rationale: Optional[str] = None,
        detector_versions: Optional[Dict[str, str]] = None,
    ) -> IncidentEvidenceGraph:
        nodes: Dict[str, GraphNode] = {}
        edges: Dict[str, GraphEdge] = {}

        def add_node(node: GraphNode) -> None:
            if node.node_id not in nodes:
                nodes[node.node_id] = node

        def add_edge(source_id: str, relation: str, target_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
            eid = self._edge_id(source_id, relation, target_id)
            if eid not in edges:
                edges[eid] = GraphEdge(
                    edge_id=eid,
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    attributes=attributes or {},
                )

        # 1. Incident root node
        inc_id = cluster.incident_id
        add_node(
            GraphNode(
                node_id=inc_id,
                node_type="incident",
                label=f"Incident {inc_id}",
                attributes={
                    "window_id": cluster.window_id,
                    "primary_host_token": cluster.primary_host_token,
                    "total_flow_count": cluster.total_flow_count,
                    "alerted_flow_count": cluster.alerted_flow_count,
                    "max_s_total": cluster.max_s_total,
                    "mean_s_total": cluster.mean_s_total,
                    "dominant_signal": cluster.dominant_signal,
                    "correlation_keys": cluster.correlation_keys,
                },
            )
        )

        # 2. Source & destination pseudonymized host nodes
        src_node_id = f"host_{cluster.primary_host_token}"
        add_node(
            GraphNode(
                node_id=src_node_id,
                node_type="host",
                label=cluster.primary_host_token,
                attributes={"host_token": cluster.primary_host_token, "role": "source"},
            )
        )
        add_edge(inc_id, "INVOLVES_SOURCE", src_node_id)

        for dst_token in cluster.target_host_tokens:
            dst_node_id = f"host_{dst_token}"
            add_node(
                GraphNode(
                    node_id=dst_node_id,
                    node_type="host",
                    label=dst_token,
                    attributes={"host_token": dst_token, "role": "destination"},
                )
            )
            add_edge(inc_id, "TARGETS_DESTINATION", dst_node_id)

        # 3. Session nodes, optional L5/L6 fingerprint/certificate nodes, and analytic signal nodes
        signal_to_sessions: Dict[str, List[str]] = {}
        for flow in cluster.flows:
            sess_node_id = f"sess_{flow.session_id}"
            sess_attrs: Dict[str, Any] = {
                "event_id": flow.event_id,
                "session_id": flow.session_id,
                "flow_order_id": flow.flow_order_id,
                "window_id": flow.window_id,
                "dst_port": flow.dst_port,
                "proto": flow.proto,
                "in_bytes": flow.in_bytes,
                "out_bytes": flow.out_bytes,
                "in_pkts": flow.in_pkts,
                "out_pkts": flow.out_pkts,
                "flow_duration_ms": flow.flow_duration_ms,
                "retransmitted_pkts": flow.retransmitted_pkts,
                "bytes_ratio": flow.bytes_ratio,
                "is_tls_port": flow.is_tls_port,
                "is_ssh_port": flow.is_ssh_port,
                "s_total": flow.s_total,
                "prediction": flow.prediction,
                "threshold": flow.threshold,
                "dominant_signal": flow.dominant_signal,
                "component_scores": flow.component_scores,
                "weighted_contributions": flow.weighted_contributions,
                "reason_codes": flow.reason_codes,
                "active_reason_codes": flow.active_reason_codes,
            }
            for opt_key in (
                "tls_version",
                "cert_status",
                "cert_spki_token",
                "ja4",
                "hassh",
                "ech_observed",
                "ssh_kex_class",
                "baseline_fp_prevalence",
                "new_edge_score",
            ):
                opt_val = getattr(flow, opt_key)
                if opt_val is not None:
                    sess_attrs[opt_key] = opt_val

            add_node(
                GraphNode(
                    node_id=sess_node_id,
                    node_type="session",
                    label=flow.session_id,
                    attributes=sess_attrs,
                )
            )
            add_edge(src_node_id, "INITIATED_SESSION", sess_node_id)
            add_edge(sess_node_id, "CONNECTED_TO", f"host_{flow.dst_host_token}", {"dst_port": flow.dst_port})

            if flow.ja4:
                fp_id = self._short_hash("fp_ja4", flow.ja4)
                add_node(
                    GraphNode(
                        node_id=fp_id,
                        node_type="fingerprint",
                        label=f"JA4:{flow.ja4}",
                        attributes={
                            "fingerprint_type": "ja4",
                            "fingerprint_value": flow.ja4,
                            "baseline_fp_prevalence": flow.baseline_fp_prevalence,
                        },
                    )
                )
                add_edge(sess_node_id, "USES_FINGERPRINT", fp_id)

            if flow.hassh:
                fp_id = self._short_hash("fp_hassh", flow.hassh)
                add_node(
                    GraphNode(
                        node_id=fp_id,
                        node_type="fingerprint",
                        label=f"HASSH:{flow.hassh}",
                        attributes={
                            "fingerprint_type": "hassh",
                            "fingerprint_value": flow.hassh,
                            "baseline_fp_prevalence": flow.baseline_fp_prevalence,
                        },
                    )
                )
                add_edge(sess_node_id, "USES_FINGERPRINT", fp_id)

            if flow.cert_spki_token or flow.cert_status:
                cert_key = f"{flow.cert_spki_token or 'unknown'}|{flow.cert_status or 'unknown'}"
                cert_id = self._short_hash("cert", cert_key)
                add_node(
                    GraphNode(
                        node_id=cert_id,
                        node_type="certificate",
                        label=flow.cert_spki_token or f"cert_status:{flow.cert_status}",
                        attributes={
                            "cert_spki_token": flow.cert_spki_token,
                            "cert_status": flow.cert_status,
                            "cert_issuer_class": flow.cert_issuer_class,
                            "cert_validity_days": flow.cert_validity_days,
                        },
                    )
                )
                add_edge(sess_node_id, "PRESENTED_CERT", cert_id)

            # Analytic signals (active reason codes + high model channels)
            for code in flow.active_reason_codes:
                sig_id = f"sig_{code}"
                add_node(
                    GraphNode(
                        node_id=sig_id,
                        node_type="analytic_signal",
                        label=code,
                        attributes={"signal_kind": "reason_code", "code": code},
                    )
                )
                add_edge(sess_node_id, "TRIGGERED_SIGNAL", sig_id)
                signal_to_sessions.setdefault(sig_id, []).append(sess_node_id)

            for ch, val in flow.component_scores.items():
                if val >= 0.50 or (flow.s_total >= flow.threshold and ch == flow.dominant_signal):
                    sig_id = f"sig_channel_{ch}"
                    add_node(
                        GraphNode(
                            node_id=sig_id,
                            node_type="analytic_signal",
                            label=f"channel:{ch}",
                            attributes={
                                "signal_kind": "fusion_channel",
                                "channel": ch,
                                "max_score": cluster.component_scores_max.get(ch, val),
                            },
                        )
                    )
                    add_edge(sess_node_id, "TRIGGERED_SIGNAL", sig_id, {"score": val})
                    signal_to_sessions.setdefault(sig_id, []).append(sess_node_id)

        # 4. Grounded MITRE ATT&CK nodes
        for mapping in attack_mappings:
            attck_id = f"attck_{mapping.technique_id}"
            add_node(
                GraphNode(
                    node_id=attck_id,
                    node_type="attack_technique",
                    label=f"{mapping.technique_id} — {mapping.technique_name}",
                    attributes=mapping.to_dict(),
                )
            )
            for sid in mapping.session_ids:
                sess_node_id = f"sess_{sid}"
                if sess_node_id in nodes:
                    add_edge(sess_node_id, "EVIDENCED_BY_SESSION", attck_id)
            for sig in mapping.observed_signals:
                sig_id = f"sig_{sig}"
                if sig_id in nodes:
                    add_edge(sig_id, "SUPPORTS_TECHNIQUE", attck_id)
            add_edge(inc_id, "MAPPED_TO_TECHNIQUE", attck_id)

        ts_candidates = [f.ts_start for f in cluster.flows if f.ts_start]
        created_at = ts_candidates[0] if ts_candidates else self.deterministic_timestamp

        sorted_nodes = [nodes[k] for k in sorted(nodes.keys())]
        sorted_edges = [edges[k] for k in sorted(edges.keys())]

        fusion_summary = {
            "max_s_total": cluster.max_s_total,
            "mean_s_total": cluster.mean_s_total,
            "dominant_signal": cluster.dominant_signal,
            "dominant_signal_counts": cluster.dominant_signal_counts,
            "component_scores_max": cluster.component_scores_max,
            "weighted_contributions_max": cluster.weighted_contributions_max,
            "reason_codes_union": cluster.all_reason_codes,
            "active_reason_codes_union": cluster.active_reason_codes,
        }

        return IncidentEvidenceGraph(
            incident_id=inc_id,
            window_id=cluster.window_id,
            primary_host_token=cluster.primary_host_token,
            target_host_tokens=cluster.target_host_tokens,
            session_ids=cluster.session_ids,
            event_ids=cluster.event_ids,
            created_at_utc=created_at,
            detector_versions=dict(detector_versions or self.default_detector_versions),
            fusion_summary=fusion_summary,
            l5_l6_observables=dict(cluster.l5_l6_availability),
            nodes=sorted_nodes,
            edges=sorted_edges,
            attack_mappings=list(attack_mappings),
            unmapped_rationale=unmapped_rationale,
        )
