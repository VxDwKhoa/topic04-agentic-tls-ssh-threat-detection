"""
Incident Correlator for Topic 04 Week 9 Threat-Hunting Pipeline.

Correlates FlowEvidence records deterministically by `(src_host_token, window_id)`
and tracks shared L5/L6 certificate/fingerprint links (`cert_spki_token`, `ja4`, `hassh`)
when present (SR-02, Section 15 Stage 1).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.hunting.schemas import (
    BENIGN_BASELINE_REASON,
    FlowEvidence,
    IncidentCluster,
    L5_L6_OBSERVABLE_FIELDS,
    flow_has_ssh,
    flow_has_tls,
    flow_is_cr01,
    flow_is_cr02,
    flow_is_cr04,
    flow_is_established_ssh,
    flow_is_established_tls,
    flow_is_ssh_attempt,
)


class IncidentCorrelator:
    """Groups correlated flow evidence into deterministic IncidentCluster objects."""

    def __init__(
        self,
        score_alert_threshold: Optional[float] = None,
        max_flows_per_incident: int = 50,
        include_low_signal_clusters: bool = False,
    ) -> None:
        if max_flows_per_incident < 1:
            raise ValueError(f"max_flows_per_incident must be >= 1, got {max_flows_per_incident}")
        self.score_alert_threshold = score_alert_threshold
        self.max_flows_per_incident = int(max_flows_per_incident)
        self.include_low_signal_clusters = bool(include_low_signal_clusters)

    @staticmethod
    def compute_incident_id(
        primary_host_token: str,
        window_id: int,
        session_ids: Sequence[str],
    ) -> str:
        payload = "|".join([str(primary_host_token), str(window_id)] + sorted(set(str(s) for s in session_ids)))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"inc_{digest}"

    def _is_flow_suspicious(self, flow: FlowEvidence) -> bool:
        thr = self.score_alert_threshold if self.score_alert_threshold is not None else flow.threshold
        has_active_rule = any(code.startswith("RULE_") for code in flow.active_reason_codes)
        return bool(
            flow.prediction == 1
            or flow.s_total >= thr
            or has_active_rule
            or flow.component_scores.get("rule", 0.0) >= 0.45
            or flow_is_cr01(flow)
            or flow_is_cr02(flow)
            or flow_is_cr04(flow)
        )

    def correlate(
        self,
        flows: Sequence[FlowEvidence],
        include_low_signal_clusters: Optional[bool] = None,
    ) -> List[IncidentCluster]:
        if not flows:
            return []

        include_low = (
            self.include_low_signal_clusters
            if include_low_signal_clusters is None
            else bool(include_low_signal_clusters)
        )

        # Index shared L5/L6 identifiers per window to annotate cross-host links
        l5_cert_hosts: Dict[Tuple[int, str], set[str]] = defaultdict(set)
        l5_ja4_hosts: Dict[Tuple[int, str], set[str]] = defaultdict(set)
        l5_hassh_hosts: Dict[Tuple[int, str], set[str]] = defaultdict(set)

        grouped: Dict[Tuple[str, int], List[FlowEvidence]] = defaultdict(list)
        for flow in flows:
            grouped[(flow.src_host_token, flow.window_id)].append(flow)
            if flow.cert_spki_token:
                l5_cert_hosts[(flow.window_id, flow.cert_spki_token)].add(flow.src_host_token)
            if flow.ja4:
                l5_ja4_hosts[(flow.window_id, flow.ja4)].add(flow.src_host_token)
            if flow.hassh:
                l5_hassh_hosts[(flow.window_id, flow.hassh)].add(flow.src_host_token)

        clusters: List[IncidentCluster] = []
        for (src_host, window_id), group_flows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            alerted_flows = [f for f in group_flows if self._is_flow_suspicious(f)]
            if not include_low and not alerted_flows:
                continue

            # Prioritize alerted/highest-score flows when capping incident size, then restore chronological order
            if len(group_flows) > self.max_flows_per_incident:
                ranked = sorted(
                    group_flows,
                    key=lambda f: (
                        0 if self._is_flow_suspicious(f) else 1,
                        -f.s_total,
                        f.flow_order_id,
                        f.session_id,
                        f.event_id,
                    ),
                )[: self.max_flows_per_incident]
            else:
                ranked = list(group_flows)

            ordered_flows = sorted(ranked, key=lambda f: (f.flow_order_id, f.session_id, f.event_id))

            target_hosts = sorted({f.dst_host_token for f in ordered_flows})
            session_ids = [f.session_id for f in ordered_flows]
            event_ids = [f.event_id for f in ordered_flows]
            incident_id = self.compute_incident_id(src_host, window_id, session_ids)

            correlation_keys = [f"src_host_window:{src_host}:{window_id}"]
            shared_certs = sorted(
                {f.cert_spki_token for f in ordered_flows if f.cert_spki_token}
            )
            for cert in shared_certs:
                peers = l5_cert_hosts[(window_id, cert)]
                if len(peers) > 1 or sum(1 for f in ordered_flows if f.cert_spki_token == cert) > 1:
                    correlation_keys.append(f"shared_cert_spki:{cert}")

            shared_ja4 = sorted({f.ja4 for f in ordered_flows if f.ja4})
            for fp in shared_ja4:
                peers = l5_ja4_hosts[(window_id, fp)]
                if len(peers) > 1 or sum(1 for f in ordered_flows if f.ja4 == fp) > 1:
                    correlation_keys.append(f"shared_ja4:{fp}")

            shared_hassh = sorted({f.hassh for f in ordered_flows if f.hassh})
            for fp in shared_hassh:
                peers = l5_hassh_hosts[(window_id, fp)]
                if len(peers) > 1 or sum(1 for f in ordered_flows if f.hassh == fp) > 1:
                    correlation_keys.append(f"shared_hassh:{fp}")

            s_totals = [f.s_total for f in ordered_flows]
            max_s_total = float(max(s_totals))
            mean_s_total = float(sum(s_totals) / len(s_totals))

            dom_counter = Counter(f.dominant_signal for f in ordered_flows)
            top_flow = max(ordered_flows, key=lambda f: (f.s_total, -f.flow_order_id))

            comp_max = {
                ch: float(max(f.component_scores.get(ch, 0.0) for f in ordered_flows))
                for ch in ("rule", "classifier", "anomaly", "sequence")
            }
            contrib_max = {
                ch: float(max(f.weighted_contributions.get(ch, 0.0) for f in ordered_flows))
                for ch in ("rule", "classifier", "anomaly", "sequence")
            }

            all_reasons: List[str] = []
            active_reasons: List[str] = []
            for f in ordered_flows:
                for code in f.reason_codes:
                    if code not in all_reasons:
                        all_reasons.append(code)
                for code in f.active_reason_codes:
                    if code != BENIGN_BASELINE_REASON and code not in active_reasons:
                        active_reasons.append(code)

            present_fields_set: set[str] = set()
            for f in ordered_flows:
                present_fields_set.update(f.present_l5_l6_fields())
            fields_present = sorted(present_fields_set)
            missing_fields = [c for c in L5_L6_OBSERVABLE_FIELDS if c not in present_fields_set and c != "hassh_or_ja4ssh"]

            l5_l6_availability = {
                "available": len(fields_present) > 0,
                "fields_present": fields_present,
                "missing_fields": missing_fields,
                "ech_observed_any": any(bool(f.ech_observed) for f in ordered_flows if f.ech_observed is not None),
                "tls_versions": sorted({f.tls_version for f in ordered_flows if f.tls_version}),
                "cert_statuses": sorted({f.cert_status for f in ordered_flows if f.cert_status}),
                "cert_spki_tokens": shared_certs,
                "ja4_fingerprints": shared_ja4,
                "hassh_fingerprints": shared_hassh,
                "ssh_kex_classes": sorted({f.ssh_kex_class for f in ordered_flows if f.ssh_kex_class}),
            }

            ssh_flows = [f for f in ordered_flows if flow_has_ssh(f)]
            tls_flows = [f for f in ordered_flows if flow_has_tls(f)]
            established_ssh_flows = [f for f in ssh_flows if flow_is_established_ssh(f)]
            ssh_attempt_flows = [f for f in ssh_flows if flow_is_ssh_attempt(f)]
            established_tls_flows = [f for f in tls_flows if flow_is_established_tls(f)]

            durations = [f.flow_duration_ms for f in ordered_flows]
            ssh_durations = [f.flow_duration_ms for f in ssh_flows]
            ssh_attempt_durations = [f.flow_duration_ms for f in ssh_attempt_flows]

            rate_vals = [f.session_rate_5m for f in ordered_flows if f.session_rate_5m is not None]
            fanout_vals = [f.fanout_5m for f in ordered_flows if f.fanout_5m is not None]
            new_edge_vals = [f.new_edge_score for f in ordered_flows if f.new_edge_score is not None]
            fp_prev_vals = [f.baseline_fp_prevalence for f in ordered_flows if f.baseline_fp_prevalence is not None]

            syn_only_count = sum(
                1 for f in ordered_flows if (f.tcp_flags & 0x02) == 0x02 and f.out_pkts == 0
            )
            low_resp_count = sum(1 for f in ordered_flows if f.out_pkts <= 1)

            metrics_summary = {
                "unique_dst_hosts": len(target_hosts),
                "unique_dst_ports": len({f.dst_port for f in ordered_flows}),
                "dst_ports": sorted({f.dst_port for f in ordered_flows}),
                "tls_flow_count": len(tls_flows),
                "established_tls_flow_count": len(established_tls_flows),
                "ssh_flow_count": len(ssh_flows),
                "established_ssh_flow_count": len(established_ssh_flows),
                "ssh_attempt_flow_count": len(ssh_attempt_flows),
                "ssh_unique_dst_hosts": len({f.dst_host_token for f in ssh_flows}),
                "established_ssh_unique_dst_hosts": len({f.dst_host_token for f in established_ssh_flows}),
                "median_duration_ms": float(median(durations)) if durations else 0.0,
                "median_ssh_duration_ms": float(median(ssh_durations)) if ssh_durations else 0.0,
                "median_ssh_attempt_duration_ms": float(median(ssh_attempt_durations)) if ssh_attempt_durations else 0.0,
                "max_in_pkts": int(max(f.in_pkts for f in ordered_flows)),
                "max_bytes_ratio": float(max(f.bytes_ratio for f in ordered_flows)),
                "max_pkts_ratio": float(max(f.pkts_ratio for f in ordered_flows)),
                "max_retransmits": int(max(f.retransmitted_pkts for f in ordered_flows)),
                "max_session_rate_5m": float(max(rate_vals)) if rate_vals else None,
                "max_fanout_5m": float(max(fanout_vals)) if fanout_vals else None,
                "max_new_edge_score": float(max(new_edge_vals)) if new_edge_vals else None,
                "min_baseline_fp_prevalence": float(min(fp_prev_vals)) if fp_prev_vals else None,
                "syn_only_flow_count": int(syn_only_count),
                "low_response_flow_ratio": float(low_resp_count / len(ordered_flows)),
                "has_host_sequence_context": any(f.component_scores.get("sequence", 0.0) > 0.0 for f in ordered_flows),
            }

            clusters.append(
                IncidentCluster(
                    incident_id=incident_id,
                    primary_host_token=src_host,
                    window_id=window_id,
                    target_host_tokens=target_hosts,
                    session_ids=session_ids,
                    event_ids=event_ids,
                    flows=ordered_flows,
                    correlation_keys=correlation_keys,
                    max_s_total=max_s_total,
                    mean_s_total=mean_s_total,
                    alerted_flow_count=len(alerted_flows),
                    total_flow_count=len(ordered_flows),
                    dominant_signal=top_flow.dominant_signal,
                    dominant_signal_counts=dict(sorted(dom_counter.items())),
                    component_scores_max=comp_max,
                    weighted_contributions_max=contrib_max,
                    all_reason_codes=all_reasons,
                    active_reason_codes=active_reasons,
                    l5_l6_availability=l5_l6_availability,
                    metrics_summary=metrics_summary,
                )
            )

        clusters.sort(
            key=lambda c: (-c.max_s_total, c.primary_host_token, c.window_id, c.incident_id)
        )
        return clusters
