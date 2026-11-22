"""
Evidence-Grounded MITRE ATT&CK Mapper for Topic 04 Week 9.

Implements SR-05, Section 15 Stage 5, Section 23, CR-03, and CR-05:
- Never reads ground-truth label columns.
- Never maps traffic mechanically solely because it uses TLS or SSH.
- Never maps ECH-only (`ech_observed=True`) or fingerprint-only (`ja4`/`hassh`)
  traffic without corroborating behavioral/analytic evidence.
- Every emitted AttackMapping cites `technique_id`, `technique_name`, `tactic`,
  `confidence`, explicit `rationale`, `observed_signals`, `event_ids`, `session_ids`,
  and `supporting_event_ids`.
- When fusion score exceeds threshold without specific behavioral observables,
  returns `[]` with an honest `unmapped_rationale`.
"""

from __future__ import annotations

from statistics import median
from typing import List, Optional, Tuple

from src.hunting.schemas import (
    AttackMapping,
    FlowEvidence,
    IncidentCluster,
    flow_has_ssh,
    flow_has_tls,
    flow_is_cr01,
    flow_is_cr02,
    flow_is_cr04,
    flow_is_established_ssh,
    flow_is_established_tls,
    flow_is_ssh_attempt,
)


class EvidenceGroundedMitreMapper:
    """Maps correlated IncidentCluster behavioral & analytic observables to ATT&CK."""

    def __init__(self, default_threshold: float = 0.45) -> None:
        self.default_threshold = float(default_threshold)

    @staticmethod
    def _flow_has_tls(flow: FlowEvidence) -> bool:
        return flow_has_tls(flow)

    @staticmethod
    def _flow_is_established_tls(flow: FlowEvidence) -> bool:
        return flow_is_established_tls(flow)

    @staticmethod
    def _flow_has_ssh(flow: FlowEvidence) -> bool:
        return flow_has_ssh(flow)

    @staticmethod
    def _flow_is_established_ssh(flow: FlowEvidence) -> bool:
        return flow_is_established_ssh(flow)

    @staticmethod
    def _flow_is_ssh_attempt(flow: FlowEvidence) -> bool:
        return flow_is_ssh_attempt(flow)

    @staticmethod
    def _is_ech_only_cluster(cluster: IncidentCluster, threshold: float) -> bool:
        ech_any = bool(cluster.l5_l6_availability.get("ech_observed_any", False))
        has_rules = any(c.startswith("RULE_") for c in cluster.active_reason_codes) or any(
            flow_is_cr01(f) or flow_is_cr02(f) or flow_is_cr04(f) for f in cluster.flows
        )
        return bool(ech_any and cluster.max_s_total < threshold and not has_rules)

    @staticmethod
    def _is_fingerprint_only_cluster(cluster: IncidentCluster, threshold: float) -> bool:
        has_fp = bool(
            cluster.l5_l6_availability.get("ja4_fingerprints")
            or cluster.l5_l6_availability.get("hassh_fingerprints")
        )
        has_rules = any(c.startswith("RULE_") for c in cluster.active_reason_codes) or any(
            flow_is_cr01(f) or flow_is_cr02(f) or flow_is_cr04(f) for f in cluster.flows
        )
        max_new_edge = cluster.metrics_summary.get("max_new_edge_score")
        has_new_edge = max_new_edge is not None and float(max_new_edge) >= 0.70
        return bool(has_fp and cluster.max_s_total < threshold and not has_rules and not has_new_edge)

    def map_cluster(
        self,
        cluster: IncidentCluster,
    ) -> Tuple[List[AttackMapping], Optional[str]]:
        if not cluster.flows:
            return [], "Empty incident cluster contains no observable flow events."

        threshold = max(
            self.default_threshold,
            max(f.threshold for f in cluster.flows),
        )

        # CR-03 & CR-05 & Benign negative guards
        if self._is_ech_only_cluster(cluster, threshold):
            return (
                [],
                "CR-03 guard: ECH (`ech_observed=True`) without corroborating malicious behavioral or rule evidence is treated as a neutral privacy feature and is not mapped to MITRE ATT&CK.",
            )

        if self._is_fingerprint_only_cluster(cluster, threshold):
            return (
                [],
                "CR-05 guard: JA4/HASSH fingerprint novelty alone without corroborating session/endpoint/fusion evidence is never mapped to MITRE ATT&CK.",
            )

        cr01 = bool(
            "RULE_CR01_DEPRECATED_TLS_VERSION" in cluster.active_reason_codes
            or any(flow_is_cr01(f) for f in cluster.flows)
        )
        cr02 = bool(
            "RULE_CR02_CERT_TRUST_ANOMALY" in cluster.active_reason_codes
            or any(flow_is_cr02(f) for f in cluster.flows)
        )
        cr04 = bool(
            "RULE_CR04_WEAK_SSH_KEX" in cluster.active_reason_codes
            or any(flow_is_cr04(f) for f in cluster.flows)
        )

        has_active_rules = bool(
            any(c.startswith("RULE_") for c in cluster.active_reason_codes) or cr01 or cr02 or cr04
        )
        if cluster.max_s_total < threshold and not has_active_rules:
            return (
                [],
                "No analytic or policy threshold exceeded; benign or low-signal metadata traffic is not mapped to MITRE ATT&CK.",
            )

        # Guard against single isolated anomaly score without threshold/rule/classifier/sequence corroboration (SR-04)
        comp = cluster.component_scores_max
        if (
            not has_active_rules
            and cluster.max_s_total < threshold
            and comp.get("classifier", 0.0) < 0.50
            and comp.get("sequence", 0.0) < 0.50
        ):
            return (
                [],
                "SR-04/SR-05 guard: Isolated stochastic anomaly score below calibrated fusion threshold lacks behavioral corroboration for ATT&CK mapping.",
            )

        mappings: List[AttackMapping] = []
        active_set = set(cluster.active_reason_codes)
        metrics = cluster.metrics_summary

        # 1. T1110 — Brute Force (SSH credential churn; excludes unanswered 1-pkt SYN-only probes)
        ssh_attempt_flows = [f for f in cluster.flows if self._flow_is_ssh_attempt(f)]
        ssh_rate = metrics.get("max_session_rate_5m")
        ssh_churn_flows = [
            f
            for f in ssh_attempt_flows
            if (
                "RULE_SSH_BRUTEFORCE_CHURN" in f.active_reason_codes
                and (f.in_pkts >= 2 or f.out_pkts >= 1)
            )
            or (f.flow_duration_ms <= 250 and (f.in_pkts >= 10 or f.retransmitted_pkts >= 5))
            or (f.session_rate_5m is not None and f.session_rate_5m >= 20.0)
        ]
        median_ssh_attempt_dur = (
            float(median([f.flow_duration_ms for f in ssh_attempt_flows])) if ssh_attempt_flows else 0.0
        )
        if ssh_attempt_flows and cluster.max_s_total >= threshold and (
            ("RULE_SSH_BRUTEFORCE_CHURN" in active_set and bool(ssh_churn_flows))
            or (len(ssh_attempt_flows) >= 5 and median_ssh_attempt_dur <= 250.0)
            or (ssh_rate is not None and float(ssh_rate) >= 20.0)
        ):
            supp = ssh_churn_flows or ssh_attempt_flows
            obs = ["ssh_auth_or_churn_context"]
            if "RULE_SSH_BRUTEFORCE_CHURN" in active_set:
                obs.append("RULE_SSH_BRUTEFORCE_CHURN")
            if ssh_rate is not None and float(ssh_rate) >= 20.0:
                obs.append(f"session_rate_5m={ssh_rate:.1f}")
            if len(ssh_attempt_flows) >= 5:
                obs.append(f"ssh_session_burst_count={len(ssh_attempt_flows)}")
            obs.append(f"max_s_total={cluster.max_s_total:.3f}")
            mappings.append(
                AttackMapping(
                    technique_id="T1110",
                    technique_name="Brute Force",
                    tactic="Credential Access",
                    confidence=min(0.98, max(0.65, cluster.max_s_total)),
                    rationale=(
                        f"SSH sessions ({len(supp)}) exhibited rapid connection churn / brute-force "
                        f"characteristics corroborated by fusion score {cluster.max_s_total:.3f} "
                        f"and observed signals {', '.join(obs)}."
                    ),
                    observed_signals=obs,
                    event_ids=[f.event_id for f in supp],
                    session_ids=[f.session_id for f in supp],
                    supporting_event_ids=[f.event_id for f in supp],
                )
            )

        # 2. T1021.004 — Remote Services: SSH (Lateral Movement; requires established bidirectional SSH)
        new_edge = metrics.get("max_new_edge_score")
        ssh_established_flows = [f for f in cluster.flows if self._flow_is_established_ssh(f)]
        ssh_est_dst_count = len({f.dst_host_token for f in ssh_established_flows})
        ssh_new_edge_vals = [float(f.new_edge_score) for f in ssh_established_flows if f.new_edge_score is not None]
        ssh_new_edge = max(ssh_new_edge_vals) if ssh_new_edge_vals else None
        ssh_lateral_flows = [
            f
            for f in ssh_established_flows
            if (f.new_edge_score is not None and f.new_edge_score >= 0.70)
            or f.s_total >= threshold
        ]
        if (
            ssh_established_flows
            and cluster.max_s_total >= threshold
            and (
                (ssh_new_edge is not None and ssh_new_edge >= 0.70)
                or (
                    ssh_est_dst_count >= 2
                    and (comp.get("sequence", 0.0) >= 0.50 or comp.get("classifier", 0.0) >= 0.65)
                )
            )
        ):
            supp = ssh_lateral_flows or ssh_established_flows
            obs = ["established_ssh_context", f"max_s_total={cluster.max_s_total:.3f}"]
            if ssh_new_edge is not None and ssh_new_edge >= 0.70:
                obs.append(f"new_edge_score={ssh_new_edge:.2f}")
            if ssh_est_dst_count >= 2:
                obs.append(f"ssh_unique_dst_hosts={ssh_est_dst_count}")
            if comp.get("sequence", 0.0) >= 0.50:
                obs.append(f"sequence_score={comp['sequence']:.3f}")
            mappings.append(
                AttackMapping(
                    technique_id="T1021.004",
                    technique_name="Remote Services: SSH",
                    tactic="Lateral Movement",
                    confidence=min(0.96, max(0.65, cluster.max_s_total)),
                    rationale=(
                        f"Corroborated established SSH lateral movement across {max(1, ssh_est_dst_count)} destination host(s) "
                        f"with new-edge / multi-channel evidence ({', '.join(obs)})."
                    ),
                    observed_signals=obs,
                    event_ids=[f.event_id for f in supp],
                    session_ids=[f.session_id for f in supp],
                    supporting_event_ids=[f.event_id for f in supp],
                )
            )

        # 3. T1557 — Adversary-in-the-Middle (TLS downgrade + cert anomaly / corroboration)
        aitm_flows = [
            f for f in cluster.flows if flow_is_cr01(f) or flow_is_cr02(f)
        ]
        if aitm_flows and (
            (cr01 and cr02)
            or (cr02 and (comp.get("anomaly", 0.0) >= 0.50 or comp.get("sequence", 0.0) >= 0.50 or comp.get("classifier", 0.0) >= 0.50 or cluster.max_s_total >= threshold))
            or (cr01 and new_edge is not None and float(new_edge) >= 0.70 and cluster.max_s_total >= threshold)
        ):
            obs = []
            if cr01:
                obs.append(
                    "RULE_CR01_DEPRECATED_TLS_VERSION"
                    if "RULE_CR01_DEPRECATED_TLS_VERSION" in active_set
                    else "CR-01_deprecated_tls_version"
                )
            if cr02:
                obs.append(
                    "RULE_CR02_CERT_TRUST_ANOMALY"
                    if "RULE_CR02_CERT_TRUST_ANOMALY" in active_set
                    else "CR-02_cert_trust_anomaly"
                )
            if comp.get("anomaly", 0.0) >= 0.50:
                obs.append(f"anomaly_score={comp['anomaly']:.3f}")
            if comp.get("classifier", 0.0) >= 0.50:
                obs.append(f"classifier_score={comp['classifier']:.3f}")
            if comp.get("sequence", 0.0) >= 0.50:
                obs.append(f"sequence_score={comp['sequence']:.3f}")
            if new_edge is not None and float(new_edge) >= 0.70:
                obs.append(f"new_edge_score={float(new_edge):.2f}")
            mappings.append(
                AttackMapping(
                    technique_id="T1557",
                    technique_name="Adversary-in-the-Middle",
                    tactic="Credential Access",
                    confidence=min(0.97, max(0.68, max(cluster.max_s_total, 0.75 if (cr01 and cr02) else 0.68))),
                    rationale=(
                        "Observed TLS protocol downgrade and/or untrusted/substituted certificate "
                        f"corroborated by secondary signals ({', '.join(obs)})."
                    ),
                    observed_signals=obs,
                    event_ids=[f.event_id for f in aitm_flows],
                    session_ids=[f.session_id for f in aitm_flows],
                    supporting_event_ids=[f.event_id for f in aitm_flows],
                )
            )

        # 4. T1046 — Network Service Discovery (L3/L4 SYN scan / multi-port low-response probe)
        scan_flows = [
            f
            for f in cluster.flows
            if "RULE_L4_SYN_SCAN_OR_FLOOD" in f.active_reason_codes
            or (f.out_pkts <= 1 and f.s_total >= threshold)
        ]
        unique_ports = int(metrics.get("unique_dst_ports", 0))
        low_resp_ratio = float(metrics.get("low_response_flow_ratio", 0.0))
        fanout = metrics.get("max_fanout_5m")
        if cluster.max_s_total >= threshold and (
            ("RULE_L4_SYN_SCAN_OR_FLOOD" in active_set and (unique_ports >= 2 or int(metrics.get("syn_only_flow_count", 0)) >= 1))
            or (unique_ports >= 3 and low_resp_ratio >= 0.50)
            or (fanout is not None and float(fanout) >= 15.0)
        ):
            supp = scan_flows or [f for f in cluster.flows if f.s_total >= threshold] or list(cluster.flows)
            obs = []
            if "RULE_L4_SYN_SCAN_OR_FLOOD" in active_set:
                obs.append("RULE_L4_SYN_SCAN_OR_FLOOD")
            if unique_ports >= 2:
                obs.append(f"unique_dst_ports={unique_ports}")
            if low_resp_ratio >= 0.50:
                obs.append(f"low_response_ratio={low_resp_ratio:.2f}")
            if fanout is not None and float(fanout) >= 15.0:
                obs.append(f"fanout_5m={float(fanout):.1f}")
            obs.append(f"max_s_total={cluster.max_s_total:.3f}")
            mappings.append(
                AttackMapping(
                    technique_id="T1046",
                    technique_name="Network Service Discovery",
                    tactic="Discovery",
                    confidence=min(0.95, max(0.62, cluster.max_s_total)),
                    rationale=(
                        f"Host probed {unique_ports} destination port(s) with SYN-only/unanswered "
                        f"discovery behavior ({', '.join(obs)})."
                    ),
                    observed_signals=obs,
                    event_ids=[f.event_id for f in supp],
                    session_ids=[f.session_id for f in supp],
                    supporting_event_ids=[f.event_id for f in supp],
                )
            )

        # 5. T1498 — Network Denial of Service (Retransmit / throughput / packet flood)
        dos_flows = [
            f
            for f in cluster.flows
            if "RULE_L4_HIGH_RETRANSMISSION" in f.active_reason_codes
            or "RULE_L4_HIGH_THROUGHPUT_BURST" in f.active_reason_codes
            or ("RULE_L3_EXTREME_FLOW_ASYMMETRY" in f.active_reason_codes and f.in_pkts >= 100)
        ]
        if cluster.max_s_total >= threshold and dos_flows:
            obs = [
                code
                for code in (
                    "RULE_L4_HIGH_RETRANSMISSION",
                    "RULE_L4_HIGH_THROUGHPUT_BURST",
                    "RULE_L3_EXTREME_FLOW_ASYMMETRY",
                )
                if code in active_set
            ]
            obs.append(f"max_in_pkts={metrics.get('max_in_pkts', 0)}")
            obs.append(f"max_s_total={cluster.max_s_total:.3f}")
            mappings.append(
                AttackMapping(
                    technique_id="T1498",
                    technique_name="Network Denial of Service",
                    tactic="Impact",
                    confidence=min(0.96, max(0.65, cluster.max_s_total)),
                    rationale=(
                        "Sustained flood/retransmission/throughput burst exceeding calibrated "
                        f"fusion threshold ({', '.join(obs)})."
                    ),
                    observed_signals=obs,
                    event_ids=[f.event_id for f in dos_flows],
                    session_ids=[f.session_id for f in dos_flows],
                    supporting_event_ids=[f.event_id for f in dos_flows],
                )
            )

        # 6. T1573 — Encrypted Channel & T1071.001 — Web Protocols (established bidirectional TLS C2 behavioral corroboration only)
        tls_flows = [f for f in cluster.flows if self._flow_is_established_tls(f)]
        tls_alerted = [f for f in tls_flows if f.s_total >= threshold or f.prediction == 1]
        if (
            len(tls_flows) >= 2
            and len(tls_alerted) >= 2
            and cluster.max_s_total >= threshold
            and (
                (comp.get("sequence", 0.0) >= 0.65 and len(tls_flows) >= 3)
                or (
                    comp.get("classifier", 0.0) >= 0.70
                    and comp.get("anomaly", 0.0) >= 0.40
                    and ("RULE_L3_EXTREME_FLOW_ASYMMETRY" in active_set or len(tls_flows) >= 3)
                )
                or (
                    comp.get("classifier", 0.0) >= 0.85
                    and comp.get("sequence", 0.0) >= 0.60
                )
            )
        ):
            obs = [
                f"tls_session_count={len(tls_alerted)}",
                f"sequence_max={comp.get('sequence', 0.0):.3f}",
                f"classifier_max={comp.get('classifier', 0.0):.3f}",
                f"max_s_total={cluster.max_s_total:.3f}",
            ]
            if comp.get("anomaly", 0.0) >= 0.40:
                obs.append(f"anomaly_max={comp['anomaly']:.3f}")
            if "RULE_L3_EXTREME_FLOW_ASYMMETRY" in active_set:
                obs.append("RULE_L3_EXTREME_FLOW_ASYMMETRY")

            mappings.append(
                AttackMapping(
                    technique_id="T1573",
                    technique_name="Encrypted Channel",
                    tactic="Command and Control",
                    confidence=min(0.95, max(0.65, cluster.max_s_total)),
                    rationale=(
                        "Multi-flow TLS sequence/behavioral evidence indicates malicious encrypted C2 "
                        f"communication rather than benign TLS usage ({', '.join(obs)})."
                    ),
                    observed_signals=obs,
                    event_ids=[f.event_id for f in tls_alerted],
                    session_ids=[f.session_id for f in tls_alerted],
                    supporting_event_ids=[f.event_id for f in tls_alerted],
                )
            )

            web_tls_flows = [f for f in tls_alerted if f.dst_port in {443, 8443}]
            if len(web_tls_flows) >= 3 and cluster.max_s_total >= 0.70 and comp.get("sequence", 0.0) >= 0.65:
                web_obs = obs + [f"web_tls_ports={sorted({f.dst_port for f in web_tls_flows})}"]
                mappings.append(
                    AttackMapping(
                        technique_id="T1071.001",
                        technique_name="Application Layer Protocol: Web Protocols",
                        tactic="Command and Control",
                        confidence=min(0.92, max(0.62, cluster.max_s_total - 0.03)),
                        rationale=(
                            "Repeated anomalous HTTPS/TLS sessions on standard web ports (443/8443) "
                            f"corroborated by temporal LSTM sequence and RF classifier ({', '.join(web_obs)})."
                        ),
                        observed_signals=web_obs,
                        event_ids=[f.event_id for f in web_tls_flows],
                        session_ids=[f.session_id for f in web_tls_flows],
                        supporting_event_ids=[f.event_id for f in web_tls_flows],
                    )
                )

        # 7. T1041 — Exfiltration Over C2 Channel
        exfil_flows = [
            f
            for f in cluster.flows
            if "RULE_L3_EXTREME_FLOW_ASYMMETRY" in f.active_reason_codes
            and f.bytes_ratio >= 100.0
            and f.in_bytes >= 50000
        ]
        if (
            exfil_flows
            and cluster.max_s_total >= threshold
            and (comp.get("classifier", 0.0) >= 0.70 or comp.get("sequence", 0.0) >= 0.60)
        ):
            obs = [
                "RULE_L3_EXTREME_FLOW_ASYMMETRY",
                f"max_bytes_ratio={float(metrics.get('max_bytes_ratio', 0.0)):.1f}",
                f"max_s_total={cluster.max_s_total:.3f}",
            ]
            mappings.append(
                AttackMapping(
                    technique_id="T1041",
                    technique_name="Exfiltration Over C2 Channel",
                    tactic="Exfiltration",
                    confidence=min(0.93, max(0.62, cluster.max_s_total)),
                    rationale=(
                        "Extreme outbound/inbound byte asymmetry (>=100x, >=50KB) paired with "
                        f"high classifier/sequence corroboration ({', '.join(obs)})."
                    ),
                    observed_signals=obs,
                    event_ids=[f.event_id for f in exfil_flows],
                    session_ids=[f.session_id for f in exfil_flows],
                    supporting_event_ids=[f.event_id for f in exfil_flows],
                )
            )

        if not mappings:
            if cr01 or cr04 or cr02:
                return (
                    [],
                    "Observed cryptographic policy violation (CR-01/CR-02/CR-04) without corroborated active exploitation or MitM indicators; routed to policy remediation rather than mechanical ATT&CK mapping.",
                )
            return (
                [],
                "Observed fusion/model score exceeds threshold, but available metadata lacks specific behavioral observables for a grounded MITRE ATT&CK technique mapping (SR-05).",
            )

        return mappings, None
