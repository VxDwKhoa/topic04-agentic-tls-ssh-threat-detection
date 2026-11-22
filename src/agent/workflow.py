"""
Deterministic LangGraph-Style Stage 1-5 Workflow & Threat-Hunting Agent (Week 9).

Runs 100% offline with zero external API keys and zero dependency on `langgraph`
or `langchain`. Implements Topic 04 Section 15 Stages 1-5:
  Pre-Stage 0: ScopePolicyGuard
  Stage 1:     Triage
  Stage 2:     Enrich (read-only crypto/policy/L5-L6 availability; CR-01..CR-05)
  Stage 3:     Hypothesis (>= 1 supporting check + >= 1 disconfirming check per hypothesis)
  Stage 4:     Verify (deterministic checks; blocks single-model & narrative-only high severity)
  Stage 5:     Map (evidence-grounded ATT&CK) + Recommend (read-only R0-R4) + Explain
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import pandas as pd

from src.agent.llm_interface import (
    BaseNarrativeGenerator,
    OfflineTemplateNarrativeGenerator,
)
from src.agent.policy_guard import PolicyAuditResult, ScopePolicyGuard
from src.agent.response_recommender import ReadOnlyResponseRecommender
from src.agent.state import (
    AgentInvestigationResult,
    HuntingAgentState,
    Hypothesis,
    HypothesisCheck,
    StageTrace,
    VerificationResult,
)
from src.hunting.correlator import IncidentCorrelator
from src.hunting.evidence_graph import EvidenceGraphBuilder
from src.hunting.mitre_mapper import EvidenceGroundedMitreMapper
from src.hunting.pipeline import ThreatHuntingPipeline, Week8ScoringAdapter
from src.hunting.schemas import (
    DEPRECATED_TLS_VERSIONS,
    UNTRUSTED_CERT_STATUSES,
    WEAK_SSH_KEX_MARKERS,
    FlowEvidence,
    IncidentCluster,
)

NodeFunc = Callable[[HuntingAgentState], Dict[str, Any]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CompiledWorkflowGraph:
    """Executable compiled deterministic state machine."""

    def __init__(
        self,
        nodes: Dict[str, NodeFunc],
        edges: Dict[str, str],
        entry_point: str,
        finish_point: str,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.finish_point = finish_point

    def invoke(self, initial_state: HuntingAgentState) -> HuntingAgentState:
        state: HuntingAgentState = dict(initial_state)  # type: ignore[assignment]
        state.setdefault("stage_traces", [])
        state.setdefault("errors", [])

        current: Optional[str] = self.entry_point
        visited = 0
        max_steps = len(self.nodes) + 2

        while current is not None:
            visited += 1
            if visited > max_steps:
                raise RuntimeError("Cycle detected in DeterministicWorkflowGraph.")
            node_fn = self.nodes[current]
            updates = node_fn(state)
            if updates:
                for k, v in updates.items():
                    state[k] = v  # type: ignore[literal-required]
            if current == self.finish_point:
                break
            current = self.edges.get(current)

        return state


class DeterministicWorkflowGraph:
    """Pure-Python LangGraph-compatible StateGraph builder (no external deps)."""

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeFunc] = {}
        self._edges: Dict[str, str] = {}
        self._entry_point: Optional[str] = None
        self._finish_point: Optional[str] = None

    def add_node(self, name: str, func: NodeFunc) -> "DeterministicWorkflowGraph":
        self._nodes[name] = func
        return self

    def add_edge(self, source: str, target: str) -> "DeterministicWorkflowGraph":
        self._edges[source] = target
        return self

    def set_entry_point(self, name: str) -> "DeterministicWorkflowGraph":
        self._entry_point = name
        return self

    def set_finish_point(self, name: str) -> "DeterministicWorkflowGraph":
        self._finish_point = name
        return self

    def compile(self) -> CompiledWorkflowGraph:
        if self._entry_point is None or self._entry_point not in self._nodes:
            raise ValueError("Workflow entry_point is missing or not registered as a node.")
        if self._finish_point is None or self._finish_point not in self._nodes:
            raise ValueError("Workflow finish_point is missing or not registered as a node.")
        return CompiledWorkflowGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            entry_point=self._entry_point,
            finish_point=self._finish_point,
        )


class ThreatHuntingAgent:
    """
    Topic 04 Week 9 Offline-First Threat-Hunting Agent executing Stages 1-5
    with ScopePolicyGuard, EvidenceGroundedMitreMapper, EvidenceGraphBuilder,
    and ReadOnlyResponseRecommender.
    """

    def __init__(
        self,
        scoring_adapter: Optional[Week8ScoringAdapter] = None,
        policy_guard: Optional[ScopePolicyGuard] = None,
        correlator: Optional[IncidentCorrelator] = None,
        mitre_mapper: Optional[EvidenceGroundedMitreMapper] = None,
        graph_builder: Optional[EvidenceGraphBuilder] = None,
        response_recommender: Optional[ReadOnlyResponseRecommender] = None,
        narrative_generator: Optional[BaseNarrativeGenerator] = None,
    ) -> None:
        self.scoring_adapter = scoring_adapter
        self.policy_guard = policy_guard or ScopePolicyGuard()
        self.correlator = correlator or IncidentCorrelator()
        self.mitre_mapper = mitre_mapper or EvidenceGroundedMitreMapper()
        self.graph_builder = graph_builder or EvidenceGraphBuilder(
            default_detector_versions=(
                scoring_adapter.detector_versions if scoring_adapter is not None else None
            )
        )
        self.pipeline = ThreatHuntingPipeline(
            scoring_adapter=self.scoring_adapter,
            correlator=self.correlator,
            mitre_mapper=self.mitre_mapper,
            graph_builder=self.graph_builder,
        )
        self.response_recommender = response_recommender or ReadOnlyResponseRecommender()
        self.narrative_generator: BaseNarrativeGenerator = (
            narrative_generator or OfflineTemplateNarrativeGenerator()
        )
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> CompiledWorkflowGraph:
        graph = DeterministicWorkflowGraph()
        graph.add_node("stage1_triage", self._node_triage)
        graph.add_node("stage2_enrich", self._node_enrich)
        graph.add_node("stage3_hypothesis", self._node_hypothesis)
        graph.add_node("stage4_verify", self._node_verify)
        graph.add_node("stage5_map_recommend_explain", self._node_map_recommend_explain)

        graph.set_entry_point("stage1_triage")
        graph.add_edge("stage1_triage", "stage2_enrich")
        graph.add_edge("stage2_enrich", "stage3_hypothesis")
        graph.add_edge("stage3_hypothesis", "stage4_verify")
        graph.add_edge("stage4_verify", "stage5_map_recommend_explain")
        graph.set_finish_point("stage5_map_recommend_explain")
        return graph.compile()

    def _node_triage(self, state: HuntingAgentState) -> Dict[str, Any]:
        t0 = _utc_now_iso()
        cluster: IncidentCluster = state["incident_cluster"]
        detector_versions = (
            self.scoring_adapter.detector_versions
            if self.scoring_adapter is not None
            else self.graph_builder.default_detector_versions
        )
        triage_summary = {
            "incident_id": cluster.incident_id,
            "primary_host_token": cluster.primary_host_token,
            "target_host_tokens": cluster.target_host_tokens,
            "window_id": cluster.window_id,
            "session_ids": cluster.session_ids,
            "event_ids": cluster.event_ids,
            "total_flow_count": cluster.total_flow_count,
            "alerted_flow_count": cluster.alerted_flow_count,
            "max_s_total": cluster.max_s_total,
            "mean_s_total": cluster.mean_s_total,
            "dominant_signal": cluster.dominant_signal,
            "component_scores_max": cluster.component_scores_max,
            "weighted_contributions_max": cluster.weighted_contributions_max,
            "active_reason_codes": cluster.active_reason_codes,
            "correlation_keys": cluster.correlation_keys,
            "detector_versions": detector_versions,
        }
        trace = StageTrace(
            stage_name="stage1_triage",
            status="completed",
            started_at_utc=t0,
            completed_at_utc=_utc_now_iso(),
            inputs_summary={"flow_count": cluster.total_flow_count, "window_id": cluster.window_id},
            outputs_summary={
                "incident_id": cluster.incident_id,
                "max_s_total": cluster.max_s_total,
                "dominant_signal": cluster.dominant_signal,
            },
            guardrail_notes=[
                "Correlated by stable session/event IDs, host tokens, and time window (SR-02).",
                "Ground-truth label columns excluded.",
            ],
        )
        traces = list(state.get("stage_traces", [])) + [trace]
        return {"triage_summary": triage_summary, "stage_traces": traces}

    def _node_enrich(self, state: HuntingAgentState) -> Dict[str, Any]:
        t0 = _utc_now_iso()
        cluster: IncidentCluster = state["incident_cluster"]
        l5 = dict(cluster.l5_l6_availability)
        active = set(cluster.active_reason_codes)

        tls_vers = [v.upper() for v in l5.get("tls_versions", [])]
        cert_statuses = [c.lower() for c in l5.get("cert_statuses", [])]
        ssh_kex = [k.lower() for k in l5.get("ssh_kex_classes", [])]

        cr01_deprecated_tls = bool(
            "RULE_CR01_DEPRECATED_TLS_VERSION" in active
            or any(v in DEPRECATED_TLS_VERSIONS for v in tls_vers)
        )
        cr02_cert_anomaly = bool(
            "RULE_CR02_CERT_TRUST_ANOMALY" in active
            or any(s in UNTRUSTED_CERT_STATUSES for s in cert_statuses)
        )
        cr03_ech_present = bool(l5.get("ech_observed_any", False))
        cr04_weak_ssh = bool(
            "RULE_CR04_WEAK_SSH_KEX" in active
            or any(any(w in k for w in WEAK_SSH_KEX_MARKERS) for k in ssh_kex)
        )

        has_ja4 = bool(l5.get("ja4_fingerprints"))
        has_hassh = bool(l5.get("hassh_fingerprints"))
        min_fp_prev = cluster.metrics_summary.get("min_baseline_fp_prevalence")
        cr05_fp_novel = bool((has_ja4 or has_hassh) and (min_fp_prev is None or float(min_fp_prev) < 0.05))

        enrichment = {
            "read_only": True,
            "no_raw_l7_accessed": True,
            "dataset_mode": "zeek_l5_l6_enriched" if l5.get("available") else "l3_l4_netflow_only",
            "l5_l6_availability": l5,
            "crypto_policy_checks": {
                "CR-01_deprecated_tls": cr01_deprecated_tls,
                "CR-02_cert_trust_anomaly": cr02_cert_anomaly,
                "CR-03_ech_observed_neutral": cr03_ech_present,
                "CR-04_weak_ssh_kex": cr04_weak_ssh,
                "CR-05_fingerprint_present": bool(has_ja4 or has_hassh),
                "CR-05_fingerprint_novel": cr05_fp_novel,
            },
            "behavioral_context": dict(cluster.metrics_summary),
            "source_versions": {
                "crypto_policy_profile": "NIST-SP-800-52r2 / RFC-9846 / RFC-9849 / RFC-9142",
                "schema_allowlist": "1.0.0",
            },
        }
        notes = [
            f"L5/L6 availability: available={l5.get('available')}, fields_present={len(l5.get('fields_present', []))}.",
            "Read-only enrichment without raw L7 payload (SR-01, SR-09).",
        ]
        if cr03_ech_present:
            notes.append("CR-03: ech_observed=True recorded as neutral privacy feature; hidden SNI/ALPN not reconstructed.")
        if has_ja4 or has_hassh:
            notes.append("CR-05: Fingerprints recorded as contextual identifiers only, never sole proof of maliciousness.")

        trace = StageTrace(
            stage_name="stage2_enrich",
            status="completed",
            started_at_utc=t0,
            completed_at_utc=_utc_now_iso(),
            inputs_summary={"l5_l6_available": l5.get("available")},
            outputs_summary={
                "dataset_mode": enrichment["dataset_mode"],
                "crypto_policy_checks": enrichment["crypto_policy_checks"],
            },
            guardrail_notes=notes,
        )
        traces = list(state.get("stage_traces", [])) + [trace]
        return {"enrichment": enrichment, "stage_traces": traces}

    def _node_hypothesis(self, state: HuntingAgentState) -> Dict[str, Any]:
        t0 = _utc_now_iso()
        cluster: IncidentCluster = state["incident_cluster"]
        enrichment: Dict[str, Any] = state["enrichment"]
        crypto = enrichment["crypto_policy_checks"]
        metrics = cluster.metrics_summary
        comp = cluster.component_scores_max
        active = set(cluster.active_reason_codes)
        threshold = max(0.45, max((f.threshold for f in cluster.flows), default=0.45))

        hypotheses: List[Hypothesis] = []

        # H1: TLS Downgrade / AiTM when CR-01 or CR-02 observed
        if crypto["CR-01_deprecated_tls"] or crypto["CR-02_cert_trust_anomaly"]:
            corrob = comp.get("anomaly", 0.0) >= 0.50 or comp.get("classifier", 0.0) >= 0.50 or comp.get("sequence", 0.0) >= 0.50
            both = crypto["CR-01_deprecated_tls"] and crypto["CR-02_cert_trust_anomaly"]
            sup = [
                HypothesisCheck(
                    check_name="crypto_downgrade_or_cert_indicator",
                    direction="supporting",
                    passed=True,
                    detail="Observed deprecated TLS version (CR-01) or untrusted certificate status (CR-02).",
                    observed_value={"cr01": crypto["CR-01_deprecated_tls"], "cr02": crypto["CR-02_cert_trust_anomaly"]},
                ),
                HypothesisCheck(
                    check_name="secondary_analytic_corroboration",
                    direction="supporting",
                    passed=both or corrob,
                    detail="Both CR-01+CR-02 or secondary ML/sequence channel corroborates AiTM.",
                    observed_value={"both": both, "max_s_total": cluster.max_s_total},
                ),
            ]
            dis = [
                HypothesisCheck(
                    check_name="planned_maintenance_or_legacy_config_only",
                    direction="disconfirming",
                    passed=not (both or corrob),
                    detail="Isolated legacy TLS or planned cert renewal without MitM/behavioral anomaly favors R4 config remediation over active AiTM.",
                    observed_value={"cert_statuses": enrichment["l5_l6_availability"].get("cert_statuses", [])},
                )
            ]
            outcome: Literal["supported", "weakened", "refuted"] = (
                "supported" if (both or (crypto["CR-02_cert_trust_anomaly"] and corrob)) else "weakened"
            )
            hypotheses.append(
                Hypothesis(
                    hypothesis_id="hyp_tls_downgrade_aitm",
                    title="TLS Downgrade / Adversary-in-the-Middle (T1557)",
                    category="threat",
                    plausibility="high" if outcome == "supported" else "medium",
                    supporting_checks=sup,
                    disconfirming_checks=dis,
                    outcome=outcome,
                    summary="Evaluates whether CR-01/CR-02 indicates active AiTM vs benign config/cert renewal.",
                )
            )

        # H2: Crypto Policy Misconfiguration (CR-01 / CR-02 / CR-04)
        if crypto["CR-01_deprecated_tls"] or crypto["CR-02_cert_trust_anomaly"] or crypto["CR-04_weak_ssh_kex"]:
            has_policy_rule = bool(
                active & {"RULE_CR01_DEPRECATED_TLS_VERSION", "RULE_CR02_CERT_TRUST_ANOMALY", "RULE_CR04_WEAK_SSH_KEX"}
            )
            no_active_exploit = cluster.max_s_total < threshold and not (
                crypto["CR-01_deprecated_tls"] and crypto["CR-02_cert_trust_anomaly"]
            )
            hypotheses.append(
                Hypothesis(
                    hypothesis_id="hyp_crypto_policy_regression",
                    title="Cryptographic Policy Misconfiguration / Regression (CR-01/CR-02/CR-04)",
                    category="policy_misconfiguration",
                    plausibility="high" if no_active_exploit else "medium",
                    supporting_checks=[
                        HypothesisCheck(
                            check_name="policy_rule_triggered",
                            direction="supporting",
                            passed=True,
                            detail=(
                                "Deprecated TLS, weak SSH KEX, or untrusted cert profile detected "
                                "via deterministic rule codes and/or L5/L6 metadata."
                            ),
                            observed_value={
                                "has_policy_rule_code": has_policy_rule,
                                "active_reason_codes": cluster.active_reason_codes,
                                "cr01": crypto["CR-01_deprecated_tls"],
                                "cr02": crypto["CR-02_cert_trust_anomaly"],
                                "cr04": crypto["CR-04_weak_ssh_kex"],
                            },
                        )
                    ],
                    disconfirming_checks=[
                        HypothesisCheck(
                            check_name="active_multi_channel_exploitation_absent",
                            direction="disconfirming",
                            passed=not no_active_exploit,
                            detail="Checks whether high multi-channel ML/sequence attack evidence overrides pure policy classification.",
                            observed_value={"max_s_total": cluster.max_s_total, "threshold": threshold},
                        )
                    ],
                    outcome="supported" if no_active_exploit else "weakened",
                    summary="Distinguishes static cryptographic policy regression (R4) from active compromise.",
                )
            )

        # H3: Encrypted C2 Channel over TLS (requires established bidirectional TLS flows)
        if metrics.get("tls_flow_count", 0) > 0:
            est_tls_count = int(metrics.get("established_tls_flow_count", 0))
            c2_seq = comp.get("sequence", 0.0) >= 0.65 and est_tls_count >= 3
            c2_cls_anom = (
                est_tls_count >= 2
                and comp.get("classifier", 0.0) >= 0.70
                and comp.get("anomaly", 0.0) >= 0.40
                and ("RULE_L3_EXTREME_FLOW_ASYMMETRY" in active or est_tls_count >= 3)
            )
            c2_cls_seq = (
                est_tls_count >= 2
                and comp.get("classifier", 0.0) >= 0.85
                and comp.get("sequence", 0.0) >= 0.60
            )
            c2_supported = bool(
                est_tls_count >= 2
                and cluster.max_s_total >= threshold
                and (c2_seq or c2_cls_anom or c2_cls_seq)
            )
            hypotheses.append(
                Hypothesis(
                    hypothesis_id="hyp_encrypted_c2_channel",
                    title="Encrypted Command-and-Control Channel (T1573)",
                    category="threat",
                    plausibility="high" if c2_supported else "low",
                    supporting_checks=[
                        HypothesisCheck(
                            check_name="multi_flow_tls_behavioral_corroboration",
                            direction="supporting",
                            passed=c2_supported,
                            detail="Requires repeated established bidirectional TLS sessions with LSTM sequence or classifier+anomaly corroboration above threshold.",
                            observed_value={
                                "tls_flow_count": metrics.get("tls_flow_count"),
                                "established_tls_flow_count": est_tls_count,
                                "sequence_max": comp.get("sequence"),
                            },
                        )
                    ],
                    disconfirming_checks=[
                        HypothesisCheck(
                            check_name="benign_tls_ech_or_syn_probe_check",
                            direction="disconfirming",
                            passed=not c2_supported,
                            detail="Tests whether traffic consists of unanswered SYN-only probes on port 443, benign HTTPS/TLS 1.3, or RFC 9849 ECH privacy traffic without established C2 sessions.",
                            observed_value={
                                "established_tls_flow_count": est_tls_count,
                                "syn_only_flow_count": metrics.get("syn_only_flow_count", 0),
                                "ech_observed": crypto["CR-03_ech_observed_neutral"],
                                "max_s_total": cluster.max_s_total,
                            },
                        )
                    ],
                    outcome="supported" if c2_supported else "refuted",
                    summary="Guards against mechanical T1573 mapping on SYN-only port-443 probes or benign TLS/ECH flows.",
                )
            )

        # H4: SSH Brute Force or Lateral Movement (excludes 1-packet SYN-only port-22 probes)
        if metrics.get("ssh_flow_count", 0) > 0:
            est_ssh_count = int(metrics.get("established_ssh_flow_count", 0))
            ssh_attempt_count = int(metrics.get("ssh_attempt_flow_count", 0))
            ssh_bf = bool(
                ssh_attempt_count > 0
                and cluster.max_s_total >= threshold
                and (
                    "RULE_SSH_BRUTEFORCE_CHURN" in active
                    or (
                        ssh_attempt_count >= 5
                        and float(metrics.get("median_ssh_attempt_duration_ms", 0.0)) <= 250.0
                    )
                    or (
                        metrics.get("max_session_rate_5m") is not None
                        and float(metrics["max_session_rate_5m"]) >= 20.0
                    )
                )
            )
            ssh_lat = bool(
                est_ssh_count > 0
                and cluster.max_s_total >= threshold
                and (
                    (
                        metrics.get("max_new_edge_score") is not None
                        and float(metrics["max_new_edge_score"]) >= 0.70
                    )
                    or (
                        int(metrics.get("established_ssh_unique_dst_hosts", 0)) >= 2
                        and (comp.get("sequence", 0.0) >= 0.50 or comp.get("classifier", 0.0) >= 0.65)
                    )
                )
            )
            ssh_sup = ssh_bf or ssh_lat
            hypotheses.append(
                Hypothesis(
                    hypothesis_id="hyp_ssh_abuse_or_lateral",
                    title="SSH Brute-Force / Lateral Movement (T1110 / T1021.004)",
                    category="threat",
                    plausibility="high" if ssh_sup else "low",
                    supporting_checks=[
                        HypothesisCheck(
                            check_name="ssh_churn_or_new_edge_corroboration",
                            direction="supporting",
                            passed=ssh_sup,
                            detail="Checks non-SYN SSH brute-force churn or established SSH lateral new-edge corroboration.",
                            observed_value={
                                "ssh_bf": ssh_bf,
                                "ssh_lat": ssh_lat,
                                "established_ssh_flow_count": est_ssh_count,
                                "ssh_attempt_flow_count": ssh_attempt_count,
                            },
                        )
                    ],
                    disconfirming_checks=[
                        HypothesisCheck(
                            check_name="benign_admin_ssh_syn_probe_or_hassh_only",
                            direction="disconfirming",
                            passed=not ssh_sup,
                            detail="Unanswered port-22 SYN scan probes, normal administrative SSH, or HASSH change alone does not confirm SSH attack (CR-05).",
                            observed_value={
                                "max_s_total": cluster.max_s_total,
                                "established_ssh_flow_count": est_ssh_count,
                                "hassh_count": len(enrichment["l5_l6_availability"].get("hassh_fingerprints", [])),
                            },
                        )
                    ],
                    outcome="supported" if ssh_sup else "refuted",
                    summary="Tests SSH threat hypotheses against SYN-only port-22 probes and normal SSH / HASSH-only baseline.",
                )
            )

        # H5: L4 Scan / DoS Burst
        l4_scan_dos = bool(
            active & {"RULE_L4_SYN_SCAN_OR_FLOOD", "RULE_L4_HIGH_RETRANSMISSION", "RULE_L4_HIGH_THROUGHPUT_BURST"}
            or (metrics.get("unique_dst_ports", 0) >= 3 and metrics.get("low_response_flow_ratio", 0.0) >= 0.50 and cluster.max_s_total >= threshold)
        )
        if l4_scan_dos or cluster.max_s_total >= threshold:
            hypotheses.append(
                Hypothesis(
                    hypothesis_id="hyp_l4_scan_or_dos",
                    title="L3/L4 Service Discovery or Denial-of-Service Burst (T1046 / T1498)",
                    category="threat",
                    plausibility="high" if l4_scan_dos else "low",
                    supporting_checks=[
                        HypothesisCheck(
                            check_name="l4_scan_or_flood_signature",
                            direction="supporting",
                            passed=l4_scan_dos,
                            detail="Checks SYN scan/flood, high retransmission, throughput burst, or multi-port probe.",
                            observed_value={"active_rules": sorted(active), "unique_dst_ports": metrics.get("unique_dst_ports")},
                        )
                    ],
                    disconfirming_checks=[
                        HypothesisCheck(
                            check_name="missing_l4_volumetric_or_scan_signature",
                            direction="disconfirming",
                            passed=not l4_scan_dos,
                            detail="High ML score without L4 scan/flood observables cannot be claimed as T1046/T1498.",
                            observed_value={"max_in_pkts": metrics.get("max_in_pkts"), "max_retransmits": metrics.get("max_retransmits")},
                        )
                    ],
                    outcome="supported" if l4_scan_dos else "weakened",
                    summary="Verifies whether L3/L4 packet/port observables ground scan/DoS claims.",
                )
            )

        # H6: Benign Alternative (ECH Privacy / Fingerprint Drift / Single-Model FP) — ALWAYS evaluated
        has_any_policy_or_rule = bool(
            active
            or crypto["CR-01_deprecated_tls"]
            or crypto["CR-02_cert_trust_anomaly"]
            or crypto["CR-04_weak_ssh_kex"]
        )
        benign_ech_only = crypto["CR-03_ech_observed_neutral"] and cluster.max_s_total < threshold and not has_any_policy_or_rule
        benign_fp_only = crypto["CR-05_fingerprint_present"] and cluster.max_s_total < threshold and not has_any_policy_or_rule
        benign_low_sig = cluster.max_s_total < threshold and not has_any_policy_or_rule
        benign_likely = benign_ech_only or benign_fp_only or benign_low_sig
        hypotheses.append(
            Hypothesis(
                hypothesis_id="hyp_benign_privacy_or_drift",
                title="Benign Traffic / ECH Privacy Feature (CR-03) / Software Fingerprint Drift (CR-05)",
                category="benign_alternative",
                plausibility="high" if benign_likely else "low",
                supporting_checks=[
                    HypothesisCheck(
                        check_name="benign_or_neutral_l5_l6_check",
                        direction="supporting",
                        passed=benign_likely,
                        detail="Fusion score below threshold and zero policy/behavioral rules triggered.",
                        observed_value={
                            "max_s_total": cluster.max_s_total,
                            "ech_observed": crypto["CR-03_ech_observed_neutral"],
                            "fingerprint_present": crypto["CR-05_fingerprint_present"],
                        },
                    )
                ],
                disconfirming_checks=[
                    HypothesisCheck(
                        check_name="corroborated_malicious_or_policy_evidence",
                        direction="disconfirming",
                        passed=not benign_likely,
                        detail="Disconfirms benign hypothesis when multi-channel fusion or deterministic rules fire.",
                        observed_value={"active_reason_codes": cluster.active_reason_codes, "max_s_total": cluster.max_s_total},
                    )
                ],
                outcome="supported" if benign_likely else "refuted",
                summary="Mandatory benign alternative hypothesis preventing ECH/fingerprint false positives.",
            )
        )

        trace = StageTrace(
            stage_name="stage3_hypothesis",
            status="completed",
            started_at_utc=t0,
            completed_at_utc=_utc_now_iso(),
            inputs_summary={"active_reason_codes": cluster.active_reason_codes},
            outputs_summary={
                "hypothesis_count": len(hypotheses),
                "supported_hypotheses": [h.hypothesis_id for h in hypotheses if h.outcome == "supported"],
            },
            guardrail_notes=[
                "Every hypothesis contains >= 1 supporting check and >= 1 disconfirming check (Sec 15 Stage 3)."
            ],
        )
        traces = list(state.get("stage_traces", [])) + [trace]
        return {"hypotheses": hypotheses, "stage_traces": traces}

    def _node_verify(self, state: HuntingAgentState) -> Dict[str, Any]:
        t0 = _utc_now_iso()
        cluster: IncidentCluster = state["incident_cluster"]
        enrichment: Dict[str, Any] = state["enrichment"]
        crypto = enrichment["crypto_policy_checks"]
        comp = cluster.component_scores_max
        active = set(cluster.active_reason_codes)
        metrics = cluster.metrics_summary
        threshold = max(0.45, max((f.threshold for f in cluster.flows), default=0.45))

        cr01 = bool(crypto["CR-01_deprecated_tls"])
        cr02 = bool(crypto["CR-02_cert_trust_anomaly"])
        cr04 = bool(crypto["CR-04_weak_ssh_kex"])
        has_any_policy_or_rule = bool(active or cr01 or cr02 or cr04)

        corroborating: List[str] = []
        if any(c.startswith("RULE_") for c in active) or cr01 or cr02 or cr04:
            corroborating.append("rule")
        if comp.get("classifier", 0.0) >= 0.70:
            corroborating.append("classifier")
        if comp.get("anomaly", 0.0) >= 0.50:
            corroborating.append("anomaly")
        if comp.get("sequence", 0.0) >= 0.65:
            corroborating.append("sequence")
        if (
            (metrics.get("max_new_edge_score") is not None and float(metrics["max_new_edge_score"]) >= 0.70)
            or (metrics.get("max_session_rate_5m") is not None and float(metrics["max_session_rate_5m"]) >= 20.0)
            or (metrics.get("max_fanout_5m") is not None and float(metrics["max_fanout_5m"]) >= 15.0)
        ):
            corroborating.append("behavioral_l5_l6")

        blocked_escalations: List[str] = []

        # Deterministic Guardrail Checks
        ech_only = crypto["CR-03_ech_observed_neutral"] and cluster.max_s_total < threshold and not has_any_policy_or_rule
        fp_only = crypto["CR-05_fingerprint_present"] and cluster.max_s_total < threshold and not has_any_policy_or_rule
        if ech_only:
            blocked_escalations.append("CR-03_ech_observed_treated_as_neutral_privacy_feature")
        if fp_only:
            blocked_escalations.append("CR-05_fingerprint_alone_cannot_prove_maliciousness")

        crypto_policy_signals: List[str] = []
        if cr01:
            crypto_policy_signals.append(
                "RULE_CR01_DEPRECATED_TLS_VERSION"
                if "RULE_CR01_DEPRECATED_TLS_VERSION" in active
                else "CR-01_deprecated_tls"
            )
        if cr02:
            crypto_policy_signals.append(
                "RULE_CR02_CERT_TRUST_ANOMALY"
                if "RULE_CR02_CERT_TRUST_ANOMALY" in active
                else "CR-02_cert_trust_anomaly"
            )
        if cr04:
            crypto_policy_signals.append(
                "RULE_CR04_WEAK_SSH_KEX"
                if "RULE_CR04_WEAK_SSH_KEX" in active
                else "CR-04_weak_ssh_kex"
            )
        # Active AiTM requires CR01+CR02 or CR02 + ML/sequence/threshold corroboration
        aitm_corroborated = bool(
            (cr01 and cr02)
            or (
                cr02
                and (
                    comp.get("anomaly", 0.0) >= 0.50
                    or comp.get("classifier", 0.0) >= 0.50
                    or comp.get("sequence", 0.0) >= 0.50
                    or cluster.max_s_total >= threshold
                )
            )
        )
        non_policy_channels = [c for c in corroborating if c != "rule"]
        policy_anomaly_only = bool(
            crypto_policy_signals
            and not aitm_corroborated
            and not non_policy_channels
            and cluster.max_s_total < threshold
        )

        # SR-04 / Section 8: No high severity from single stochastic model alone or LLM narrative alone
        stochastic_only = (
            len(corroborating) == 1
            and corroborating[0] in {"classifier", "anomaly", "sequence"}
        )
        if stochastic_only:
            blocked_escalations.append(
                f"SR-04_blocked_single_stochastic_high_severity({corroborating[0]})"
            )

        if ech_only or fp_only:
            verified_malicious = False
            severity: Literal["info", "low", "medium", "high", "critical"] = "info"
            confidence = 0.15
            rationale = "Neutral ECH (CR-03) or fingerprint-only (CR-05) context without corroborating threat signals."
        elif policy_anomaly_only:
            verified_malicious = False
            severity = "medium"
            confidence = 0.85
            rationale = f"Deterministic crypto policy violation ({sorted(crypto_policy_signals)}) without active compromise."
        elif (len(corroborating) >= 2 or (cr01 and cr02)) and (cluster.max_s_total >= threshold or aitm_corroborated):
            verified_malicious = True
            if cluster.max_s_total >= 0.90 and len(corroborating) >= 3:
                severity = "critical"
            elif cluster.max_s_total >= 0.70 or len(corroborating) >= 2:
                severity = "high"
            else:
                severity = "medium"
            confidence = float(min(0.99, max(0.70, cluster.max_s_total)))
            rationale = (
                f"Multi-channel deterministic verification passed with {len(corroborating)} "
                f"corroborating channels: {corroborating}."
            )
        elif stochastic_only or cluster.max_s_total >= threshold or active:
            verified_malicious = False
            if cluster.max_s_total < threshold:
                severity = "low"
                confidence = float(min(0.55, max(0.25, cluster.max_s_total)))
            else:
                severity = "low" if corroborating == ["anomaly"] else "medium"
                confidence = float(min(0.69, max(0.35, cluster.max_s_total)))
            rationale = (
                f"Single-model or uncorroborated signal ({corroborating or sorted(active)}); "
                "high severity blocked per SR-04 pending analyst triage."
            )
        else:
            verified_malicious = False
            severity = "info"
            confidence = float(max(0.10, 1.0 - cluster.max_s_total))
            rationale = "All channels within benign baseline; no malicious activity verified."

        data_quality = {
            "l5_l6_available": bool(enrichment["l5_l6_availability"].get("available", False)),
            "sequence_available": bool(metrics.get("has_host_sequence_context", False)),
            "total_flows": cluster.total_flow_count,
            "contradictory_signals": [
                s for s in ("WITHIN_BENIGN_CALIBRATED_BASELINE",)
                if s in cluster.all_reason_codes and cluster.max_s_total >= threshold
            ],
        }
        deterministic_checks = {
            "no_single_stochastic_high_severity": not (stochastic_only and severity in {"high", "critical"}),
            "no_narrative_only_high_severity": True,
            "ech_neutrality_cr03": not (ech_only and verified_malicious),
            "fingerprint_not_sole_proof_cr05": not (fp_only and verified_malicious),
            "data_quality_assessed": True,
        }

        verification = VerificationResult(
            verified_malicious=verified_malicious,
            policy_anomaly_only=policy_anomaly_only,
            severity=severity,
            confidence=confidence,
            corroborating_channels=corroborating,
            deterministic_checks=deterministic_checks,
            data_quality=data_quality,
            blocked_escalations=blocked_escalations,
            rationale=rationale,
        )
        trace = StageTrace(
            stage_name="stage4_verify",
            status="completed",
            started_at_utc=t0,
            completed_at_utc=_utc_now_iso(),
            inputs_summary={"corroborating_channels": corroborating},
            outputs_summary={
                "verified_malicious": verified_malicious,
                "policy_anomaly_only": policy_anomaly_only,
                "severity": severity,
                "blocked_escalations": blocked_escalations,
            },
            guardrail_notes=[
                "Deterministic verifier prevents high severity from single stochastic model or LLM narrative alone (SR-04, Sec 15 Stage 4)."
            ],
        )
        traces = list(state.get("stage_traces", [])) + [trace]
        return {"verification": verification, "stage_traces": traces}

    def _node_map_recommend_explain(self, state: HuntingAgentState) -> Dict[str, Any]:
        t0 = _utc_now_iso()
        cluster: IncidentCluster = state["incident_cluster"]
        verification: VerificationResult = state["verification"]
        environment = str(state.get("environment", "production"))
        asset_criticality = str(state.get("asset_criticality", "medium"))
        detector_versions = state["triage_summary"]["detector_versions"]

        # Stage 5a: Evidence-grounded MITRE ATT&CK Mapping
        crypto_checks = state["enrichment"]["crypto_policy_checks"]
        has_crypto_policy = bool(
            crypto_checks["CR-01_deprecated_tls"]
            or crypto_checks["CR-02_cert_trust_anomaly"]
            or crypto_checks["CR-04_weak_ssh_kex"]
        )
        if (
            verification.deterministic_checks.get("ech_neutrality_cr03")
            and cluster.l5_l6_availability.get("ech_observed_any")
            and not verification.verified_malicious
            and not verification.policy_anomaly_only
            and not cluster.active_reason_codes
            and not has_crypto_policy
        ):
            mappings, unmapped_rationale = (
                [],
                "CR-03 guard: ECH (`ech_observed=True`) without corroborating malicious evidence is not mapped to MITRE ATT&CK.",
            )
        else:
            mappings, unmapped_rationale = self.mitre_mapper.map_cluster(cluster)

        # Stage 5b: Build Serializable Evidence Graph
        evidence_graph = self.graph_builder.build(
            cluster=cluster,
            attack_mappings=mappings,
            unmapped_rationale=unmapped_rationale,
            detector_versions=detector_versions,
        )

        # Stage 5c: Read-only Policy Tier Recommendation (R0-R4)
        policy_rec = self.response_recommender.recommend(
            verification=verification,
            cluster=cluster,
            attack_mappings=mappings,
            environment=environment,
            asset_criticality=asset_criticality,
        )

        # Stage 5d: Sanitized Narrative Generation (cannot override Stage 4 severity)
        raw_ctx = {
            "incident_id": cluster.incident_id,
            "primary_host_token": cluster.primary_host_token,
            "window_id": cluster.window_id,
            "triage_summary": state["triage_summary"],
            "enrichment": state["enrichment"],
            "hypotheses": [h.to_dict() for h in state["hypotheses"]],
            "verification": verification.to_dict(),
            "attack_mappings": [m.to_dict() for m in mappings],
            "unmapped_rationale": unmapped_rationale,
            "policy_recommendation": policy_rec.to_dict(),
        }
        sanitized_ctx, llm_audit = self.policy_guard.validate_and_sanitize_payload(
            raw_ctx, strict=True, strip_ground_truth=True
        )
        narrative_result = self.narrative_generator.generate(sanitized_ctx)
        narrative_result["context_guard_passed"] = llm_audit.passed
        # Enforce immutable deterministic severity regardless of narrative text
        narrative_result["enforced_severity"] = verification.severity
        narrative_result["enforced_verified_malicious"] = verification.verified_malicious

        trace = StageTrace(
            stage_name="stage5_map_recommend_explain",
            status="completed",
            started_at_utc=t0,
            completed_at_utc=_utc_now_iso(),
            inputs_summary={"verified_malicious": verification.verified_malicious, "severity": verification.severity},
            outputs_summary={
                "attack_technique_ids": [m.technique_id for m in mappings],
                "recommended_tier": policy_rec.recommended_tier,
                "human_approval_required": policy_rec.human_approval_required,
                "execution_status": policy_rec.execution_status,
            },
            guardrail_notes=[
                "Evidence-grounded ATT&CK mapping only (SR-05); no mechanical TLS/SSH mapping.",
                "Read-only policy tier attachment only (`PROPOSED_ONLY_NOT_EXECUTED`, SR-07).",
                "Narrative generation pre-guarded by ScopePolicyGuard and cannot escalate severity.",
            ],
        )
        traces = list(state.get("stage_traces", [])) + [trace]
        return {
            "attack_mappings": mappings,
            "unmapped_rationale": unmapped_rationale,
            "evidence_graph": evidence_graph,
            "policy_recommendation": policy_rec,
            "narrative_result": narrative_result,
            "stage_traces": traces,
        }

    def investigate_cluster(
        self,
        cluster: IncidentCluster,
        environment: str = "production",
        asset_criticality: str = "medium",
        policy_audit: Optional[PolicyAuditResult] = None,
    ) -> AgentInvestigationResult:
        ScopePolicyGuard.check_environment_scope(environment, asset_criticality)
        flow_audit = self.policy_guard.validate_flows(cluster.flows, strict=True)
        combined_audit = (policy_audit or flow_audit).to_dict()

        final_state = self.workflow.invoke(
            {
                "incident_cluster": cluster,
                "environment": environment.strip().lower(),
                "asset_criticality": asset_criticality.strip().lower(),
                "policy_audit": combined_audit,
                "stage_traces": [],
                "errors": [],
            }
        )
        return AgentInvestigationResult(
            incident_id=cluster.incident_id,
            evidence_graph=final_state["evidence_graph"],
            triage_summary=final_state["triage_summary"],
            enrichment=final_state["enrichment"],
            hypotheses=final_state["hypotheses"],
            verification=final_state["verification"],
            attack_mappings=final_state["attack_mappings"],
            unmapped_rationale=final_state.get("unmapped_rationale"),
            policy_recommendation=final_state["policy_recommendation"],
            narrative=final_state["narrative_result"],
            stage_traces=final_state["stage_traces"],
            audit_summary=combined_audit,
        )

    def investigate_scored_flows(
        self,
        flows: Sequence[FlowEvidence],
        environment: str = "production",
        asset_criticality: str = "medium",
        include_low_signal_clusters: Optional[bool] = None,
    ) -> List[AgentInvestigationResult]:
        ScopePolicyGuard.check_environment_scope(environment, asset_criticality)
        flow_audit = self.policy_guard.validate_flows(flows, strict=True)
        clusters = self.correlator.correlate(
            flows, include_low_signal_clusters=include_low_signal_clusters
        )
        return [
            self.investigate_cluster(
                cluster,
                environment=environment,
                asset_criticality=asset_criticality,
                policy_audit=flow_audit,
            )
            for cluster in clusters
        ]

    def investigate_dataframe(
        self,
        df: pd.DataFrame,
        environment: str = "production",
        asset_criticality: str = "medium",
        decomposed_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
        include_low_signal_clusters: Optional[bool] = None,
    ) -> List[AgentInvestigationResult]:
        ScopePolicyGuard.check_environment_scope(environment, asset_criticality)
        df_clean, df_audit = self.policy_guard.validate_dataframe(
            df, strip_ground_truth=True, strict=True
        )
        flows = self.pipeline.flows_from_dataframe(
            df_clean, decomposed_evidence=decomposed_evidence
        )
        flow_audit = self.policy_guard.validate_flows(flows, strict=True)
        merged_audit = PolicyAuditResult(
            passed=df_audit.passed and flow_audit.passed,
            violations=df_audit.violations + flow_audit.violations,
            redactions_applied=df_audit.redactions_applied + flow_audit.redactions_applied,
            stripped_ground_truth_columns=df_audit.stripped_ground_truth_columns,
            l5_l6_fields_present=sorted(
                set(df_audit.l5_l6_fields_present) | set(flow_audit.l5_l6_fields_present)
            ),
        )
        clusters = self.correlator.correlate(
            flows, include_low_signal_clusters=include_low_signal_clusters
        )
        return [
            self.investigate_cluster(
                cluster,
                environment=environment,
                asset_criticality=asset_criticality,
                policy_audit=merged_audit,
            )
            for cluster in clusters
        ]
