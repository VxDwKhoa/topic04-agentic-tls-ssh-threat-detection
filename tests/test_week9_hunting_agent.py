"""
Comprehensive unit and integration tests for Topic 04 Week 9 deliverables:
- `src/hunting/` (schemas, IncidentCorrelator, EvidenceGraphBuilder, EvidenceGroundedMitreMapper, Week8ScoringAdapter, ThreatHuntingPipeline)
- `src/agent/` (ScopePolicyGuard, OfflineTemplateNarrativeGenerator, CallableLLMNarrativeAdapter, ReadOnlyResponseRecommender, DeterministicWorkflowGraph, ThreatHuntingAgent)
- End-to-end NF-UQ-NIDS-v2 sample test with Weeks 5-8 checkpoints + validation of `report/week9_incident_evidence_graphs.json`
- Synthetic L5/L6 Zeek metadata fixtures (`tls_version`, `cert_status`, `ja4`, `hassh`, `ech_observed`)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.agent import (
    CallableLLMNarrativeAdapter,
    DeterministicWorkflowGraph,
    OfflineTemplateNarrativeGenerator,
    ReadOnlyResponseRecommender,
    ScopePolicyGuard,
    ScopePolicyViolationError,
    ThreatHuntingAgent,
)
from src.features.data_loader import NFUQDataLoader
from src.hunting import (
    AttackMapping,
    EvidenceGraphBuilder,
    EvidenceGroundedMitreMapper,
    FlowEvidence,
    IncidentCorrelator,
    ThreatHuntingPipeline,
    Week8ScoringAdapter,
)
from src.utils.config import PathResolver

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = Path(r"D:\Crypto\DoAn\Notebook\Checkpoint")
REPORT_PATH = REPO_ROOT / "report" / "week9_incident_evidence_graphs.json"


def _make_prescored_row(
    *,
    idx: int = 0,
    src_host: str = "host_aaa1112223334445",
    dst_host: str = "host_bbb6667778889990",
    window_id: int = 1,
    dst_port: int = 443,
    proto: int = 6,
    in_bytes: int = 1200,
    out_bytes: int = 2400,
    in_pkts: int = 10,
    out_pkts: int = 10,
    flow_duration_ms: int = 300,
    tcp_flags: int = 24,
    retransmitted_pkts: int = 0,
    bytes_ratio: float = 0.5,
    pkts_ratio: float = 1.0,
    s_rule: float = 0.0,
    s_classifier: float = 0.05,
    s_anomaly: float = 0.05,
    s_sequence: float = 0.05,
    s_total: float | None = None,
    threshold: float = 0.45,
    reason_codes: List[str] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    weights = {"rule": 0.02, "classifier": 0.5867, "anomaly": 0.05, "sequence": 0.3433}
    comp = {
        "rule": s_rule,
        "classifier": s_classifier,
        "anomaly": s_anomaly,
        "sequence": s_sequence,
    }
    contribs = {k: weights[k] * comp[k] for k in comp}
    total = float(sum(contribs.values())) if s_total is None else float(s_total)
    dom = max(contribs.items(), key=lambda kv: kv[1])[0]
    row: Dict[str, Any] = {
        "flow_order_id": idx + 1,
        "window_id": window_id,
        "src_host_token": src_host,
        "dst_host_token": dst_host,
        "dst_port": dst_port,
        "proto": proto,
        "in_bytes": in_bytes,
        "out_bytes": out_bytes,
        "in_pkts": in_pkts,
        "out_pkts": out_pkts,
        "flow_duration_ms": flow_duration_ms,
        "tcp_flags": tcp_flags,
        "retransmitted_pkts": retransmitted_pkts,
        "bytes_ratio": bytes_ratio,
        "pkts_ratio": pkts_ratio,
        "src_to_dst_avg_throughput": 10000.0,
        "s_total": total,
        "prediction": int(total >= threshold),
        "threshold": threshold,
        "dominant_signal": dom,
        "component_scores": comp,
        "weighted_contributions": contribs,
        "reason_codes": reason_codes if reason_codes is not None else ["WITHIN_BENIGN_CALIBRATED_BASELINE"],
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# 1. Hunting: Correlator, Evidence Graph Determinism, & GT/Baseline Isolation
# ---------------------------------------------------------------------------


def test_correlator_and_evidence_graph_determinism_and_decomposed_fields() -> None:
    rows = [
        _make_prescored_row(
            idx=0,
            dst_port=80,
            in_pkts=150,
            out_pkts=0,
            tcp_flags=2,
            bytes_ratio=120.0,
            s_rule=0.85,
            s_classifier=0.98,
            s_anomaly=0.65,
            s_sequence=0.95,
            s_total=0.95,
            reason_codes=[
                "RULE_L4_SYN_SCAN_OR_FLOOD",
                "RULE_L3_EXTREME_FLOW_ASYMMETRY",
                "WITHIN_BENIGN_CALIBRATED_BASELINE",
            ],
            # Include GT columns to prove stripping/isolation
            label=1,
            attack_family="DoS",
            attack_mapping="T1498",
        ),
        _make_prescored_row(
            idx=1,
            dst_port=8080,
            in_pkts=120,
            out_pkts=0,
            tcp_flags=2,
            s_rule=0.45,
            s_classifier=0.96,
            s_anomaly=0.55,
            s_sequence=0.92,
            s_total=0.91,
            reason_codes=["RULE_L4_SYN_SCAN_OR_FLOOD"],
            label=1,
            attack_family="DoS",
        ),
    ]
    df = pd.DataFrame(rows)

    pipeline_a = ThreatHuntingPipeline()
    pipeline_b = ThreatHuntingPipeline()

    res_a = pipeline_a.run(df)
    res_b = pipeline_b.run(df)

    assert len(res_a) == 1
    cluster_a, graph_a = res_a[0]
    _, graph_b = res_b[0]

    # Determinism across runs
    assert graph_a.incident_id.startswith("inc_")
    assert graph_a.incident_id == graph_b.incident_id
    assert graph_a.to_json() == graph_b.to_json()

    # Valid JSON with allow_nan=False and no GT leakage
    parsed = json.loads(graph_a.to_json())
    gt_keys = {"label", "attack_family", "attack_label", "attack_layers", "attack_mapping", "ground_truth_source"}
    for node in parsed["nodes"]:
        assert not (gt_keys & set(node["attributes"].keys()))
    assert "WITHIN_BENIGN_CALIBRATED_BASELINE" not in cluster_a.active_reason_codes

    # Decomposed Week 8 payload & graph structure checks
    session_nodes = [n for n in graph_a.nodes if n.node_type == "session"]
    host_nodes = [n for n in graph_a.nodes if n.node_type == "host"]
    signal_nodes = [n for n in graph_a.nodes if n.node_type == "analytic_signal"]
    tech_nodes = [n for n in graph_a.nodes if n.node_type == "attack_technique"]

    assert len(session_nodes) == 2
    assert len(host_nodes) == 2
    assert len(signal_nodes) >= 2
    assert len(tech_nodes) >= 1
    for sn in session_nodes:
        assert "s_total" in sn.attributes
        assert "dominant_signal" in sn.attributes
        assert "component_scores" in sn.attributes
        assert "weighted_contributions" in sn.attributes
        assert "reason_codes" in sn.attributes


def test_correlator_derives_missing_session_and_window_ids() -> None:
    flow = FlowEvidence.from_row_and_fusion(
        {
            "flow_order_id": 42,
            "time_bucket": 7,
            "src_host_token": "host_1111222233334444",
            "dst_host_token": "host_5555666677778888",
            "s_total": 0.80,
            "prediction": 1,
        }
    )
    assert flow.window_id == 7
    assert flow.session_id.startswith("sess_host_1111222233334444_7_42")
    assert flow.event_id.startswith("evt_")


# ---------------------------------------------------------------------------
# 2. Evidence-Grounded MITRE ATT&CK Mapper (Positive & Negative Tests)
# ---------------------------------------------------------------------------


def test_mitre_mapper_grounded_vs_no_mechanical_tls_ssh_mapping() -> None:
    mapper = EvidenceGroundedMitreMapper()
    correlator = IncidentCorrelator(include_low_signal_clusters=True)

    # Negative 1: Benign TLS 1.3 on port 443 must NEVER map mechanically to T1573/T1071.001
    benign_tls = [
        FlowEvidence.from_row_and_fusion(
            _make_prescored_row(
                idx=i,
                dst_port=443,
                tls_version="TLSv1.3",
                ja4="t13d1516h2_8daaf6152771_e5627efa2ab1",
                s_total=0.05,
            )
        )
        for i in range(4)
    ]
    cls_benign_tls = correlator.correlate(benign_tls)[0]
    m_tls, r_tls = mapper.map_cluster(cls_benign_tls)
    assert m_tls == []
    assert r_tls is not None

    # Negative 2: Benign SSH on port 22 with HASSH must NEVER map mechanically to T1110/T1021.004
    benign_ssh = [
        FlowEvidence.from_row_and_fusion(
            _make_prescored_row(
                idx=i,
                dst_port=22,
                hassh_or_ja4ssh="ec7c3c6f1b8e9a0d1234567890abcdef",
                s_total=0.04,
            )
        )
        for i in range(2)
    ]
    cls_benign_ssh = correlator.correlate(benign_ssh)[0]
    m_ssh, r_ssh = mapper.map_cluster(cls_benign_ssh)
    assert m_ssh == []
    assert r_ssh is not None

    # Negative 3 (SR-05): High generic score without specific behavioral signature -> empty + unmapped_rationale
    generic_high = [
        FlowEvidence.from_row_and_fusion(
            _make_prescored_row(
                idx=0,
                dst_port=5555,
                s_classifier=1.0,
                s_anomaly=0.20,
                s_sequence=0.0,
                s_total=0.60,
                reason_codes=["WITHIN_BENIGN_CALIBRATED_BASELINE"],
            )
        )
    ]
    cls_gen = correlator.correlate(generic_high)[0]
    m_gen, r_gen = mapper.map_cluster(cls_gen)
    assert m_gen == []
    assert r_gen is not None and "SR-05" in r_gen

    # Positive: Corroborated SSH brute-force churn -> T1110 with complete grounding fields
    bf_ssh = [
        FlowEvidence.from_row_and_fusion(
            _make_prescored_row(
                idx=i,
                dst_port=22,
                in_pkts=65,
                flow_duration_ms=25,
                hassh="9292b285a080f9b1a1b2c3d4e5f60708",
                s_rule=0.50,
                s_classifier=0.95,
                s_anomaly=0.60,
                s_sequence=0.88,
                s_total=0.90,
                reason_codes=["RULE_SSH_BRUTEFORCE_CHURN"],
            )
        )
        for i in range(5)
    ]
    cls_bf = correlator.correlate(bf_ssh)[0]
    m_bf, r_bf = mapper.map_cluster(cls_bf)
    assert r_bf is None
    assert any(m.technique_id == "T1110" for m in m_bf)
    for mapping in m_bf:
        assert isinstance(mapping, AttackMapping)
        assert mapping.technique_id
        assert mapping.rationale
        assert len(mapping.observed_signals) >= 1
        assert len(mapping.event_ids) >= 1
        assert len(mapping.session_ids) >= 1
        assert len(mapping.supporting_event_ids) >= 1


# ---------------------------------------------------------------------------
# 3. ScopePolicyGuard (No-L7, No-Raw-IP, No-Secrets, L5/L6 Allowlist)
# ---------------------------------------------------------------------------


def test_scope_policy_guard_rejects_l7_raw_ip_secrets_and_allows_l5_l6() -> None:
    guard = ScopePolicyGuard()

    # Valid L5/L6 Zeek metadata DataFrame passes cleanly & strips GT labels
    valid_df = pd.DataFrame(
        [
            _make_prescored_row(
                tls_version="TLSv1.3",
                cert_status="valid",
                ja4="t13d1516h2_8daaf6152771_e5627efa2ab1",
                hassh_or_ja4ssh="ec7c3c6f1b8e9a0d1234567890abcdef",
                ech_observed=True,
                ssh_kex_class="curve25519-sha256",
                attack_family="Benign",
                label=0,
            )
        ]
    )
    cleaned_df, audit = guard.validate_dataframe(valid_df, strip_ground_truth=True, strict=True)
    assert audit.passed is True
    assert "attack_family" not in cleaned_df.columns
    assert "label" not in cleaned_df.columns
    assert "tls_version" in audit.l5_l6_fields_present
    assert "ech_observed" in audit.l5_l6_fields_present

    # Prohibited raw L7 columns rejected
    for bad_col in ("http_body", "http_request_uri", "ssh_commands", "decrypted_tls_payload", "tls_session_keys", "raw_sni"):
        bad_df = valid_df.drop(columns=["attack_family", "label"]).copy()
        bad_df[bad_col] = "secret_value"
        with pytest.raises(ScopePolicyViolationError):
            guard.validate_dataframe(bad_df, strict=True)

    # Raw IPv4 in src_host_token rejected
    raw_ip_df = valid_df.drop(columns=["attack_family", "label"]).copy()
    raw_ip_df["src_host_token"] = "192.168.1.42"
    with pytest.raises(ScopePolicyViolationError):
        guard.validate_dataframe(raw_ip_df, strict=True)

    # Secret / key material rejected in LLM payload guard (CR-06)
    secret_payload = {
        "incident_id": "inc_1234",
        "primary_host_token": "host_aaa1112223334445",
        "note": "CLIENT_RANDOM aabbccddeeff00112233445566778899 deadbeef",
    }
    with pytest.raises(ScopePolicyViolationError):
        guard.validate_and_sanitize_payload(secret_payload, strict=True)

    # Unknown environment rejected (DENY_BY_DEFAULT / SR-06)
    with pytest.raises(ScopePolicyViolationError):
        ScopePolicyGuard.check_environment_scope("internet_prod_unknown", "high")


# ---------------------------------------------------------------------------
# 4. Offline Stages 1-5 Workflow, Hypothesis Checks, Stage 4 Guards, & LLM Hook
# ---------------------------------------------------------------------------


def test_offline_workflow_hypothesis_checks_stage4_guards_and_pluggable_llm() -> None:
    # Verify pure-Python workflow graph works cleanly offline
    agent = ThreatHuntingAgent()

    # 4a. Single stochastic model alone (only IF anomaly=0.92, rule=0, classifier=0.1, sequence=0.0) -> blocked from high severity
    single_if_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=0,
                dst_port=443,
                s_rule=0.0,
                s_classifier=0.10,
                s_anomaly=0.92,
                s_sequence=0.0,
                s_total=0.50,
                reason_codes=["ANOMALY_SCORE_ABOVE_CALIBRATED_FPR_0.050"],
            )
        ]
    )
    results_if = agent.investigate_dataframe(single_if_df, environment="production", asset_criticality="high")
    assert len(results_if) == 1
    r_if = results_if[0]
    assert len(r_if.stage_traces) == 5
    assert [t.stage_name for t in r_if.stage_traces] == [
        "stage1_triage",
        "stage2_enrich",
        "stage3_hypothesis",
        "stage4_verify",
        "stage5_map_recommend_explain",
    ]
    for hyp in r_if.hypotheses:
        assert len(hyp.supporting_checks) >= 1
        assert len(hyp.disconfirming_checks) >= 1
    assert r_if.verification.verified_malicious is False
    assert r_if.verification.severity not in {"high", "critical"}
    assert r_if.verification.deterministic_checks["no_single_stochastic_high_severity"] is True
    assert r_if.policy_recommendation.recommended_tier == "R1"
    assert r_if.policy_recommendation.execution_status == "PROPOSED_ONLY_NOT_EXECUTED"

    # 4b. Adversarial mock LLM narrative claiming CRITICAL intrusion cannot raise Stage 4 severity
    captured_contexts: List[Dict[str, Any]] = []

    def mock_llm(ctx: Dict[str, Any]) -> str:
        captured_contexts.append(ctx)
        return "CRITICAL APT INTRUSION DETECTED: isolate host immediately!"

    adapter = CallableLLMNarrativeAdapter(llm_callable=mock_llm)
    agent_with_llm = ThreatHuntingAgent(narrative_generator=adapter)
    benign_df = pd.DataFrame(
        [_make_prescored_row(idx=0, s_total=0.04, attack_family="Benign", label=0)]
    )
    res_adv = agent_with_llm.investigate_dataframe(
        benign_df,
        environment="production",
        asset_criticality="high",
        include_low_signal_clusters=True,
    )
    assert len(res_adv) == 1
    adv = res_adv[0]
    assert len(captured_contexts) == 1
    assert "attack_family" not in json.dumps(captured_contexts[0])
    assert adv.verification.verified_malicious is False
    assert adv.verification.severity == "info"
    assert adv.narrative["enforced_severity"] == "info"
    assert adv.narrative["severity_override_blocked"] is True
    assert adv.policy_recommendation.recommended_tier == "R0"

    # Disabled callable adapter falls back deterministically offline
    disabled_adapter = CallableLLMNarrativeAdapter(llm_callable=None)
    out_offline = disabled_adapter.generate({"incident_id": "inc_test", "primary_host_token": "host_1"})
    assert out_offline["offline"] is True
    assert out_offline["llm_enabled"] is False


# ---------------------------------------------------------------------------
# 5. Synthetic L5/L6 Zeek Fixtures & Response Tier Policy Tests (CR-01..CR-05, R0..R4)
# ---------------------------------------------------------------------------


def test_synthetic_zeek_l5_l6_rules_and_policy_tiers() -> None:
    agent = ThreatHuntingAgent()

    # 5a. ech_observed=True alone -> benign/neutral (CR-03), R0 Observe, no ATT&CK mapping
    ech_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=i,
                dst_port=443,
                tls_version="TLSv1.3",
                ech_observed=True,
                cert_status="valid",
                s_total=0.05,
            )
            for i in range(3)
        ]
    )
    res_ech = agent.investigate_dataframe(
        ech_df, environment="production", asset_criticality="high", include_low_signal_clusters=True
    )[0]
    assert res_ech.enrichment["l5_l6_availability"]["available"] is True
    assert res_ech.enrichment["crypto_policy_checks"]["CR-03_ech_observed_neutral"] is True
    assert res_ech.verification.verified_malicious is False
    assert res_ech.verification.severity == "info"
    assert res_ech.attack_mappings == []
    assert res_ech.policy_recommendation.recommended_tier == "R0"

    # 5b. Novel ja4/hassh fingerprint alone -> never high severity / malicious (CR-05), R0 Observe
    fp_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=0,
                dst_port=443,
                tls_version="TLSv1.3",
                ja4="t13d9999h2_112233445566_aabbccddeeff",
                cert_status="valid",
                baseline_fp_prevalence=0.001,
                s_total=0.12,
            ),
            _make_prescored_row(
                idx=1,
                dst_port=22,
                hassh="deadbeefcafebabe1234567890abcdef",
                baseline_fp_prevalence=0.002,
                s_total=0.10,
            ),
        ]
    )
    res_fp = agent.investigate_dataframe(
        fp_df, environment="production", asset_criticality="medium", include_low_signal_clusters=True
    )[0]
    assert res_fp.enrichment["crypto_policy_checks"]["CR-05_fingerprint_present"] is True
    assert res_fp.verification.verified_malicious is False
    assert res_fp.verification.severity == "info"
    assert res_fp.attack_mappings == []
    assert res_fp.policy_recommendation.recommended_tier == "R0"

    # 5c. CR-01 / CR-02 / CR-04 policy anomaly -> R4 Crypto Policy Remediation (always human approval)
    cr01_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=0,
                dst_port=443,
                tls_version="TLSv1.0",
                cert_status="valid",
                s_rule=0.60,
                s_classifier=0.05,
                s_anomaly=0.05,
                s_sequence=0.0,
                s_total=0.04,
                reason_codes=["RULE_CR01_DEPRECATED_TLS_VERSION"],
            )
        ]
    )
    res_cr01 = agent.investigate_dataframe(cr01_df, environment="lab", asset_criticality="low")[0]
    assert res_cr01.verification.policy_anomaly_only is True
    assert res_cr01.policy_recommendation.recommended_tier == "R4"
    assert res_cr01.policy_recommendation.human_approval_required is True
    assert res_cr01.policy_recommendation.automatic_production_merge_allowed is False
    assert res_cr01.policy_recommendation.retest_after_action is True
    assert res_cr01.policy_recommendation.execution_status == "PROPOSED_ONLY_NOT_EXECUTED"

    cr04_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=0,
                dst_port=22,
                hassh="11223344556677889900aabbccddeeff",
                ssh_kex_class="diffie-hellman-group1-sha1",
                s_rule=0.55,
                s_classifier=0.05,
                s_anomaly=0.05,
                s_sequence=0.0,
                s_total=0.04,
                reason_codes=["RULE_CR04_WEAK_SSH_KEX"],
            )
        ]
    )
    res_cr04 = agent.investigate_dataframe(cr04_df, environment="production", asset_criticality="high")[0]
    assert res_cr04.policy_recommendation.recommended_tier == "R4"
    assert res_cr04.policy_recommendation.human_approval_required is True

    # 5d. Verified multi-channel AiTM (CR-01 + CR-02 + anomaly) -> T1557 + R3 (human approval in production)
    aitm_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=0,
                dst_port=443,
                tls_version="TLSv1.0",
                cert_status="self_signed",
                cert_spki_token="spki_rogue_001",
                ja4="t10d1210h1_111111111111_222222222222",
                s_rule=1.0,
                s_classifier=0.88,
                s_anomaly=0.72,
                s_sequence=0.80,
                s_total=0.85,
                reason_codes=["RULE_CR01_DEPRECATED_TLS_VERSION", "RULE_CR02_CERT_TRUST_ANOMALY"],
            )
        ]
    )
    res_aitm_prod = agent.investigate_dataframe(aitm_df, environment="production", asset_criticality="high")[0]
    assert res_aitm_prod.verification.verified_malicious is True
    assert any(m.technique_id == "T1557" for m in res_aitm_prod.attack_mappings)
    assert res_aitm_prod.policy_recommendation.recommended_tier == "R3"
    assert res_aitm_prod.policy_recommendation.human_approval_required is True
    assert "propose_host_isolation" in res_aitm_prod.policy_recommendation.proposed_actions
    assert res_aitm_prod.policy_recommendation.execution_status == "PROPOSED_ONLY_NOT_EXECUTED"

    res_aitm_lab = agent.investigate_dataframe(aitm_df, environment="lab", asset_criticality="medium")[0]
    assert res_aitm_lab.policy_recommendation.recommended_tier == "R3"
    assert "bounded_host_isolation_lab" in res_aitm_lab.policy_recommendation.proposed_actions

    # 5e. Bounded scan (T1046) -> R2 in lab, constrained to R1 in production (DENY_BY_DEFAULT)
    scan_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=i,
                dst_port=20 + i,
                in_pkts=2,
                out_pkts=0,
                tcp_flags=2,
                s_rule=0.45,
                s_classifier=0.75,
                s_anomaly=0.40,
                s_sequence=0.70,
                s_total=0.72,
                reason_codes=["RULE_L4_SYN_SCAN_OR_FLOOD"],
            )
            for i in range(4)
        ]
    )
    res_scan_lab = agent.investigate_dataframe(scan_df, environment="lab", asset_criticality="medium")[0]
    assert any(m.technique_id == "T1046" for m in res_scan_lab.attack_mappings)
    assert res_scan_lab.policy_recommendation.recommended_tier == "R2"
    assert res_scan_lab.policy_recommendation.auto_expire_minutes == 30
    assert res_scan_lab.policy_recommendation.rollback_required is True

    res_scan_prod = agent.investigate_dataframe(scan_df, environment="production", asset_criticality="medium")[0]
    assert res_scan_prod.policy_recommendation.recommended_tier == "R1"
    assert res_scan_prod.policy_recommendation.network_change_allowed is False


# ---------------------------------------------------------------------------
# 6. End-to-End NF-UQ-NIDS-v2 Checkpoints + Report Validation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not CHECKPOINT_ROOT.is_dir(),
    reason=f"External Weeks 5-8 checkpoints not available at {CHECKPOINT_ROOT}",
)
def test_end_to_end_nfuq_sample_with_trained_checkpoints_and_report() -> None:
    paths = PathResolver.resolve(prefer_sample=True)
    df_test = NFUQDataLoader(paths).load_tabular_splits()["test"]

    adapter = Week8ScoringAdapter.from_checkpoint_root(CHECKPOINT_ROOT, device="cpu")
    agent = ThreatHuntingAgent(scoring_adapter=adapter)
    results = agent.investigate_dataframe(
        df_test,
        environment="production",
        asset_criticality="high",
        include_low_signal_clusters=True,
    )

    assert len(results) >= 5
    # Verify L3/L4-only mode cleanly recorded without error (SR-09)
    assert all(r.enrichment["l5_l6_availability"]["available"] is False for r in results)
    assert all(r.policy_recommendation.execution_status == "PROPOSED_ONLY_NOT_EXECUTED" for r in results)

    # Verify at least one verified malicious incident and at least one grounded ATT&CK mapping
    assert any(r.verification.verified_malicious for r in results)
    assert any(len(r.attack_mappings) >= 1 for r in results)

    # Verify report/week9_incident_evidence_graphs.json exists and passes schema & privacy checks
    assert REPORT_PATH.is_file(), f"Expected Week 9 report at {REPORT_PATH}"
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["incident_count"] >= 3
    assert len(report["incidents"]) == report["incident_count"]
    raw_text = json.dumps(report)
    assert not re.search(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", raw_text)
    for forbidden in ("http_body", "ssh_commands", "decrypted_tls_payload", "tls_session_keys"):
        assert forbidden not in raw_text

    gt_keys = {"label", "attack_family", "attack_label", "attack_layers", "attack_mapping", "ground_truth_source"}
    for item in report["incidents"]:
        eg = item["evidence_graph"]
        for node in eg["nodes"]:
            assert not (gt_keys & set(node["attributes"].keys()))
        assert eg["incident_id"].startswith("inc_")
        assert len(eg["nodes"]) >= 3
        assert len(eg["edges"]) >= 2
        for m in item["attack_mappings"]:
            assert m["technique_id"]
            assert m["rationale"]
            assert len(m["observed_signals"]) >= 1
            assert len(m["event_ids"]) >= 1
            assert len(m["session_ids"]) >= 1
        rec = item["policy_recommendation"]
        assert rec["execution_status"] == "PROPOSED_ONLY_NOT_EXECUTED"
        if rec["recommended_tier"] in {"R3", "R4"}:
            assert rec["human_approval_required"] is True

    by_id = {item["incident_id"]: item for item in report["incidents"]}
    assert [m["technique_id"] for m in by_id["inc_5ef5e3cb6382bba5"]["attack_mappings"]] == ["T1046"]
    assert set(m["technique_id"] for m in by_id["inc_61907afa7a014ca9"]["attack_mappings"]) >= {
        "T1573",
        "T1071.001",
    }


# ---------------------------------------------------------------------------
# 7. Wave 2 Regression Tests (SYN-Probe Guard, NaN Mapping, CR-01/02/04 Alignment, Guard Bypasses)
# ---------------------------------------------------------------------------


def test_syn_only_443_and_22_probes_never_map_tls_ssh_and_positive_techniques_map() -> None:
    agent = ThreatHuntingAgent()
    mapper = EvidenceGroundedMitreMapper()
    correlator = IncidentCorrelator()

    # 7a. Unanswered 1-pkt SYN-only probes on port 443 -> T1046 ONLY (never T1573 / T1071.001)
    syn_443_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=i,
                dst_port=443 if i < 4 else 80,
                in_bytes=44,
                out_bytes=0,
                in_pkts=1,
                out_pkts=0,
                flow_duration_ms=0,
                tcp_flags=2,
                s_rule=0.45,
                s_classifier=0.98,
                s_anomaly=0.60,
                s_sequence=0.95,
                s_total=0.95,
                reason_codes=["RULE_L4_SYN_SCAN_OR_FLOOD"],
            )
            for i in range(5)
        ]
    )
    res_syn443 = agent.investigate_dataframe(syn_443_df, environment="production", asset_criticality="high")[0]
    tech_443 = {m.technique_id for m in res_syn443.attack_mappings}
    assert tech_443 == {"T1046"}
    hyp_c2 = next(h for h in res_syn443.hypotheses if h.hypothesis_id == "hyp_encrypted_c2_channel")
    assert hyp_c2.outcome == "refuted"

    # 7b. Unanswered 1-pkt SYN-only probes on port 22 -> T1046 ONLY (never T1110 / T1021.004)
    syn_22_df = pd.DataFrame(
        [
            _make_prescored_row(
                idx=i,
                dst_host=f"host_dst_{i:04d}",
                dst_port=22 if i < 5 else 80,
                in_bytes=44,
                out_bytes=0,
                in_pkts=1,
                out_pkts=0,
                flow_duration_ms=0,
                tcp_flags=2,
                new_edge_score=0.95,
                session_rate_5m=50.0,
                s_rule=0.45,
                s_classifier=0.95,
                s_anomaly=0.60,
                s_sequence=0.90,
                s_total=0.92,
                reason_codes=["RULE_L4_SYN_SCAN_OR_FLOOD"],
            )
            for i in range(6)
        ]
    )
    res_syn22 = agent.investigate_dataframe(syn_22_df, environment="production", asset_criticality="high")[0]
    tech_22 = {m.technique_id for m in res_syn22.attack_mappings}
    assert "T1110" not in tech_22
    assert "T1021.004" not in tech_22
    assert "T1046" in tech_22
    hyp_ssh = next(h for h in res_syn22.hypotheses if h.hypothesis_id == "hyp_ssh_abuse_or_lateral")
    assert hyp_ssh.outcome == "refuted"

    # 7c. Positive established bidirectional TLS C2 -> T1573 + T1071.001
    est_tls_flows = [
        FlowEvidence.from_row_and_fusion(
            _make_prescored_row(
                idx=i,
                dst_port=443,
                in_bytes=1800,
                out_bytes=4200,
                in_pkts=10,
                out_pkts=11,
                flow_duration_ms=450,
                tcp_flags=31,
                s_rule=0.0,
                s_classifier=0.92,
                s_anomaly=0.55,
                s_sequence=0.88,
                s_total=0.85,
                reason_codes=["ANOMALY_SCORE_ABOVE_CALIBRATED_FPR_0.050"],
            )
        )
        for i in range(3)
    ]
    m_tls, r_tls = mapper.map_cluster(correlator.correlate(est_tls_flows)[0])
    assert r_tls is None
    assert {"T1573", "T1071.001"} <= {m.technique_id for m in m_tls}

    # 7d. Positive established SSH lateral movement -> T1021.004
    est_ssh_flows = [
        FlowEvidence.from_row_and_fusion(
            _make_prescored_row(
                idx=i,
                dst_host=f"host_lat_{i:04d}",
                dst_port=22,
                in_bytes=900,
                out_bytes=1600,
                in_pkts=8,
                out_pkts=9,
                tcp_flags=24,
                new_edge_score=0.90,
                s_rule=0.0,
                s_classifier=0.80,
                s_anomaly=0.55,
                s_sequence=0.72,
                s_total=0.75,
                reason_codes=[],
            )
        )
        for i in range(2)
    ]
    m_lat, _ = mapper.map_cluster(correlator.correlate(est_ssh_flows)[0])
    assert any(m.technique_id == "T1021.004" for m in m_lat)

    # 7e. Positive T1498 (DoS burst) and T1041 (Exfiltration over C2)
    dos_exfil_flows = [
        FlowEvidence.from_row_and_fusion(
            _make_prescored_row(
                idx=0,
                dst_port=443,
                in_bytes=150000,
                out_bytes=100,
                in_pkts=220,
                out_pkts=5,
                retransmitted_pkts=35,
                bytes_ratio=1500.0,
                s_rule=0.80,
                s_classifier=0.92,
                s_anomaly=0.65,
                s_sequence=0.85,
                s_total=0.90,
                reason_codes=["RULE_L4_HIGH_RETRANSMISSION", "RULE_L3_EXTREME_FLOW_ASYMMETRY"],
            )
        )
    ]
    m_de, _ = mapper.map_cluster(correlator.correlate(dos_exfil_flows)[0])
    assert {"T1498", "T1041"} <= {m.technique_id for m in m_de}


def test_nan_inf_sanitized_in_mapping_fusion_payloads() -> None:
    # Omitted s_total with NaN/Inf in weighted_contributions must stay 0.0 (never clamp nan -> 1.0)
    flow_nan = FlowEvidence.from_row_and_fusion(
        {
            "src_host_token": "host_nan_1",
            "dst_host_token": "host_nan_2",
            "component_scores": {"rule": 0.0, "classifier": float("nan"), "anomaly": float("inf"), "sequence": 0.0},
            "weighted_contributions": {"rule": 0.0, "classifier": float("nan"), "anomaly": float("-inf"), "sequence": 0.0},
            "threshold": 0.45,
        }
    )
    assert flow_nan.s_total == 0.0
    assert flow_nan.prediction == 0
    assert all(v == 0.0 for v in flow_nan.component_scores.values())
    assert all(v == 0.0 for v in flow_nan.weighted_contributions.values())

    # Multi-flow cluster with NaN in flow 0 must NOT poison comp_scores_max or drop Stage 4 channels
    flow_0 = FlowEvidence.from_row_and_fusion(
        {
            "flow_order_id": 1,
            "src_host_token": "host_nan_multi",
            "dst_host_token": "host_dst_multi",
            "component_scores": {"rule": float("nan"), "classifier": 0.90, "anomaly": float("nan"), "sequence": 0.80},
            "weighted_contributions": {"rule": 0.0, "classifier": 0.52, "anomaly": 0.0, "sequence": 0.27},
            "s_total": 0.79,
            "threshold": 0.45,
        }
    )
    flow_1 = FlowEvidence.from_row_and_fusion(
        {
            "flow_order_id": 2,
            "src_host_token": "host_nan_multi",
            "dst_host_token": "host_dst_multi",
            "component_scores": {"rule": 0.50, "classifier": 0.90, "anomaly": 0.85, "sequence": 0.80},
            "weighted_contributions": {"rule": 0.01, "classifier": 0.52, "anomaly": 0.04, "sequence": 0.27},
            "s_total": 0.84,
            "threshold": 0.45,
        }
    )
    agent = ThreatHuntingAgent()
    res = agent.investigate_scored_flows([flow_0, flow_1], environment="production", asset_criticality="high")[0]
    assert res.triage_summary["component_scores_max"] == {
        "rule": 0.5,
        "classifier": 0.9,
        "anomaly": 0.85,
        "sequence": 0.8,
    }
    assert "anomaly" in res.verification.corroborating_channels
    json.loads(res.to_json())


def test_l5_l6_cr01_cr02_cr04_consistent_when_reason_codes_empty() -> None:
    agent = ThreatHuntingAgent()

    # CR-01 + CR-02 with reason_codes=[] -> Stages 2-5 agree on T1557 + R3
    aitm_no_rc = pd.DataFrame(
        [
            _make_prescored_row(
                idx=0,
                dst_port=443,
                tls_version="TLSv1.0",
                cert_status="self_signed",
                s_rule=0.0,
                s_classifier=0.05,
                s_anomaly=0.05,
                s_sequence=0.05,
                s_total=0.75,
                reason_codes=[],
            )
        ]
    )
    res_aitm = agent.investigate_dataframe(aitm_no_rc, environment="production", asset_criticality="high")[0]
    assert res_aitm.enrichment["crypto_policy_checks"]["CR-01_deprecated_tls"] is True
    assert res_aitm.enrichment["crypto_policy_checks"]["CR-02_cert_trust_anomaly"] is True
    assert res_aitm.verification.verified_malicious is True
    assert any(m.technique_id == "T1557" for m in res_aitm.attack_mappings)
    assert res_aitm.policy_recommendation.recommended_tier == "R3"

    # Low-score TLSv1.0 with valid cert and reason_codes=[] -> policy_anomaly_only=True & R4
    cr01_no_rc = pd.DataFrame(
        [
            _make_prescored_row(
                idx=0,
                dst_port=443,
                tls_version="TLSv1.0",
                cert_status="valid",
                s_rule=0.0,
                s_classifier=0.05,
                s_anomaly=0.05,
                s_sequence=0.05,
                s_total=0.04,
                reason_codes=[],
            )
        ]
    )
    res_cr01 = agent.investigate_dataframe(
        cr01_no_rc, environment="production", asset_criticality="medium", include_low_signal_clusters=True
    )[0]
    assert res_cr01.verification.policy_anomaly_only is True
    assert res_cr01.policy_recommendation.recommended_tier == "R4"
    assert res_cr01.policy_recommendation.human_approval_required is True

    # Low-score TLSv1.0 with reason_codes=[] is also surfaced by default (include_low_signal_clusters=False)
    res_cr01_default = agent.investigate_dataframe(
        cr01_no_rc, environment="production", asset_criticality="medium"
    )
    assert len(res_cr01_default) == 1
    assert res_cr01_default[0].verification.policy_anomaly_only is True
    assert res_cr01_default[0].policy_recommendation.recommended_tier == "R4"
    assert res_cr01_default[0].policy_recommendation.human_approval_required is True

    # Low-score weak SSH KEX with reason_codes=[] -> policy_anomaly_only=True & R4
    cr04_no_rc = pd.DataFrame(
        [
            _make_prescored_row(
                idx=0,
                dst_port=22,
                ssh_kex_class="diffie-hellman-group1-sha1",
                s_rule=0.0,
                s_classifier=0.05,
                s_anomaly=0.05,
                s_sequence=0.05,
                s_total=0.04,
                reason_codes=[],
            )
        ]
    )
    res_cr04 = agent.investigate_dataframe(
        cr04_no_rc, environment="production", asset_criticality="medium", include_low_signal_clusters=True
    )[0]
    assert res_cr04.verification.policy_anomaly_only is True
    assert res_cr04.policy_recommendation.recommended_tier == "R4"


def test_scope_policy_guard_bypasses_closed_and_llm_fallback_on_error() -> None:
    guard = ScopePolicyGuard()
    base_rows = [_make_prescored_row(idx=i) for i in range(560)]

    # 1. Secret beyond row 500 (row 550) in DataFrame string column
    df_row550 = pd.DataFrame(base_rows)
    df_row550["ja4"] = "t13d1516h2_8daaf6152771_e5627efa2ab1"
    df_row550.loc[550, "ja4"] = "sk-proj-1234567890abcdef"
    with pytest.raises(ScopePolicyViolationError):
        guard.validate_dataframe(df_row550, strict=True)

    # 2. Categorical dtype secret
    df_cat = pd.DataFrame([_make_prescored_row(idx=0)])
    df_cat["tls_version"] = pd.Categorical(["-----BEGIN PRIVATE KEY-----"])
    with pytest.raises(ScopePolicyViolationError):
        guard.validate_dataframe(df_cat, strict=True)

    # 3. Embedded IPv4 in host tokens (DataFrame, FlowEvidence, and payload list)
    for bad_ip in ("192.168.1.1:443", "host_192.168.1.1"):
        df_ip = pd.DataFrame([_make_prescored_row(idx=0, src_host=bad_ip)])
        with pytest.raises(ScopePolicyViolationError):
            guard.validate_dataframe(df_ip, strict=True)

        flow_ip = FlowEvidence.from_row_and_fusion(_make_prescored_row(idx=0, src_host=bad_ip))
        with pytest.raises(ScopePolicyViolationError):
            guard.validate_flows([flow_ip], strict=True)

        with pytest.raises(ScopePolicyViolationError):
            guard.validate_and_sanitize_payload({"target_host_tokens": [bad_ip]}, strict=True)

    # 3b. Secret or IPv4 inside set/frozenset payload values rejected and redacted cleanly
    with pytest.raises(ScopePolicyViolationError):
        guard.validate_and_sanitize_payload({"extra_set": {"sk-proj-1234567890abcdef"}}, strict=True)
    with pytest.raises(ScopePolicyViolationError):
        guard.validate_and_sanitize_payload(
            {"extra_frozenset": frozenset({"Bearer eyJhbGciOiJIUzI1NiJ9.secretpayload"})},
            strict=True,
        )
    with pytest.raises(ScopePolicyViolationError):
        guard.validate_and_sanitize_payload({"target_host_tokens": frozenset({"10.0.0.1"})}, strict=True)
    sanitized_set, set_audit = guard.validate_and_sanitize_payload(
        {"extra_set": {"sk-proj-1234567890abcdef", "safe_value"}},
        strict=False,
    )
    assert "sk-proj-1234567890abcdef" not in str(sanitized_set["extra_set"])
    assert "safe_value" in sanitized_set["extra_set"]
    assert "OPENAI_STYLE_KEY" in set_audit.redactions_applied

    # 4. Forbidden L7 / raw-IP aliases & nested dict cell in DataFrame
    for alias in ("src_ip", "dst_ip", "ssh_password", "raw_http_body", "sni"):
        df_alias = pd.DataFrame([_make_prescored_row(idx=0)])
        df_alias[alias] = "forbidden_val"
        with pytest.raises(ScopePolicyViolationError):
            guard.validate_dataframe(df_alias, strict=True)

    df_nested = pd.DataFrame([_make_prescored_row(idx=0, custom_meta={"http_body": "password123"})])
    with pytest.raises(ScopePolicyViolationError):
        guard.validate_dataframe(df_nested, strict=True)

    # 5. FlowEvidence secret scan across additional string metadata fields
    for field_name in ("session_id", "ssh_kex_class", "cert_issuer_class", "ts_start"):
        flow_sec = FlowEvidence.from_row_and_fusion(
            _make_prescored_row(idx=0, **{field_name: "Bearer eyJhbGciOiJIUzI1NiJ9.secretpayload"})
        )
        with pytest.raises(ScopePolicyViolationError):
            guard.validate_flows([flow_sec], strict=True)

    # 6. CallableLLMNarrativeAdapter gracefully falls back offline if callable raises (with CR-06 secret redaction)
    def broken_llm(_ctx: Dict[str, Any]) -> str:
        raise RuntimeError("simulated offline network failure")

    adapter = CallableLLMNarrativeAdapter(llm_callable=broken_llm)
    out = adapter.generate({"incident_id": "inc_fallback", "primary_host_token": "host_1"})
    assert out["offline"] is True
    assert out["llm_enabled"] is False
    assert out["fallback_used"] is True
    assert "RuntimeError" in out["fallback_reason"]

    def secret_broken_llm(_ctx: Dict[str, Any]) -> str:
        raise RuntimeError(
            "Auth failed with Bearer eyJhbGciOiJIUzI1NiJ9.secretpayload and sk-proj-1234567890abcdef"
        )

    secret_adapter = CallableLLMNarrativeAdapter(llm_callable=secret_broken_llm)
    out_secret = secret_adapter.generate({"incident_id": "inc_fallback_sec", "primary_host_token": "host_1"})
    assert out_secret["offline"] is True
    assert out_secret["fallback_used"] is True
    assert "RuntimeError:" in out_secret["fallback_reason"]
    assert "eyJhbGciOiJIUzI1NiJ9.secretpayload" not in out_secret["fallback_reason"]
    assert "sk-proj-1234567890abcdef" not in out_secret["fallback_reason"]
