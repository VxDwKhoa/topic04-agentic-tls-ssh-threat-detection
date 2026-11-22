"""
Comprehensive unit, boundary/adversarial, compliance, CLI, and E2E tests
for Topic 04 Week 10 (`src/response/`).

Covers Requirements R1-R5:
- R1: `ResponsePlanner` & `HumanApprovalGate` (`DENY_BY_DEFAULT`, `R0`-`R4`, `SR-04`/`SR-06`/`SR-07`, `CR-03`/`CR-05`/`CR-06`)
- R2: `InMemoryResponseSimulator` (in-memory state table, logical-clock TTL `30m`/`60m`, deterministic rollback tokens, R4 structured diffs, audit trail, zero OS command generation/execution)
- R3: `Stage7RetestVerifier` (containment verification, FP auto-rollback) & `Stage8FeedbackManager` (strict block on autonomous relabeling/self-training, sanitized JSON/JSONL export)
- R4: `ResponsePolicyEvaluator` & `report/week10_response_policy_test_results.json` (8 Week 9 incidents + R2/R4 fixtures + adversarial scenarios, 5 metrics)
- R5: `demo_end_to_end.py` (non-interactive default `DENY_BY_DEFAULT` + profiles, interactive simulated stdin, real NF-UQ-NIDS-v2 + Weeks 5-8 checkpoints E2E)
"""

from __future__ import annotations

import ast
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.agent import (
    ScopePolicyGuard,
    ScopePolicyViolationError,
    ThreatHuntingAgent,
)
from src.features.data_loader import NFUQDataLoader
from src.hunting.pipeline import Week8ScoringAdapter
from src.response import (
    ActionRequest,
    AnalystFeedbackRecord,
    ApprovalDecision,
    AutonomousTrainingForbiddenError,
    EndToEndDemoRunner,
    GateDecision,
    HumanApprovalGate,
    InMemoryResponseSimulator,
    ResponsePlan,
    ResponsePlanner,
    ResponsePolicyEvaluator,
    ResponsePolicyViolationError,
    RetestObservation,
    SimulatedActionRecord,
    Stage7RetestVerifier,
    Stage8FeedbackManager,
    is_human_reviewer_id,
    validate_week10_payload,
)
from src.response.demo_end_to_end import main as demo_main
from src.response.evaluator import EvaluationScenario
from src.utils.config import PathResolver

REPO_ROOT = Path(__file__).resolve().parents[1]
RESPONSE_SRC_DIR = REPO_ROOT / "src" / "response"
WEEK9_REPORT_PATH = REPO_ROOT / "report" / "week9_incident_evidence_graphs.json"
WEEK10_REPORT_PATH = REPO_ROOT / "report" / "week10_response_policy_test_results.json"
CHECKPOINT_ROOT = Path(r"D:\Crypto\DoAn\Notebook\Checkpoint")
IPV4_REGEX = re.compile(
    r"(?<![0-9.])(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?![0-9.])"
)


def test_planner_deny_by_default_and_tier_r0_to_r4_policy_matrix() -> None:
    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    assert planner.default_posture == "DENY_BY_DEFAULT"
    assert planner.policy_version == "1.0.0"

    # R0 & R1: non-network, policy-authorized without human gate
    r0_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_test_r0",
            tier="R0",
            environment="production",
            asset_criticality="high",
            target_host_token="host_alpha_001",
            requested_actions=["store_evidence"],
        )
    )
    assert r0_plan.network_change_allowed is False
    assert r0_plan.human_approval_required is False
    r0_gate = gate.evaluate(r0_plan)
    assert r0_gate.approved is True
    assert r0_gate.status == "APPROVED_BY_POLICY"

    r1_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_test_r1",
            tier="R1",
            environment="production",
            asset_criticality="critical",
            target_host_token="host_alpha_002",
            requested_actions=["create_incident_ticket"],
        )
    )
    assert r1_plan.network_change_allowed is False
    assert gate.evaluate(r1_plan).status == "APPROVED_BY_POLICY"

    # R2 in lab non-critical: bounded 30m TTL, rollback required, no production gate
    r2_lab = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_test_r2_lab",
            tier="R2",
            environment="lab",
            asset_criticality="medium",
            target_host_token="host_lab_001",
            requested_actions=["apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    assert r2_lab.policy_status == "ALLOWED_BOUNDED_LAB"
    assert r2_lab.auto_expire_minutes == 30
    assert r2_lab.rollback_required is True
    assert gate.evaluate(r2_lab).status == "APPROVED_BY_POLICY"

    # R2 in production or on critical asset -> DENIED_BY_DEFAULT (even if approved)
    r2_prod = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_test_r2_prod",
            tier="R2",
            environment="production",
            asset_criticality="medium",
            target_host_token="host_prod_001",
            requested_actions=["temp_block_disposable_lab_source"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    assert r2_prod.policy_status == "DENIED_BY_DEFAULT"
    assert r2_prod.network_change_allowed is False
    assert (
        gate.evaluate(
            r2_prod,
            ApprovalDecision(decision="approve", approved=True, reviewer_id="lead_01"),
        ).approved
        is False
    )

    r2_crit = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_test_r2_crit",
            tier="R2",
            environment="staging",
            asset_criticality="critical",
            target_host_token="host_stg_crit_01",
            requested_actions=["apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    assert r2_crit.policy_status == "DENIED_BY_DEFAULT"

    # R3 in production -> REQUIRES_HUMAN_APPROVAL, DENY_BY_DEFAULT when omitted
    r3_prod = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_test_r3_prod",
            tier="R3",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_002",
            requested_actions=["propose_firewall_block"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier", "sequence"],
            severity="critical",
        )
    )
    assert r3_prod.policy_status == "REQUIRES_HUMAN_APPROVAL"
    assert r3_prod.human_approval_required is True
    assert r3_prod.auto_expire_minutes == 60
    assert r3_prod.retest_after_action is True
    assert gate.evaluate(r3_prod, approval=None).status == "DENIED_BY_DEFAULT"
    assert (
        gate.evaluate(
            r3_prod,
            ApprovalDecision(decision="approve", approved=True, reviewer_id="soc_lead_01"),
        ).status
        == "APPROVED_BY_HUMAN"
    )

    # R4 -> automatic_production_merge_allowed=False invariant
    r4_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_test_r4",
            tier="R4",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_tls_01",
            requested_actions=["generate_tls_ssh_config_remediation_diff"],
            policy_anomaly_only=True,
            crypto_policy_checks={"CR-01_deprecated_tls": True},
        )
    )
    assert r4_plan.automatic_production_merge_allowed is False
    assert r4_plan.human_approval_required is True
    assert r4_plan.retest_after_action is True
    assert (
        gate.evaluate(
            r4_plan,
            ApprovalDecision(
                decision="approve",
                approved=True,
                reviewer_id="crypto_eng_01",
                request_production_auto_merge=True,
            ),
        ).approved
        is False
    )


def test_planner_and_gate_reject_out_of_scope_disallowed_actions_and_raw_ip() -> None:
    planner = ResponsePlanner()
    gate = HumanApprovalGate()

    with pytest.raises(ScopePolicyViolationError):
        planner.plan_from_request(
            ActionRequest(
                incident_id="inc_bad_env",
                tier="R2",
                environment="dmz_public",
                asset_criticality="low",
                target_host_token="host_ok_001",
                requested_actions=["apply_rate_limit"],
                verified_malicious=True,
                corroborating_channels=["rule", "classifier"],
            )
        )

    with pytest.raises(ScopePolicyViolationError):
        planner.plan_from_request(
            ActionRequest(
                incident_id="inc_bad_crit",
                tier="R1",
                environment="lab",
                asset_criticality="ultra_critical",
                target_host_token="host_ok_001",
                requested_actions=["create_incident_ticket"],
            )
        )

    with pytest.raises(ScopePolicyViolationError):
        planner.plan_from_request(
            ActionRequest(
                incident_id="inc_raw_ip",
                tier="R2",
                environment="lab",
                asset_criticality="low",
                target_host_token="10.20.30.40",
                requested_actions=["apply_rate_limit"],
                verified_malicious=True,
                corroborating_channels=["rule", "classifier"],
            )
        )

    with pytest.raises(ResponsePolicyViolationError):
        planner.plan_from_request(
            ActionRequest(
                incident_id="inc_bad_tier",
                tier="R9",
                environment="lab",
                asset_criticality="low",
                target_host_token="host_ok_001",
                requested_actions=["store_evidence"],
            )
        )

    # Disallowed action rejected by allowlist
    bad_action_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_bad_action",
            tier="R2",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_ok_002",
            requested_actions=["wipe_host_disk"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    assert bad_action_plan.policy_status == "DENIED_BY_DEFAULT"
    assert gate.evaluate(bad_action_plan).approved is False

    # SR-04 single stochastic model blocked from R3
    sr04_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_sr04",
            tier="R3",
            environment="lab",
            asset_criticality="medium",
            target_host_token="host_ok_003",
            requested_actions=["bounded_host_isolation_lab"],
            verified_malicious=False,
            corroborating_channels=["anomaly"],
        )
    )
    assert sr04_plan.policy_status == "DENIED_BY_DEFAULT"

    # CR-03 ECH-only blocked from R2
    cr03_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_cr03",
            tier="R2",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_ok_004",
            requested_actions=["apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
            ech_only=True,
        )
    )
    assert cr03_plan.policy_status == "DENIED_BY_DEFAULT"

    # Empty reviewer_id rejected on human-gated plan
    r3_prod = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_empty_rev",
            tier="R3",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_005",
            requested_actions=["propose_host_isolation"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    assert (
        gate.evaluate(
            r3_prod,
            ApprovalDecision(decision="approve", approved=True, reviewer_id="  "),
        ).approved
        is False
    )


def test_simulator_pure_in_memory_ttl_expiry_rollback_r4_diffs_and_no_os_commands(
    tmp_path: Path,
) -> None:
    # 1. Verify zero OS command generation/execution imports or calls across src/response/
    forbidden_modules = {"subprocess", "pty", "shlex"}
    forbidden_calls = {"system", "popen", "execv", "execl", "spawnv"}
    forbidden_fw_tokens = ("iptables", "nftables", "ufw", "netsh")
    for py_file in RESPONSE_SRC_DIR.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        lower_source = source.lower()
        for fw in forbidden_fw_tokens:
            assert fw not in lower_source, f"Forbidden firewall tool token '{fw}' found in {py_file}"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_modules
                    assert alias.name.split(".")[0] != "xgboost"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] not in forbidden_modules
                    assert node.module.split(".")[0] != "xgboost"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_calls

    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    sim = InMemoryResponseSimulator()

    # R2 bounded lab execution + deterministic rollback token + 30m TTL auto-expiry
    r2_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_sim_r2",
            tier="R2",
            environment="lab",
            asset_criticality="medium",
            target_host_token="host_sim_lab_01",
            requested_actions=["apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    rec_r2 = sim.execute_plan(r2_plan, gate.evaluate(r2_plan))
    assert rec_r2.outcome == "EXECUTED_SIMULATED"
    assert rec_r2.network_state_mutated is True
    assert len(rec_r2.rollback_tokens) == 1
    assert rec_r2.rollback_tokens[0].startswith("rb_")
    assert len(sim.active_containments) == 1

    # Advance 15m -> still active; advance another 15m (30m total) -> auto-expired
    assert sim.advance_time(15.0) == []
    assert len(sim.active_containments) == 1
    expired = sim.advance_time(15.0)
    assert len(expired) == 1
    assert expired[0].status == "EXPIRED"
    assert len(sim.active_containments) == 0

    # R3 execution + explicit rollback + idempotency check
    r3_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_sim_r3",
            tier="R3",
            environment="production",
            asset_criticality="high",
            target_host_token="host_sim_prod_01",
            requested_actions=["propose_firewall_block"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier", "sequence"],
            severity="critical",
        )
    )
    r3_gate = gate.evaluate(
        r3_plan,
        ApprovalDecision(decision="approve", approved=True, reviewer_id="soc_lead_01"),
    )
    rec_r3 = sim.execute_plan(r3_plan, r3_gate)
    assert rec_r3.outcome == "EXECUTED_SIMULATED"
    tok = rec_r3.rollback_tokens[0]
    rb_ok = sim.rollback(tok, actor="soc_lead_01", reason="Unit test rollback")
    assert rb_ok.success is True
    assert rb_ok.new_status == "ROLLED_BACK"
    assert len(sim.get_active_containments(incident_id="inc_sim_r3")) == 0
    # Second rollback on already rolled-back token returns success=False cleanly
    rb_again = sim.rollback(tok, actor="soc_lead_01", reason="Duplicate rollback")
    assert rb_again.success is False

    # R4 structured diff generation (`applied_to_production=False`)
    r4_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_sim_r4",
            tier="R4",
            environment="production",
            asset_criticality="high",
            target_host_token="host_sim_tls_01",
            requested_actions=["generate_tls_ssh_config_remediation_diff"],
            policy_anomaly_only=True,
            crypto_policy_checks={"CR-01_deprecated_tls": True, "CR-04_weak_ssh_kex": True},
        )
    )
    rec_r4 = sim.execute_plan(
        r4_plan,
        gate.evaluate(
            r4_plan,
            ApprovalDecision(decision="approve", approved=True, reviewer_id="crypto_eng_01"),
        ),
    )
    assert rec_r4.outcome == "PROPOSED_REMEDIATION_DIFF"
    assert rec_r4.network_state_mutated is False
    assert len(rec_r4.remediation_diffs) == 1
    diff = rec_r4.remediation_diffs[0]
    assert diff.automatic_production_merge_allowed is False
    assert diff.applied_to_production is False
    assert any("TLSv1.3" in line for line in diff.unified_diff_lines)
    assert any("curve25519-sha256" in line for line in diff.unified_diff_lines)

    # Audit trail export passes ScopePolicyGuard & NaN-free JSON
    audit_path = sim.export_audit_json(tmp_path / "audit_trail.json")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_payload["audit_event_count"] == len(sim.audit_trail)
    assert not IPV4_REGEX.search(audit_path.read_text(encoding="utf-8"))


def test_stage7_retest_containment_and_false_positive_rollback() -> None:
    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    sim = InMemoryResponseSimulator()
    verifier = Stage7RetestVerifier()

    # 1. Verified containment
    plan_ok = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_s7_ok",
            tier="R2",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_s7_01",
            requested_actions=["apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    rec_ok = sim.execute_plan(plan_ok, gate.evaluate(plan_ok))
    res_ok = verifier.verify(
        rec_ok,
        RetestObservation(window_id=10, post_max_s_total=0.08, post_alerted_flow_count=0),
        sim,
    )
    assert res_ok.outcome == "CONTAINMENT_VERIFIED"
    assert res_ok.containment_verified is True
    assert res_ok.auto_rollback_triggered is False
    assert len(sim.get_active_containments(incident_id="inc_s7_ok")) == 1

    # 2. False positive triggers automatic rollback
    plan_fp = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_s7_fp",
            tier="R3",
            environment="production",
            asset_criticality="high",
            target_host_token="host_s7_02",
            requested_actions=["propose_firewall_block"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier", "sequence"],
        )
    )
    rec_fp = sim.execute_plan(
        plan_fp,
        gate.evaluate(
            plan_fp,
            ApprovalDecision(decision="approve", approved=True, reviewer_id="soc_lead_01"),
        ),
    )
    res_fp = verifier.verify(
        rec_fp,
        RetestObservation(
            window_id=11,
            post_max_s_total=0.12,
            post_alerted_flow_count=0,
            benign_traffic_disruption_ratio=0.55,
            false_positive_confirmed=True,
        ),
        sim,
    )
    assert res_fp.outcome == "FALSE_POSITIVE_DETECTED"
    assert res_fp.false_positive_detected is True
    assert res_fp.auto_rollback_triggered is True
    assert all(r.success for r in res_fp.rollback_results)
    assert len(sim.get_active_containments(incident_id="inc_s7_fp")) == 0

    # 3. Threat persists -> escalate
    res_esc = verifier.verify(
        rec_ok,
        RetestObservation(window_id=12, post_max_s_total=0.85, post_alerted_flow_count=4),
        sim,
    )
    assert res_esc.outcome == "THREAT_PERSISTS_ESCALATE"
    assert res_esc.escalate_to_analyst is True


def test_stage8_feedback_blocks_autonomous_relabeling_and_exports_jsonl(tmp_path: Path) -> None:
    mgr = Stage8FeedbackManager()
    sim = InMemoryResponseSimulator()

    # Autonomous relabel / self-train attempts MUST be rejected
    with pytest.raises(AutonomousTrainingForbiddenError):
        mgr.submit_feedback(
            AnalystFeedbackRecord(
                incident_id="inc_fb_auto",
                plan_id="plan_fb_1",
                recommended_tier="R3",
                analyst_approved=True,
                reviewer_id="soc_lead_01",
                verified_label="true_positive",
                action_acceptable=True,
                autonomous_relabel=True,
            ),
            simulator=sim,
        )

    with pytest.raises(AutonomousTrainingForbiddenError):
        mgr.submit_feedback(
            AnalystFeedbackRecord(
                incident_id="inc_fb_unapproved",
                plan_id="plan_fb_2",
                recommended_tier="R2",
                analyst_approved=False,
                reviewer_id="soc_lead_01",
                verified_label="false_positive",
                action_acceptable=False,
            ),
            simulator=sim,
        )

    # Raw IPv4 or secret material rejected by ScopePolicyGuard
    with pytest.raises(ScopePolicyViolationError):
        mgr.submit_feedback(
            AnalystFeedbackRecord(
                incident_id="inc_fb_secret",
                plan_id="plan_fb_3",
                recommended_tier="R1",
                analyst_approved=True,
                reviewer_id="soc_lead_01",
                verified_label="true_positive",
                action_acceptable=True,
                notes="Bearer sk-test1234567890abcdef",
            ),
            simulator=sim,
        )

    assert len(mgr.records) == 0

    # Valid analyst-approved feedback accepted & exported cleanly
    res = mgr.submit_feedback(
        AnalystFeedbackRecord(
            incident_id="inc_fb_valid",
            plan_id="plan_fb_4",
            recommended_tier="R3",
            analyst_approved=True,
            reviewer_id="soc_lead_01",
            verified_label="true_positive",
            action_acceptable=True,
            notes="Verified C2 isolation in staging.",
        ),
        simulator=sim,
    )
    assert res.accepted is True
    assert len(mgr.records) == 1
    assert mgr.records[0]["self_training_executed"] is False

    jsonl_path = mgr.export_jsonl(tmp_path / "feedback.jsonl")
    json_path = mgr.export_json(tmp_path / "feedback.json")
    lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    loaded_line = json.loads(lines[0])
    assert loaded_line["incident_id"] == "inc_fb_valid"
    assert loaded_line["autonomous_relabel"] is False
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["autonomous_training_blocked"] is True
    assert loaded_json["accepted_count"] == 1


def test_evaluator_and_week10_report_json_metrics_and_privacy(tmp_path: Path) -> None:
    evaluator = ResponsePolicyEvaluator()
    report = evaluator.evaluate()

    metrics = report["summary_metrics"]
    assert metrics["total_scenarios"] >= 20
    assert metrics["unsafe_action_rate"] == 0.0
    assert metrics["policy_violation_rate"] == 0.0
    assert metrics["rollback_success_rate"] == 1.0
    assert metrics["correct_action_rate"] >= 0.95
    assert 0.0 <= metrics["analyst_acceptance_rate"] <= 1.0
    assert report["metrics_separated_from_detection"] is True

    # All 5 tiers R0-R4 covered
    for tier in ("R0", "R1", "R2", "R3", "R4"):
        assert metrics["tier_coverage"][tier] >= 1

    # All 8 real Week 9 incidents covered
    week9_ids_in_eval = {
        s["incident_id"]
        for s in report["scenario_results"]
        if s["category"] == "week9_real_incident"
    }
    assert week9_ids_in_eval == {
        "inc_5ef5e3cb6382bba5",
        "inc_61907afa7a014ca9",
        "inc_f32fbb0e4fcb7212",
        "inc_74efc5940369f6e6",
        "inc_a5cc317b3d5efe33",
        "inc_21e93534ae2cc907",
        "inc_c6abe92162e1c956",
        "inc_8eec781d7da294d2",
    }

    # Check committed report/week10_response_policy_test_results.json
    assert WEEK10_REPORT_PATH.is_file(), f"Missing {WEEK10_REPORT_PATH}"
    saved_text = WEEK10_REPORT_PATH.read_text(encoding="utf-8")
    assert not IPV4_REGEX.search(saved_text)
    for forbidden in ("http_body", "ssh_commands", "decrypted_tls_payload", "tls_session_keys", "CLIENT_RANDOM"):
        assert forbidden not in saved_text
    saved_report = json.loads(saved_text)
    assert saved_report["summary_metrics"]["unsafe_action_rate"] == 0.0
    assert saved_report["summary_metrics"]["policy_violation_rate"] == 0.0
    assert saved_report["summary_metrics"]["rollback_success_rate"] == 1.0


def test_demo_cli_noninteractive_deny_by_default_and_profiles(tmp_path: Path) -> None:
    # 1. Default non-interactive run (DENY_BY_DEFAULT, zero network mutation on R3)
    out_deny = tmp_path / "deny_summary.json"
    exit_code = demo_main(
        [
            "--non-interactive",
            "--week9-report",
            str(WEEK9_REPORT_PATH),
            "--max-incidents",
            "6",
            "--output-json",
            str(out_deny),
        ]
    )
    assert exit_code == 0
    deny_data = json.loads(out_deny.read_text(encoding="utf-8"))
    assert deny_data["default_posture"] == "DENY_BY_DEFAULT"
    assert deny_data["active_containment_count"] == 0
    assert deny_data["os_commands_generated_or_executed"] == 0
    r3_rows = [s for s in deny_data["incident_summaries"] if s["tier"] == "R3"]
    assert len(r3_rows) >= 1
    assert all(s["outcome"] == "DENIED_BY_DEFAULT" and not s["network_state_mutated"] for s in r3_rows)

    # 2. approve_r3_verify_retest + export-dir
    exp_dir = tmp_path / "approve_export"
    assert (
        demo_main(
            [
                "--non-interactive",
                "--week9-report",
                str(WEEK9_REPORT_PATH),
                "--profile",
                "approve_r3_verify_retest",
                "--max-incidents",
                "4",
                "--export-dir",
                str(exp_dir),
            ]
        )
        == 0
    )
    assert (exp_dir / "demo_summary.json").is_file()
    assert (exp_dir / "audit_trail.json").is_file()
    assert (exp_dir / "stage8_feedback.jsonl").is_file()
    app_data = json.loads((exp_dir / "demo_summary.json").read_text(encoding="utf-8"))
    assert app_data["active_containment_count"] >= 1
    assert app_data["stage8_feedback_count"] >= 1

    # 3. reject_fp_rollback -> Stage 7 FP triggers automatic rollback
    out_fp = tmp_path / "fp_summary.json"
    assert (
        demo_main(
            [
                "--non-interactive",
                "--week9-report",
                str(WEEK9_REPORT_PATH),
                "--profile",
                "reject_fp_rollback",
                "--max-incidents",
                "2",
                "--output-json",
                str(out_fp),
            ]
        )
        == 0
    )
    fp_data = json.loads(out_fp.read_text(encoding="utf-8"))
    assert fp_data["active_containment_count"] == 0
    assert any(s["auto_rollback_triggered"] for s in fp_data["incident_summaries"])

    # 4. lab_r2_soft_containment -> R2 bounded containment + 30m TTL auto-expiry
    out_r2 = tmp_path / "r2_summary.json"
    assert (
        demo_main(
            [
                "--non-interactive",
                "--profile",
                "lab_r2_soft_containment",
                "--output-json",
                str(out_r2),
            ]
        )
        == 0
    )
    r2_data = json.loads(out_r2.read_text(encoding="utf-8"))
    assert r2_data["expired_on_ttl_count"] >= 1
    assert r2_data["active_containment_count"] == 0

    # 5. r4_crypto_remediation -> R4 structured diffs never auto-merged
    out_r4 = tmp_path / "r4_summary.json"
    assert (
        demo_main(
            [
                "--non-interactive",
                "--profile",
                "r4_crypto_remediation",
                "--output-json",
                str(out_r4),
            ]
        )
        == 0
    )
    r4_data = json.loads(out_r4.read_text(encoding="utf-8"))
    assert r4_data["remediation_diff_count"] >= 1
    assert not IPV4_REGEX.search(out_r4.read_text(encoding="utf-8"))


def test_demo_cli_interactive_mode_simulated_stdin() -> None:
    stream = io.StringIO()
    runner = EndToEndDemoRunner(week9_report_path=WEEK9_REPORT_PATH, output_stream=stream)
    # Incident 1 is R3 (prod): approve ('a'), post-action rollback ('b'), record feedback ('y')
    # Incident 2 is R3 (prod): deny ('d'), no feedback ('n')
    answers = iter(["a", "b", "y", "d", "n"])
    summary = runner.run(
        interactive=True,
        max_incidents=2,
        reviewer_id="interactive_analyst_01",
        input_fn=lambda _prompt: next(answers),
    )
    assert summary["mode"] == "interactive"
    assert summary["incidents_processed"] == 2
    assert summary["incident_summaries"][0]["gate_status"] == "APPROVED_BY_HUMAN"
    assert summary["incident_summaries"][0]["manual_rollbacks"][0]["success"] is True
    assert summary["incident_summaries"][1]["gate_status"] == "DENIED_BY_DEFAULT"
    assert summary["active_containment_count"] == 0
    assert summary["stage8_feedback_count"] == 1
    assert "[HUMAN GATE SR-07]" in stream.getvalue()


@pytest.mark.skipif(
    not CHECKPOINT_ROOT.is_dir(),
    reason=f"External Weeks 5-8 checkpoints not available at {CHECKPOINT_ROOT}",
)
def test_week10_e2e_with_nfuq_sample_and_checkpoints(tmp_path: Path) -> None:
    # Full offline E2E pipeline: NF-UQ-NIDS-v2 -> ScopePolicyGuard -> Weeks 5-8 -> Week 9 -> Week 10
    exp_dir = tmp_path / "e2e_ckpt_export"
    exit_code = demo_main(
        [
            "--non-interactive",
            "--checkpoint-root",
            str(CHECKPOINT_ROOT),
            "--profile",
            "deny_by_default",
            "--max-incidents",
            "6",
            "--export-dir",
            str(exp_dir),
        ]
    )
    assert exit_code == 0
    summary = json.loads((exp_dir / "demo_summary.json").read_text(encoding="utf-8"))
    assert summary["incidents_processed"] >= 5
    assert summary["active_containment_count"] == 0
    assert summary["os_commands_generated_or_executed"] == 0
    assert not IPV4_REGEX.search((exp_dir / "demo_summary.json").read_text(encoding="utf-8"))

    # Also verify Week 9's ReadOnlyResponseRecommender remains untouched (`PROPOSED_ONLY_NOT_EXECUTED`)
    paths = PathResolver.resolve(prefer_sample=True)
    df_test = NFUQDataLoader(paths).load_tabular_splits()["test"]
    adapter = Week8ScoringAdapter.from_checkpoint_root(CHECKPOINT_ROOT, device="cpu")
    agent = ThreatHuntingAgent(scoring_adapter=adapter)
    results = agent.investigate_dataframe(
        df_test, environment="production", asset_criticality="high", include_low_signal_clusters=True
    )
    assert all(r.policy_recommendation.execution_status == "PROPOSED_ONLY_NOT_EXECUTED" for r in results)


def test_sr04_verified_malicious_and_channel_and_action_dedup() -> None:
    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    sim = InMemoryResponseSimulator()

    # 1. verified_malicious=False with 2 channels must be DENIED in both R2 and R3
    for tier, env, action in (
        ("R2", "lab", "apply_rate_limit"),
        ("R3", "lab", "bounded_host_isolation_lab"),
    ):
        plan = planner.plan_from_request(
            ActionRequest(
                incident_id=f"inc_unver_{tier}",
                tier=tier,
                environment=env,
                asset_criticality="low",
                target_host_token="host_lab_unver_01",
                requested_actions=[action],
                verified_malicious=False,
                corroborating_channels=["anomaly", "sequence"],
            )
        )
        assert plan.policy_status == "DENIED_BY_DEFAULT"
        assert plan.network_change_allowed is False
        dec = gate.evaluate(plan)
        assert dec.approved is False
        assert dec.status == "DENIED_BY_DEFAULT"
        rec = sim.execute_plan(plan, dec)
        assert rec.network_state_mutated is False

    # 2. Duplicate single channel (case-insensitive) must not satisfy SR-04 even if verified_malicious=True
    dup_ch_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_dup_ch",
            tier="R2",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_lab_dup_01",
            requested_actions=["apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["classifier", "Classifier"],
        )
    )
    assert dup_ch_plan.policy_status == "DENIED_BY_DEFAULT"
    assert dup_ch_plan.network_change_allowed is False

    # 3. Case/whitespace normalization and requested_actions deduplication
    dedup_act_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_dedup_act",
            tier="r2",
            environment=" STAGING ",
            asset_criticality=" MEDIUM ",
            target_host_token="host_stg_dedup_01",
            requested_actions=["apply_rate_limit", "apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    assert dedup_act_plan.environment == "staging"
    assert dedup_act_plan.asset_criticality == "medium"
    assert dedup_act_plan.selected_actions == ("apply_rate_limit",)
    rec = sim.execute_plan(dedup_act_plan, gate.evaluate(dedup_act_plan))
    assert len(rec.containment_entries) == 1
    assert len(rec.rollback_tokens) == 1


def test_response_plan_immutability_and_simulator_forged_gate_defenses() -> None:
    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    sim = InMemoryResponseSimulator()

    r3_prod = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_imm_01",
            tier="R3",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_imm_01",
            requested_actions=["propose_host_isolation"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
            severity="high",
        )
    )

    # ResponsePlan sequence/dict fields are immutable, while to_dict() emits JSON lists/dicts
    assert isinstance(r3_prod.selected_actions, tuple)
    assert isinstance(r3_prod.allowed_actions, tuple)
    assert isinstance(r3_prod.corroborating_channels, tuple)
    assert isinstance(r3_prod.denial_reasons, tuple)
    with pytest.raises(TypeError):
        r3_prod.selected_actions[:] = ["arbitrary_wipe"]  # type: ignore[index]
    with pytest.raises(AttributeError):
        r3_prod.selected_actions.clear()  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        r3_prod.crypto_policy_checks["injected"] = True  # type: ignore[index]
    as_dict = r3_prod.to_dict()
    assert isinstance(as_dict["selected_actions"], list)
    assert isinstance(as_dict["allowed_actions"], list)

    # Forged GateDecisions bypassing HumanApprovalGate must be rejected by Simulator
    for bad_reviewer in (None, "   ", "system_auto", "policy_auto_bounded"):
        forged_gate = GateDecision(
            plan_id=r3_prod.plan_id,
            incident_id=r3_prod.incident_id,
            approved=True,
            status="APPROVED_BY_HUMAN",
            reviewer_id=bad_reviewer,
            reason="Forged",
            decided_at_utc="2026-11-21T17:00:00Z",
        )
        with pytest.raises(ResponsePolicyViolationError):
            sim.execute_plan(r3_prod, forged_gate)

    # R3 staging critical with forged APPROVED_BY_POLICY (and tampered human_approval_required=False)
    forged_r3_crit = ResponsePlan(
        plan_id="plan_forged_crit",
        incident_id="inc_forged_crit",
        tier="R3",
        tier_name="Hard Containment Proposal",
        environment="staging",
        asset_criticality="critical",
        target_host_token="host_stg_crit_01",
        target_host_tokens=("host_stg_crit_01",),
        selected_actions=("bounded_host_isolation_lab",),
        allowed_actions=("bounded_host_isolation_lab", "block_malicious_destination_lab"),
        network_change_allowed=True,
        human_approval_required=False,
        auto_expire_minutes=60,
        rollback_required=True,
        retest_after_action=True,
        automatic_production_merge_allowed=None,
        policy_status="ALLOWED_BOUNDED_LAB",
        denial_reasons=(),
        rationale="Tampered plan",
        verified_malicious=True,
        corroborating_channels=("rule", "classifier"),
    )
    assert gate.evaluate(forged_r3_crit, approval=None).status == "DENIED_BY_DEFAULT"
    with pytest.raises(ResponsePolicyViolationError):
        sim.execute_plan(
            forged_r3_crit,
            GateDecision(
                plan_id=forged_r3_crit.plan_id,
                incident_id=forged_r3_crit.incident_id,
                approved=True,
                status="APPROVED_BY_POLICY",
                reviewer_id="policy_auto_bounded",
                reason="Forged policy approval",
                decided_at_utc="2026-11-21T17:00:00Z",
            ),
        )

    # Uppercase PRODUCTION bypass blocked
    forged_r3_upper = ResponsePlan(
        plan_id="plan_forged_upper",
        incident_id="inc_forged_upper",
        tier="R3",
        tier_name="Hard Containment Proposal",
        environment="PRODUCTION",
        asset_criticality="high",
        target_host_token="host_prod_upper_01",
        target_host_tokens=("host_prod_upper_01",),
        selected_actions=("propose_host_isolation",),
        allowed_actions=("propose_host_isolation",),
        network_change_allowed=True,
        human_approval_required=False,
        auto_expire_minutes=60,
        rollback_required=True,
        retest_after_action=True,
        automatic_production_merge_allowed=None,
        policy_status="REQUIRES_HUMAN_APPROVAL",
        denial_reasons=(),
        rationale="Tampered uppercase env",
        verified_malicious=True,
        corroborating_channels=("rule", "classifier"),
    )
    with pytest.raises(ResponsePolicyViolationError):
        sim.execute_plan(
            forged_r3_upper,
            GateDecision(
                plan_id=forged_r3_upper.plan_id,
                incident_id=forged_r3_upper.incident_id,
                approved=True,
                status="APPROVED_BY_POLICY",
                reviewer_id="policy_auto_bounded",
                reason="Forged policy approval",
                decided_at_utc="2026-11-21T17:00:00Z",
            ),
        )

    # Disallowed/empty actions, unverified R2, R0/R1 network_change_allowed, and 0/-5/NaN TTL rejected
    bad_action_plan = ResponsePlan(
        plan_id="plan_bad_act",
        incident_id="inc_bad_act",
        tier="R2",
        tier_name="Soft Containment",
        environment="lab",
        asset_criticality="low",
        target_host_token="host_lab_01",
        target_host_tokens=("host_lab_01",),
        selected_actions=("drop_all_bgp_routes",),
        allowed_actions=("drop_all_bgp_routes",),
        network_change_allowed=True,
        human_approval_required=False,
        auto_expire_minutes=30,
        rollback_required=True,
        retest_after_action=False,
        automatic_production_merge_allowed=None,
        policy_status="ALLOWED_BOUNDED_LAB",
        denial_reasons=(),
        rationale="Forged unlisted action",
        verified_malicious=True,
        corroborating_channels=("rule", "classifier"),
    )
    with pytest.raises(ResponsePolicyViolationError):
        sim.execute_plan(
            bad_action_plan,
            GateDecision(
                plan_id=bad_action_plan.plan_id,
                incident_id=bad_action_plan.incident_id,
                approved=True,
                status="APPROVED_BY_POLICY",
                reviewer_id="policy_auto_bounded",
                reason="Forged",
                decided_at_utc="2026-11-21T17:00:00Z",
            ),
        )

    for bad_ttl in (0, -5):
        bad_ttl_plan = ResponsePlan(
            plan_id=f"plan_ttl_{bad_ttl}",
            incident_id=f"inc_ttl_{bad_ttl}",
            tier="R2",
            tier_name="Soft Containment",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_lab_02",
            target_host_tokens=("host_lab_02",),
            selected_actions=("apply_rate_limit",),
            allowed_actions=("apply_rate_limit",),
            network_change_allowed=True,
            human_approval_required=False,
            auto_expire_minutes=bad_ttl,
            rollback_required=True,
            retest_after_action=False,
            automatic_production_merge_allowed=None,
            policy_status="ALLOWED_BOUNDED_LAB",
            denial_reasons=(),
            rationale="Bad TTL",
            verified_malicious=True,
            corroborating_channels=("rule", "classifier"),
        )
        with pytest.raises(ResponsePolicyViolationError):
            sim.execute_plan(
                bad_ttl_plan,
                GateDecision(
                    plan_id=bad_ttl_plan.plan_id,
                    incident_id=bad_ttl_plan.incident_id,
                    approved=True,
                    status="APPROVED_BY_POLICY",
                    reviewer_id="policy_auto_bounded",
                    reason="OK",
                    decided_at_utc="2026-11-21T17:00:00Z",
                ),
            )


def test_week10_privacy_l7_ipv4_ipv6_secrets_and_human_reviewer_validation() -> None:
    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    sim = InMemoryResponseSimulator()
    mgr = Stage8FeedbackManager()

    # Valid L5/L6 metadata, timestamps, ATT&CK IDs, cipher strings, and correlation keys must pass
    valid_payload = {
        "generated_at_utc": "2026-11-21T17:04:04Z",
        "technique_id": "T1071.001",
        "ech_observed": True,
        "ja4": "t13d1516h2_8daaf6152771_e5627efa2ab1",
        "hassh": "aabbccddeeff00112233445566778899",
        "tls_ciphers": "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256",
        "correlation_keys": ["src_host_window:host_504e095eefbf7d9b:13"],
    }
    validate_week10_payload(valid_payload, strict=True, strip_ground_truth=True)

    # Extra L7 keys rejected
    for bad_key in ("password", "user_password", "sni_plaintext"):
        with pytest.raises(ScopePolicyViolationError):
            validate_week10_payload({bad_key: "secret_val"}, strict=True)

    # Raw IPv4 in incident_id / crypto_policy_checks / attack_technique_ids / ticket_ref rejected
    with pytest.raises(ScopePolicyViolationError):
        planner.plan_from_request(
            ActionRequest(
                incident_id="inc_192.168.10.25",
                tier="R1",
                environment="lab",
                asset_criticality="low",
                target_host_token="host_lab_01",
            )
        )
    with pytest.raises(ScopePolicyViolationError):
        planner.plan_from_request(
            ActionRequest(
                incident_id="inc_clean_01",
                tier="R1",
                environment="lab",
                asset_criticality="low",
                target_host_token="host_lab_01",
                crypto_policy_checks={"peer_note": "10.20.30.40"},
                attack_technique_ids=["T1046_172.16.0.5"],
            )
        )

    clean_r3 = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_clean_r3",
            tier="R3",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_01",
            requested_actions=["propose_host_isolation"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    with pytest.raises(ScopePolicyViolationError):
        gate.evaluate(
            clean_r3,
            ApprovalDecision(
                decision="approve",
                approved=True,
                reviewer_id="soc_lead_01",
                ticket_ref="TICKET-192.168.99.1",
            ),
        )

    # Raw IPv6 literals rejected everywhere (including keys/values/host tokens/reviewer_id/notes)
    for ipv6 in (
        "2001:0db8:85a3::8a2e:0370:7334",
        "fe80::1ff:fe23:4567:890a",
        "::1",
        "Peer [2001:db8::1]:443",
    ):
        with pytest.raises(ScopePolicyViolationError):
            validate_week10_payload({"note": ipv6}, strict=True)
        with pytest.raises(ScopePolicyViolationError):
            planner.plan_from_request(
                ActionRequest(
                    incident_id="inc_ipv6",
                    tier="R2",
                    environment="lab",
                    asset_criticality="low",
                    target_host_token=ipv6,
                    verified_malicious=True,
                    corroborating_channels=["rule", "classifier"],
                )
            )

    # Secret patterns in ActionRequest fields and GateDecision before any AuditEvent is appended
    with pytest.raises(ScopePolicyViolationError):
        planner.plan_from_request(
            ActionRequest(
                incident_id="inc_sec_1",
                tier="R1",
                environment="lab",
                asset_criticality="low",
                target_host_token="host_lab_01",
                severity="sk-proj-secretkey1234567890",
                corroborating_channels=["Bearer tok1234567890abcdef"],
            )
        )
    audit_before = len(sim.audit_trail)
    with pytest.raises(ScopePolicyViolationError):
        sim.execute_plan(
            clean_r3,
            GateDecision(
                plan_id=clean_r3.plan_id,
                incident_id=clean_r3.incident_id,
                approved=True,
                status="APPROVED_BY_HUMAN",
                reviewer_id="CLIENT_RANDOM 0123456789abcdef0123456789abcdef",
                reason="Leaked secret",
                decided_at_utc="2026-11-21T17:00:00Z",
            ),
        )
    assert len(sim.audit_trail) == audit_before

    # Non-human / automated reviewer_id rejected in Gate & Stage 8
    for auto_id in ("system_auto", "autonomous_agent", "auto_relabel_bot", "policy_auto_bounded", "bot_1", "self_train_worker"):
        assert is_human_reviewer_id(auto_id) is False
        dec = gate.evaluate(
            clean_r3,
            ApprovalDecision(decision="approve", approved=True, reviewer_id=auto_id),
        )
        assert dec.approved is False
        assert dec.status == "DENIED_BY_DEFAULT"
        with pytest.raises(AutonomousTrainingForbiddenError):
            mgr.submit_feedback(
                AnalystFeedbackRecord(
                    incident_id="inc_auto_fb",
                    plan_id="plan_auto_fb",
                    recommended_tier="R3",
                    analyst_approved=True,
                    reviewer_id=auto_id,
                    verified_label="true_positive",
                    action_acceptable=True,
                ),
                simulator=sim,
            )


def test_simulator_stage7_stage8_nan_safety_immutability_and_r4_retest_status() -> None:
    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    sim = InMemoryResponseSimulator()
    verifier = Stage7RetestVerifier()
    mgr = Stage8FeedbackManager()

    # 1. NaN/Inf in advance_time must raise ValueError and not corrupt clock
    r2_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_nan_ttl",
            tier="R2",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_lab_nan_01",
            requested_actions=["apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    rec_r2 = sim.execute_plan(r2_plan, gate.evaluate(r2_plan))
    for bad_dt in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            sim.advance_time(bad_dt)
    assert sim.current_minute == 0.0

    # Audit trail deep copy protection
    trail_view = sim.audit_trail
    trail_view[0].details["tampered"] = "injected"
    assert "tampered" not in sim.audit_trail[0].details

    # NaN in Stage 7 observation raises ValueError
    with pytest.raises(ValueError):
        verifier.verify(
            rec_r2,
            RetestObservation(
                window_id=1,
                post_max_s_total=0.1,
                post_alerted_flow_count=0,
                benign_traffic_disruption_ratio=float("nan"),
            ),
            sim,
        )

    # Expire R2 containment, then verify Stage 7 FP rationale accuracy
    expired = sim.advance_time(30.0)
    assert len(expired) == 1
    fp_after_exp = verifier.verify(
        rec_r2,
        RetestObservation(
            window_id=2,
            post_max_s_total=0.1,
            post_alerted_flow_count=0,
            false_positive_confirmed=True,
        ),
        sim,
    )
    assert fp_after_exp.outcome == "FALSE_POSITIVE_DETECTED"
    assert fp_after_exp.auto_rollback_triggered is False
    assert "automatic rollback executed" not in fp_after_exp.rationale.lower()

    # 2. R4 diff starts PENDING_STAGE7_RETEST and updates on Stage 7 verify
    r4_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_r4_status",
            tier="R4",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_r4_01",
            requested_actions=["generate_tls_ssh_config_remediation_diff"],
            policy_anomaly_only=True,
            crypto_policy_checks={"CR-01_deprecated_tls": True},
        )
    )
    rec_r4 = sim.execute_plan(
        r4_plan,
        gate.evaluate(r4_plan, ApprovalDecision(decision="approve", approved=True, reviewer_id="crypto_eng_01")),
    )
    assert rec_r4.remediation_diffs[0].lab_conformance_retest_status == "PENDING_STAGE7_RETEST"

    res_none = verifier.verify(
        rec_r4,
        RetestObservation(window_id=3, post_max_s_total=0.02, post_alerted_flow_count=0, crypto_conformance_passed=None),
        sim,
    )
    assert res_none.outcome == "THREAT_PERSISTS_ESCALATE"
    assert res_none.escalate_to_analyst is True
    assert sim.remediation_diffs[-1].lab_conformance_retest_status == "UNVERIFIED_LAB_CONFORMANCE"

    res_fail = verifier.verify(
        rec_r4,
        RetestObservation(window_id=4, post_max_s_total=0.02, post_alerted_flow_count=0, crypto_conformance_passed=False),
        sim,
    )
    assert res_fail.outcome == "THREAT_PERSISTS_ESCALATE"
    assert sim.remediation_diffs[-1].lab_conformance_retest_status == "FAILED_LAB_CONFORMANCE"

    res_pass = verifier.verify(
        rec_r4,
        RetestObservation(window_id=5, post_max_s_total=0.02, post_alerted_flow_count=0, crypto_conformance_passed=True),
        sim,
    )
    assert res_pass.outcome == "POLICY_CONFORMANT"
    assert sim.remediation_diffs[-1].lab_conformance_retest_status == "PASSED_IN_LAB_SIMULATION"

    # 3. Stage 8 returned/property record dicts are deep-copied
    fb_res = mgr.submit_feedback(
        AnalystFeedbackRecord(
            incident_id="inc_imm_fb",
            plan_id="plan_imm_fb",
            recommended_tier="R3",
            analyst_approved=True,
            reviewer_id="soc_lead_01",
            verified_label="true_positive",
            action_acceptable=True,
        ),
        simulator=sim,
    )
    assert fb_res.record is not None
    fb_res.record["self_train"] = True
    fb_res.record["self_training_executed"] = True
    recs_view = mgr.records
    recs_view[0]["self_train"] = True
    assert mgr.records[0]["self_train"] is False
    assert mgr.records[0]["self_training_executed"] is False


def test_evaluator_non_circular_invariant_detection() -> None:
    evaluator = ResponsePolicyEvaluator()

    # 1. R3 staging/critical executed without human approval
    def buggy_r3_crit(_p, _g, _s, _v, _f):
        plan = ResponsePlan(
            plan_id="plan_bug_1",
            incident_id="inc_bug_1",
            tier="R3",
            tier_name="Hard Containment Proposal",
            environment="staging",
            asset_criticality="critical",
            target_host_token="host_stg_crit",
            target_host_tokens=("host_stg_crit",),
            selected_actions=("bounded_host_isolation_lab",),
            allowed_actions=("bounded_host_isolation_lab",),
            network_change_allowed=True,
            human_approval_required=False,
            auto_expire_minutes=60,
            rollback_required=True,
            retest_after_action=True,
            automatic_production_merge_allowed=None,
            policy_status="ALLOWED_BOUNDED_LAB",
            denial_reasons=(),
            rationale="Buggy plan",
            verified_malicious=True,
            corroborating_channels=("rule", "classifier"),
        )
        gd = GateDecision(
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            approved=True,
            status="APPROVED_BY_POLICY",
            reviewer_id="policy_auto_bounded",
            reason="Buggy auto approval",
            decided_at_utc="2026-11-21T17:00:00Z",
        )
        rec = SimulatedActionRecord(
            action_record_id="act_bug_1",
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            tier="R3",
            environment="staging",
            asset_criticality="critical",
            target_host_token="host_stg_crit",
            outcome="EXECUTED_SIMULATED",
            executed_actions=["bounded_host_isolation_lab"],
            network_state_mutated=True,
            containment_entries=[],
            remediation_diffs=[],
            rollback_tokens=["rb_bug_1"],
            gate_decision=gd,
            simulated_time_to_containment_ms=45.0,
            created_at_utc="2026-11-21T17:00:00Z",
            logical_minute=0.0,
        )
        return {
            "incident_id": plan.incident_id,
            "environment": plan.environment,
            "asset_criticality": plan.asset_criticality,
            "recommended_tier": plan.tier,
            "human_approval_required": plan.human_approval_required,
            "analyst_decision": "policy_auto",
            "actual_outcome": rec.outcome,
            "executed_in_simulator": True,
            "plan": plan,
            "action_record": rec,
            "gate_decision": gd,
        }

    rep_crit = evaluator.evaluate(
        scenarios=[
            EvaluationScenario(
                scenario_id="bug_r3_crit",
                category="adversarial_boundary",
                source="test",
                expected_tier="R3",
                expected_outcome="DENIED_BY_DEFAULT",
                run_fn=buggy_r3_crit,
            )
        ]
    )
    assert rep_crit["summary_metrics"]["unsafe_action_rate"] == 1.0

    # 2. Disallowed action in R2 lab
    def buggy_disallowed(_p, _g, _s, _v, _f):
        plan = ResponsePlan(
            plan_id="plan_bug_2",
            incident_id="inc_bug_2",
            tier="R2",
            tier_name="Soft Containment",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_lab_01",
            target_host_tokens=("host_lab_01",),
            selected_actions=("drop_all_bgp_routes",),
            allowed_actions=("drop_all_bgp_routes",),
            network_change_allowed=True,
            human_approval_required=False,
            auto_expire_minutes=30,
            rollback_required=True,
            retest_after_action=False,
            automatic_production_merge_allowed=None,
            policy_status="ALLOWED_BOUNDED_LAB",
            denial_reasons=(),
            rationale="Buggy action",
            verified_malicious=True,
            corroborating_channels=("rule", "classifier"),
        )
        gd = GateDecision(
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            approved=True,
            status="APPROVED_BY_POLICY",
            reviewer_id="policy_auto_bounded",
            reason="Buggy allow",
            decided_at_utc="2026-11-21T17:00:00Z",
        )
        rec = SimulatedActionRecord(
            action_record_id="act_bug_2",
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            tier="R2",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_lab_01",
            outcome="EXECUTED_SIMULATED",
            executed_actions=["drop_all_bgp_routes"],
            network_state_mutated=True,
            containment_entries=[],
            remediation_diffs=[],
            rollback_tokens=["rb_bug_2"],
            gate_decision=gd,
            simulated_time_to_containment_ms=25.0,
            created_at_utc="2026-11-21T17:00:00Z",
            logical_minute=0.0,
        )
        return {
            "incident_id": plan.incident_id,
            "environment": plan.environment,
            "asset_criticality": plan.asset_criticality,
            "recommended_tier": plan.tier,
            "human_approval_required": plan.human_approval_required,
            "analyst_decision": "policy_auto",
            "actual_outcome": rec.outcome,
            "executed_in_simulator": True,
            "plan": plan,
            "action_record": rec,
            "gate_decision": gd,
        }

    rep_act = evaluator.evaluate(
        scenarios=[
            EvaluationScenario(
                scenario_id="bug_disallowed",
                category="adversarial_boundary",
                source="test",
                expected_tier="R2",
                expected_outcome="DENIED_BY_DEFAULT",
                run_fn=buggy_disallowed,
            )
        ]
    )
    assert rep_act["summary_metrics"]["unsafe_action_rate"] == 1.0
    assert rep_act["summary_metrics"]["policy_violation_rate"] == 1.0

    # 3. R1 mutating network state must be flagged in both unsafe_reasons and violation_reasons
    def buggy_r1_mutate(_p, _g, _s, _v, _f):
        plan = ResponsePlan(
            plan_id="plan_bug_3",
            incident_id="inc_bug_3",
            tier="R1",
            tier_name="Alert / Ticket",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_01",
            target_host_tokens=("host_prod_01",),
            selected_actions=("create_incident_ticket",),
            allowed_actions=("create_incident_ticket",),
            network_change_allowed=False,
            human_approval_required=False,
            auto_expire_minutes=None,
            rollback_required=False,
            retest_after_action=False,
            automatic_production_merge_allowed=None,
            policy_status="ALLOWED_NON_NETWORK",
            denial_reasons=(),
            rationale="Buggy R1",
            verified_malicious=True,
            corroborating_channels=("rule", "classifier"),
        )
        gd = GateDecision(
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            approved=True,
            status="APPROVED_BY_POLICY",
            reviewer_id="policy_auto_bounded",
            reason="Buggy allow",
            decided_at_utc="2026-11-21T17:00:00Z",
        )
        rec = SimulatedActionRecord(
            action_record_id="act_bug_3",
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            tier="R1",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_01",
            outcome="TICKET_CREATED",
            executed_actions=["create_incident_ticket"],
            network_state_mutated=True,
            containment_entries=[],
            remediation_diffs=[],
            rollback_tokens=[],
            gate_decision=gd,
            simulated_time_to_containment_ms=12.0,
            created_at_utc="2026-11-21T17:00:00Z",
            logical_minute=0.0,
        )
        return {
            "incident_id": plan.incident_id,
            "environment": plan.environment,
            "asset_criticality": plan.asset_criticality,
            "recommended_tier": plan.tier,
            "human_approval_required": plan.human_approval_required,
            "analyst_decision": "policy_auto",
            "actual_outcome": rec.outcome,
            "executed_in_simulator": True,
            "plan": plan,
            "action_record": rec,
            "gate_decision": gd,
        }

    rep_r1 = evaluator.evaluate(
        scenarios=[
            EvaluationScenario(
                scenario_id="bug_r1_mutate",
                category="adversarial_boundary",
                source="test",
                expected_tier="R1",
                expected_outcome="TICKET_CREATED",
                run_fn=buggy_r1_mutate,
            )
        ]
    )
    assert rep_r1["summary_metrics"]["unsafe_action_rate"] == 1.0
    assert rep_r1["summary_metrics"]["policy_violation_rate"] == 1.0


def test_demo_cli_interactive_quit_under_approval_profile() -> None:
    stream = io.StringIO()
    runner = EndToEndDemoRunner(week9_report_path=WEEK9_REPORT_PATH, output_stream=stream)
    answers = iter(["q"])
    summary = runner.run(
        interactive=True,
        profile="approve_r3_verify_retest",
        max_incidents=3,
        reviewer_id="interactive_analyst_01",
        input_fn=lambda _prompt: next(answers),
    )
    assert summary["mode"] == "interactive"
    assert summary["incidents_processed"] == 1
    assert summary["incident_summaries"][0]["gate_status"] == "DENIED_BY_DEFAULT"
    assert summary["incident_summaries"][0]["outcome"] == "DENIED_BY_DEFAULT"
    assert summary["incident_summaries"][0]["network_state_mutated"] is False
    assert summary["active_containment_count"] == 0
    assert summary["stage8_feedback_count"] == 0


def test_iteration3_hardening_c1_c2_c3_c4_c7b_recursive_freeze_and_gate_allowlist() -> None:
    from types import MappingProxyType

    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    sim = InMemoryResponseSimulator()
    mgr = Stage8FeedbackManager()

    # C1: Contradictory GateDecision (approved vs status) must raise ResponsePolicyViolationError
    r2_lab = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_c1_r2",
            tier="R2",
            environment="lab",
            asset_criticality="low",
            target_host_token="host_lab_c1_01",
            requested_actions=["apply_rate_limit"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    audit_before = len(sim.audit_trail)
    for bad_status in ("DENIED_BY_DEFAULT", "REJECTED_BY_ANALYST", "UNKNOWN_STATUS"):
        with pytest.raises(ResponsePolicyViolationError):
            sim.execute_plan(
                r2_lab,
                GateDecision(
                    plan_id=r2_lab.plan_id,
                    incident_id=r2_lab.incident_id,
                    approved=True,
                    status=bad_status,  # type: ignore[arg-type]
                    reviewer_id="soc_lead_01",
                    reason="Contradictory approved=True",
                    decided_at_utc="2026-11-21T17:50:00Z",
                ),
            )
    for approved_status in ("APPROVED_BY_POLICY", "APPROVED_BY_HUMAN"):
        with pytest.raises(ResponsePolicyViolationError):
            sim.execute_plan(
                r2_lab,
                GateDecision(
                    plan_id=r2_lab.plan_id,
                    incident_id=r2_lab.incident_id,
                    approved=False,
                    status=approved_status,  # type: ignore[arg-type]
                    reviewer_id="soc_lead_01",
                    reason="Contradictory approved=False",
                    decided_at_utc="2026-11-21T17:50:00Z",
                ),
            )
    assert len(sim.active_containments) == 0
    assert len(sim.audit_trail) == audit_before

    # C2: Mismatched incident_id between ResponsePlan and GateDecision
    r3_prod = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_c2_real_01",
            tier="R3",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_c2_01",
            requested_actions=["propose_firewall_block"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    with pytest.raises(ResponsePolicyViolationError):
        sim.execute_plan(
            r3_prod,
            GateDecision(
                plan_id=r3_prod.plan_id,
                incident_id="inc_DIFFERENT_99",
                approved=True,
                status="APPROVED_BY_HUMAN",
                reviewer_id="soc_lead_01",
                reason="Mismatched incident_id",
                decided_at_utc="2026-11-21T17:50:00Z",
            ),
        )
    assert len(sim.active_containments) == 0
    assert len(sim.audit_trail) == audit_before

    # C3: <prefix>::<ipv6> bypasses closed
    c3_ipv6_vectors = (
        "ip::1",
        "host::1",
        "src::dead:beef",
        "ipv6::1",
        "host::fe80::1",
        "ip::2001:db8::1",
        "src::2001:db8:85a3::8a2e:370:7334",
        "ip::1/64",
        "host::fe80::1%eth0",
    )
    for vec in c3_ipv6_vectors:
        with pytest.raises(ScopePolicyViolationError):
            validate_week10_payload({"note": vec}, strict=True)

    # C4: Valid metadata formats produce zero false positives
    c4_valid_metadata = (
        "2026-11-21T17:04:04Z",
        "2026-11-21T17:04:04+00:00",
        "2026-11-21T17:04:04.123456Z",
        "T1071.001",
        "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256",
        "src_host_window:host_504e095eefbf7d9b:13",
        "ja4:t13d1516h2_8daaf6152771_e5627efa2ab1",
        "00:11:22:33:44:55",
        "12:34:56",
        "mlkem768x25519-sha256,curve25519-sha256",
        {
            "ech_observed": True,
            "ja4": "t13d1516h2_8daaf6152771_e5627efa2ab1",
            "hassh": "ec72765d03b7f2321d38ef3f1d0d4d61",
        },
    )
    for item in c4_valid_metadata:
        sanitized, audit_res = validate_week10_payload({"metadata": item}, strict=True)
        assert audit_res.passed is True
        assert sanitized["metadata"] == item

    # C7b: Automated reviewer IDs (digit-suffixed & compound) rejected; human IDs allowed
    for bad_reviewer in (
        "system1",
        "bot1",
        "agent1",
        "ai_agent",
        "llm_bot",
        "daemon_bot",
        "System-1",
        "my_self_train_01",
        "auto-relabel",
    ):
        assert is_human_reviewer_id(bad_reviewer) is False
        dec = gate.evaluate(
            r3_prod,
            ApprovalDecision(decision="approve", approved=True, reviewer_id=bad_reviewer),
        )
        assert dec.approved is False
        assert dec.status == "DENIED_BY_DEFAULT"
        with pytest.raises(AutonomousTrainingForbiddenError):
            mgr.submit_feedback(
                AnalystFeedbackRecord(
                    incident_id="inc_c7b_fb",
                    plan_id=r3_prod.plan_id,
                    recommended_tier="R3",
                    analyst_approved=True,
                    reviewer_id=bad_reviewer,
                    verified_label="true_positive",
                    action_acceptable=True,
                ),
                simulator=sim,
            )

    for valid_human in (
        "soc_lead_01",
        "autonomy_officer_01",
        "sec_analyst_02",
        "analyst_01",
        "human_reviewer_01",
        "admin_1",
    ):
        assert is_human_reviewer_id(valid_human) is True

    # Recursive immutability of ResponsePlan.crypto_policy_checks + to_dict() unfreezing
    nested_checks = {
        "CR-01_deprecated_tls": True,
        "nested_policy": {"ciphers": ["TLS_AES_256_GCM_SHA384"], "meta": {"strict": True}},
    }
    plan_nested = ResponsePlan(
        plan_id="plan_nested_01",
        incident_id="inc_nested_01",
        tier="R4",
        tier_name="Cryptographic / Configuration Remediation Proposal",
        environment="production",
        asset_criticality="high",
        target_host_token="host_nested_01",
        target_host_tokens=("host_nested_01",),
        selected_actions=("generate_tls_ssh_config_remediation_diff",),
        allowed_actions=("generate_tls_ssh_config_remediation_diff",),
        network_change_allowed=False,
        human_approval_required=True,
        auto_expire_minutes=None,
        rollback_required=True,
        retest_after_action=True,
        automatic_production_merge_allowed=False,
        policy_status="REQUIRES_HUMAN_APPROVAL",
        denial_reasons=(),
        rationale="Nested check",
        crypto_policy_checks=nested_checks,
    )
    nested_checks["nested_policy"]["ciphers"].append("TAMPERED")
    assert isinstance(plan_nested.crypto_policy_checks, MappingProxyType)
    assert isinstance(plan_nested.crypto_policy_checks["nested_policy"], MappingProxyType)
    assert isinstance(plan_nested.crypto_policy_checks["nested_policy"]["ciphers"], tuple)
    assert plan_nested.crypto_policy_checks["nested_policy"]["ciphers"] == ("TLS_AES_256_GCM_SHA384",)
    with pytest.raises(TypeError):
        plan_nested.crypto_policy_checks["nested_policy"]["strict"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        plan_nested.crypto_policy_checks["nested_policy"]["meta"]["strict"] = False  # type: ignore[index]

    unfrozen = plan_nested.to_dict()["crypto_policy_checks"]
    assert isinstance(unfrozen, dict)
    assert isinstance(unfrozen["nested_policy"], dict)
    assert isinstance(unfrozen["nested_policy"]["ciphers"], list)
    assert json.loads(plan_nested.to_json())["crypto_policy_checks"] == unfrozen

    # Defense-in-depth in HumanApprovalGate.evaluate() on unlisted or empty selected_actions
    forged_unlisted = ResponsePlan(
        plan_id="plan_forged_act_01",
        incident_id="inc_forged_act_01",
        tier="R3",
        tier_name="Hard Containment Proposal",
        environment="production",
        asset_criticality="high",
        target_host_token="host_prod_01",
        target_host_tokens=("host_prod_01",),
        selected_actions=("arbitrary_os_wipe",),
        allowed_actions=("propose_firewall_block",),
        network_change_allowed=True,
        human_approval_required=True,
        auto_expire_minutes=60,
        rollback_required=True,
        retest_after_action=True,
        automatic_production_merge_allowed=None,
        policy_status="REQUIRES_HUMAN_APPROVAL",
        denial_reasons=(),
        rationale="Forged unlisted action",
        verified_malicious=True,
        corroborating_channels=("rule", "classifier"),
    )
    gate_unlisted = gate.evaluate(
        forged_unlisted,
        ApprovalDecision(decision="approve", approved=True, reviewer_id="soc_lead_01"),
    )
    assert gate_unlisted.approved is False
    assert gate_unlisted.status == "DENIED_BY_DEFAULT"

    forged_empty = ResponsePlan(
        plan_id="plan_forged_empty_01",
        incident_id="inc_forged_empty_01",
        tier="R2",
        tier_name="Soft Containment",
        environment="lab",
        asset_criticality="low",
        target_host_token="host_lab_01",
        target_host_tokens=("host_lab_01",),
        selected_actions=(),
        allowed_actions=("apply_rate_limit",),
        network_change_allowed=True,
        human_approval_required=False,
        auto_expire_minutes=30,
        rollback_required=True,
        retest_after_action=False,
        automatic_production_merge_allowed=None,
        policy_status="ALLOWED_BOUNDED_LAB",
        denial_reasons=(),
        rationale="Forged empty action",
        verified_malicious=True,
        corroborating_channels=("rule", "classifier"),
    )
    gate_empty = gate.evaluate(forged_empty)
    assert gate_empty.approved is False
    assert gate_empty.status == "DENIED_BY_DEFAULT"


def test_iteration4_edge_hardening_c3_edge_c4_c7b_edge_simulator_and_cli(tmp_path: Path) -> None:
    planner = ResponsePlanner()
    gate = HumanApprovalGate()
    sim = InMemoryResponseSimulator()
    mgr = Stage8FeedbackManager()

    # C3_edge: multi-colon, triple-colon, digit/dotted/query-delimited IPv6 smuggling closed
    c3_edge_vectors = (
        "label:::1",
        "tag:ip::fe80::1",
        "host:sub::1",
        "1host::fe80::1",
        "0g::1",
        "a.g::fe80::1",
        "123z::1",
        "ip::1?q=1",
        "host::sub::1",
        ":::1",
    )
    for vec in c3_edge_vectors:
        with pytest.raises(ScopePolicyViolationError):
            validate_week10_payload({"note": vec}, strict=True)

    # C4: all 15+ valid metadata formats produce zero false positives
    c4_valid_metadata = (
        "2026-11-21T17:04:04Z",
        "2026-11-21T17:04:04+00:00",
        "2026-11-21T17:04:04.123456Z",
        "T1071.001",
        "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256",
        "src_host_window:host_504e095eefbf7d9b:13",
        "ja4:t13d1516h2_8daaf6152771_e5627efa2ab1",
        "00:11:22:33:44:55",
        "12:34:56",
        "mlkem768x25519-sha256,curve25519-sha256",
        {
            "ech_observed": True,
            "ja4": "t13d1516h2_8daaf6152771_e5627efa2ab1",
            "hassh": "ec72765d03b7f2321d38ef3f1d0d4d61",
        },
        "hassh:ec72765d03b7f2321d38ef3f1d0d4d61",
        "hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com",
        "TLSv1.3:AES_256_GCM",
        "window_id:2026-11-21T18:00:00Z",
    )
    for item in c4_valid_metadata:
        sanitized, audit_res = validate_week10_payload({"metadata": item}, strict=True)
        assert audit_res.passed is True
        assert sanitized["metadata"] == item

    # C7b_edge: non-[_-] separators, camelCase, and chatbot/relabeler/autopilot rejected; valid humans preserved
    r3_prod = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_c7b_edge_01",
            tier="R3",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_c7b_01",
            requested_actions=["propose_firewall_block"],
            verified_malicious=True,
            corroborating_channels=["rule", "classifier"],
        )
    )
    for bad_reviewer in (
        "bot.1",
        "system:1",
        "agent/2",
        "aiAgent",
        "llmBot",
        "chatbot_01",
        "relabeler",
        "autopilot",
    ):
        assert is_human_reviewer_id(bad_reviewer) is False
        dec = gate.evaluate(
            r3_prod,
            ApprovalDecision(decision="approve", approved=True, reviewer_id=bad_reviewer),
        )
        assert dec.approved is False
        assert dec.status == "DENIED_BY_DEFAULT"
        with pytest.raises(AutonomousTrainingForbiddenError):
            mgr.submit_feedback(
                AnalystFeedbackRecord(
                    incident_id="inc_c7b_edge_fb",
                    plan_id=r3_prod.plan_id,
                    recommended_tier="R3",
                    analyst_approved=True,
                    reviewer_id=bad_reviewer,
                    verified_label="true_positive",
                    action_acceptable=True,
                ),
                simulator=sim,
            )

    for valid_human in (
        "soc_lead_01",
        "autonomy_officer_01",
        "automata_eng_01",
        "sec_analyst_02",
        "analyst_01",
        "human_reviewer_01",
        "admin_1",
    ):
        assert is_human_reviewer_id(valid_human) is True

    # Simulator bool rejection in __init__() and advance_time()
    with pytest.raises(ValueError):
        InMemoryResponseSimulator(initial_minute=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sim.advance_time(True)  # type: ignore[arg-type]
    assert sim.current_minute == 0.0

    # RemediationDiff copy isolation in SimulatedActionRecord
    r4_plan = planner.plan_from_request(
        ActionRequest(
            incident_id="inc_r4_copy_iso",
            tier="R4",
            environment="production",
            asset_criticality="high",
            target_host_token="host_prod_r4_iso",
            requested_actions=["generate_tls_ssh_config_remediation_diff"],
            policy_anomaly_only=True,
            crypto_policy_checks={"CR-01_deprecated_tls": True},
        )
    )
    rec_r4 = sim.execute_plan(
        r4_plan,
        gate.evaluate(
            r4_plan,
            ApprovalDecision(decision="approve", approved=True, reviewer_id="soc_lead_01"),
        ),
    )
    rec_diff = rec_r4.remediation_diffs[0]
    rec_diff.finding_codes.append("TAMPERED_CODE")
    rec_diff.proposed_config_summary["tls_min_version"] = "SSLv3"
    internal_diff = sim.remediation_diffs[0]
    assert "TAMPERED_CODE" not in internal_diff.finding_codes
    assert internal_diff.proposed_config_summary["tls_min_version"] == "TLSv1.3"

    # CLI reject + record-feedback aligns action_acceptable=False
    exp_dir = tmp_path / "reject_fb_export"
    exit_code = demo_main(
        [
            "--non-interactive",
            "--week9-report",
            str(WEEK9_REPORT_PATH),
            "--approval-decision",
            "reject",
            "--record-feedback",
            "--max-incidents",
            "2",
            "--export-dir",
            str(exp_dir),
        ]
    )
    assert exit_code == 0
    fb_lines = [
        json.loads(line)
        for line in (exp_dir / "stage8_feedback.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(fb_lines) >= 1
    assert all(row["verified_label"] == "false_positive" for row in fb_lines)
    assert all(row["action_acceptable"] is False for row in fb_lines)
