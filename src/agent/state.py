"""
Typed agent state and Stage 1-5 investigation contracts for Topic 04 Week 9.

Supports pure-Python deterministic LangGraph-style state transitions without
requiring external `langgraph` or `langchain` packages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

from src.hunting.schemas import (
    AttackMapping,
    IncidentCluster,
    IncidentEvidenceGraph,
    sanitize_json_value,
)


@dataclass(frozen=True)
class StageTrace:
    """Auditable execution trace for a single agent pipeline stage."""

    stage_name: str
    status: Literal["completed", "blocked", "skipped"]
    started_at_utc: str
    completed_at_utc: str
    inputs_summary: Dict[str, Any] = field(default_factory=dict)
    outputs_summary: Dict[str, Any] = field(default_factory=dict)
    guardrail_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "stage_name": self.stage_name,
                "status": self.status,
                "started_at_utc": self.started_at_utc,
                "completed_at_utc": self.completed_at_utc,
                "inputs_summary": self.inputs_summary,
                "outputs_summary": self.outputs_summary,
                "guardrail_notes": self.guardrail_notes,
            }
        )


@dataclass(frozen=True)
class HypothesisCheck:
    """Individual supporting or disconfirming check inside Stage 3 Hypothesis."""

    check_name: str
    direction: Literal["supporting", "disconfirming"]
    passed: bool
    detail: str
    observed_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "check_name": self.check_name,
                "direction": self.direction,
                "passed": self.passed,
                "detail": self.detail,
                "observed_value": self.observed_value,
            }
        )


@dataclass(frozen=True)
class Hypothesis:
    """
    Stage 3 Hypothesis object enforcing Section 15 guardrail:
    every hypothesis MUST contain at least one supporting check AND at least one
    disconfirming check.
    """

    hypothesis_id: str
    title: str
    category: Literal["threat", "benign_alternative", "policy_misconfiguration"]
    plausibility: Literal["low", "medium", "high"]
    supporting_checks: List[HypothesisCheck]
    disconfirming_checks: List[HypothesisCheck]
    outcome: Literal["supported", "weakened", "refuted"]
    summary: str

    def __post_init__(self) -> None:
        if not self.supporting_checks:
            raise ValueError(f"Hypothesis '{self.hypothesis_id}' requires >= 1 supporting check.")
        if not self.disconfirming_checks:
            raise ValueError(f"Hypothesis '{self.hypothesis_id}' requires >= 1 disconfirming check.")

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "hypothesis_id": self.hypothesis_id,
                "title": self.title,
                "category": self.category,
                "plausibility": self.plausibility,
                "supporting_checks": [c.to_dict() for c in self.supporting_checks],
                "disconfirming_checks": [c.to_dict() for c in self.disconfirming_checks],
                "outcome": self.outcome,
                "summary": self.summary,
            }
        )


@dataclass(frozen=True)
class VerificationResult:
    """Stage 4 deterministic verification outcome (SR-04, CR-03, CR-05, Sec 15 Stage 4)."""

    verified_malicious: bool
    policy_anomaly_only: bool
    severity: Literal["info", "low", "medium", "high", "critical"]
    confidence: float
    corroborating_channels: List[str]
    deterministic_checks: Dict[str, bool]
    data_quality: Dict[str, Any]
    blocked_escalations: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "verified_malicious": self.verified_malicious,
                "policy_anomaly_only": self.policy_anomaly_only,
                "severity": self.severity,
                "confidence": float(max(0.0, min(1.0, self.confidence))),
                "corroborating_channels": self.corroborating_channels,
                "deterministic_checks": self.deterministic_checks,
                "data_quality": self.data_quality,
                "blocked_escalations": self.blocked_escalations,
                "rationale": self.rationale,
            }
        )


@dataclass(frozen=True)
class PolicyRecommendation:
    """Read-only R0-R4 response tier recommendation grounded in response_policy_tiers.json."""

    recommended_tier: Literal["R0", "R1", "R2", "R3", "R4"]
    tier_name: str
    environment: str
    asset_criticality: str
    allowed_actions: List[str]
    proposed_actions: List[str]
    network_change_allowed: bool
    human_approval_required: bool
    auto_expire_minutes: Optional[int]
    rollback_required: bool
    retest_after_action: bool
    automatic_production_merge_allowed: Optional[bool]
    rationale: str
    policy_version: str = "1.0.0"
    default_posture: str = "DENY_BY_DEFAULT"
    execution_status: str = "PROPOSED_ONLY_NOT_EXECUTED"
    policy_constraints_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "recommended_tier": self.recommended_tier,
                "tier_name": self.tier_name,
                "environment": self.environment,
                "asset_criticality": self.asset_criticality,
                "allowed_actions": self.allowed_actions,
                "proposed_actions": self.proposed_actions,
                "network_change_allowed": self.network_change_allowed,
                "human_approval_required": self.human_approval_required,
                "auto_expire_minutes": self.auto_expire_minutes,
                "rollback_required": self.rollback_required,
                "retest_after_action": self.retest_after_action,
                "automatic_production_merge_allowed": self.automatic_production_merge_allowed,
                "rationale": self.rationale,
                "policy_version": self.policy_version,
                "default_posture": self.default_posture,
                "execution_status": self.execution_status,
                "policy_constraints_applied": self.policy_constraints_applied,
            }
        )


class HuntingAgentState(TypedDict, total=False):
    """Mutable dict state flowing through DeterministicWorkflowGraph Stages 1-5."""

    incident_cluster: IncidentCluster
    environment: str
    asset_criticality: str
    policy_audit: Dict[str, Any]
    triage_summary: Dict[str, Any]
    enrichment: Dict[str, Any]
    hypotheses: List[Hypothesis]
    verification: VerificationResult
    attack_mappings: List[AttackMapping]
    unmapped_rationale: Optional[str]
    evidence_graph: IncidentEvidenceGraph
    policy_recommendation: PolicyRecommendation
    narrative_result: Dict[str, Any]
    stage_traces: List[StageTrace]
    errors: List[str]


@dataclass(frozen=True)
class AgentInvestigationResult:
    """Complete Week 9 Threat-Hunting Agent investigation output for one incident."""

    incident_id: str
    evidence_graph: IncidentEvidenceGraph
    triage_summary: Dict[str, Any]
    enrichment: Dict[str, Any]
    hypotheses: List[Hypothesis]
    verification: VerificationResult
    attack_mappings: List[AttackMapping]
    unmapped_rationale: Optional[str]
    policy_recommendation: PolicyRecommendation
    narrative: Dict[str, Any]
    stage_traces: List[StageTrace]
    audit_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "incident_id": self.incident_id,
                "triage_summary": self.triage_summary,
                "enrichment": self.enrichment,
                "hypotheses": [h.to_dict() for h in self.hypotheses],
                "verification": self.verification.to_dict(),
                "attack_mappings": [m.to_dict() for m in self.attack_mappings],
                "unmapped_rationale": self.unmapped_rationale,
                "policy_recommendation": self.policy_recommendation.to_dict(),
                "narrative": self.narrative,
                "stage_traces": [t.to_dict() for t in self.stage_traces],
                "audit_summary": self.audit_summary,
                "evidence_graph": self.evidence_graph.to_dict(),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)
