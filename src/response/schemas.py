"""
Typed data contracts, status types, and exceptions for Topic 04 Week 10
Response Planning, Pure In-Memory Simulation, Stage 7 Re-test, and Stage 8 Feedback.

All serialized structures enforce NaN-free JSON safety (`sanitize_json_value`)
and pass ScopePolicyGuard checks (`SR-01`, `SR-06`, `SR-07`, `CR-06`).
"""

from __future__ import annotations

import copy
import ipaddress
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from src.agent.policy_guard import (
    IPV4_REGEX,
    PolicyAuditResult,
    ScopePolicyGuard,
    ScopePolicyViolationError,
)
from src.hunting.schemas import sanitize_json_value

EXTRA_L7_KEY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|_)password$", re.IGNORECASE),
    re.compile(r"^sni_plaintext$", re.IGNORECASE),
)

IPV6_CANDIDATE_REGEX = re.compile(
    r"(?<![0-9a-zA-Z:])([0-9a-fA-F:]*:[0-9a-fA-F:]*:[0-9a-fA-F:.]*[0-9a-fA-F:])(?![0-9a-zA-Z:])"
)
_IPV6_TOKEN_SPLIT_REGEX = re.compile(r"[\s,\[\](){};\"'<>|=?&#]+")
_IPV6_SINGLE_PREFIX_REGEX = re.compile(r"^(?:[a-zA-Z0-9_.-]+|):(?!(?::))(.+)$")
_IPV6_DOUBLE_PREFIX_REGEX = re.compile(r"^(?:::+|[a-zA-Z0-9_.-]+::+)(.+)$")

NON_HUMAN_REVIEWER_PREFIX_REGEX = re.compile(
    r"^(?:system|auto(?:mated|mation|nomous|pilot)?|bot|chatbot|agent|policy_auto|self_train|relabel(?:er)?)\d*(?:$|[^a-zA-Z0-9])",
    re.IGNORECASE,
)
NON_HUMAN_REVIEWER_SUBSTR_REGEX = re.compile(
    r"(?:auto[_-]?relabel(?:er)?|self[_-]?train(?:ing)?|auto[_-]?train(?:ing)?|policy[_-]?auto|autopilot|chatbot)|"
    r"(?:^|[^a-zA-Z0-9])(?:system|auto(?:mated|mation|nomous|pilot)?|bot|chatbot|agent|policy_auto|self_train|relabel(?:er)?)\d*(?:$|[^a-zA-Z0-9])",
    re.IGNORECASE,
)
NON_HUMAN_REVIEWER_CAMEL_REGEX = re.compile(
    r"(?<=[a-z0-9])(?:System|Automated|Automation|Autonomous|Autopilot|Bot|ChatBot|Agent|Relabel(?:er)?|SelfTrain|AutoTrain)\d*(?:$|[^a-z0-9])"
    r"|^(?:[Ss]ystem|[Aa]uto(?:mated|mation|nomous|pilot)?|[Bb]ot|[Cc]hat[Bb]ot|[Aa]gent|[Rr]elabel(?:er)?)[A-Z]"
)

_DEFAULT_POLICY_GUARD: Optional[ScopePolicyGuard] = None


def _get_default_policy_guard() -> ScopePolicyGuard:
    global _DEFAULT_POLICY_GUARD
    if _DEFAULT_POLICY_GUARD is None:
        _DEFAULT_POLICY_GUARD = ScopePolicyGuard()
    return _DEFAULT_POLICY_GUARD


def is_human_reviewer_id(reviewer_id: Optional[str]) -> bool:
    """Return True only for non-empty, non-automated human reviewer identifiers."""
    if not isinstance(reviewer_id, str):
        return False
    cleaned = reviewer_id.strip()
    if not cleaned:
        return False
    if NON_HUMAN_REVIEWER_PREFIX_REGEX.search(cleaned):
        return False
    if NON_HUMAN_REVIEWER_SUBSTR_REGEX.search(cleaned):
        return False
    if NON_HUMAN_REVIEWER_CAMEL_REGEX.search(cleaned):
        return False
    return True


def _try_parse_ipv6(candidate: str) -> Optional[str]:
    if ":" not in candidate:
        return None
    # Strip optional /prefix or %zone suffix
    base = candidate.split("/", 1)[0].split("%", 1)[0]
    if base.count(":") < 2:
        return None
    try:
        ipaddress.IPv6Address(base)
        return candidate
    except ValueError:
        return None


def contains_ipv6_literal(text: str) -> Optional[str]:
    """Detect raw IPv6 literals while avoiding false positives on ISO timestamps or prose colons."""
    if not text or ":" not in text:
        return None
    for match in IPV6_CANDIDATE_REGEX.finditer(text):
        parsed = _try_parse_ipv6(match.group(1))
        if parsed is not None:
            return parsed
    for token in _IPV6_TOKEN_SPLIT_REGEX.split(text):
        if ":" not in token:
            continue
        cleaned = token.strip(".")
        if not cleaned:
            continue
        current = cleaned
        for _ in range(16):
            parsed = _try_parse_ipv6(current)
            if parsed is not None:
                return parsed
            dbl_match = _IPV6_DOUBLE_PREFIX_REGEX.match(current)
            if dbl_match:
                suffix = dbl_match.group(1)
                for candidate in (suffix, f"::{suffix}"):
                    parsed = _try_parse_ipv6(candidate)
                    if parsed is not None:
                        return parsed
                current = suffix
                continue
            pref_match = _IPV6_SINGLE_PREFIX_REGEX.match(current)
            if pref_match:
                current = pref_match.group(1)
                continue
            break
    return None


def validate_week10_payload(
    payload: Any,
    strict: bool = True,
    strip_ground_truth: bool = True,
    policy_guard: Optional[ScopePolicyGuard] = None,
) -> Tuple[Any, PolicyAuditResult]:
    """
    Wrap `ScopePolicyGuard.validate_and_sanitize_payload` and enforce Week 10 privacy guards:
    - Extra L7 keys (`(?:^|_)password$`, `^sni_plaintext$`) preserving allowed CR-03/CR-05 metadata.
    - Raw IPv4 (`IPV4_REGEX`) detection across ALL string values and keys.
    - Raw IPv6 literal detection across ALL string values and keys.
    - Secret pattern detection across ALL string keys as well as values.
    """
    guard = policy_guard or _get_default_policy_guard()
    sanitized, base_audit = guard.validate_and_sanitize_payload(
        payload, strict=strict, strip_ground_truth=strip_ground_truth
    )

    extra_violations: List[str] = []

    def _scan_text(text: str, path: str, *, is_key: bool) -> None:
        if is_key:
            for pat in EXTRA_L7_KEY_PATTERNS:
                if pat.search(text.strip()):
                    extra_violations.append(
                        f"{path}: forbidden L7 key '{text}' matches '{pat.pattern}' (SR-01)."
                    )
                    break
            secret_hits = guard._detect_secrets(text)
            if secret_hits:
                extra_violations.append(
                    f"{path}: prohibited secret/key material in key ({', '.join(secret_hits)}) (CR-06)."
                )
        if IPV4_REGEX.search(text):
            extra_violations.append(
                f"{path}: raw IPv4 address detected in '{text}' (SR-01/CR-06)."
            )
        ipv6_hit = contains_ipv6_literal(text)
        if ipv6_hit is not None:
            extra_violations.append(
                f"{path}: raw IPv6 literal '{ipv6_hit}' detected (SR-01/CR-06)."
            )

    def _walk_extra(obj: Any, path: str = "root") -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                k_str = str(k)
                _scan_text(k_str, f"{path}.<key:{k_str}>", is_key=True)
                _walk_extra(v, f"{path}.{k_str}")
            return
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _walk_extra(item, f"{path}[{i}]")
            return
        if isinstance(obj, (set, frozenset)):
            for i, item in enumerate(sorted(obj, key=lambda x: (type(x).__qualname__, repr(x)))):
                _walk_extra(item, f"{path}[{i}]")
            return
        if isinstance(obj, str):
            _scan_text(obj, path, is_key=False)

    _walk_extra(sanitized)

    if not extra_violations:
        return sanitized, base_audit

    all_violations = list(base_audit.violations) + extra_violations
    merged_audit = PolicyAuditResult(
        passed=False,
        violations=all_violations,
        redactions_applied=list(base_audit.redactions_applied),
        stripped_ground_truth_columns=list(base_audit.stripped_ground_truth_columns),
        l5_l6_fields_present=list(base_audit.l5_l6_fields_present),
        compliance_anchors=list(base_audit.compliance_anchors),
    )
    if strict:
        raise ScopePolicyViolationError(f"Week 10 privacy/policy guard rejected payload: {all_violations}")
    return sanitized, merged_audit

TierId = Literal["R0", "R1", "R2", "R3", "R4"]
EnvironmentType = Literal["lab", "staging", "production"]
AssetCriticality = Literal["low", "medium", "high", "critical"]

PlanPolicyStatus = Literal[
    "ALLOWED_NON_NETWORK",
    "ALLOWED_BOUNDED_LAB",
    "REQUIRES_HUMAN_APPROVAL",
    "DENIED_BY_DEFAULT",
]

GateStatus = Literal[
    "APPROVED_BY_POLICY",
    "APPROVED_BY_HUMAN",
    "DENIED_BY_DEFAULT",
    "REJECTED_BY_ANALYST",
]

ContainmentStatus = Literal["ACTIVE", "ROLLED_BACK", "EXPIRED"]

ExecutionOutcome = Literal[
    "OBSERVED_ONLY",
    "TICKET_CREATED",
    "EXECUTED_SIMULATED",
    "PROPOSED_REMEDIATION_DIFF",
    "DENIED_BY_DEFAULT",
    "REJECTED_BY_ANALYST",
]

RetestOutcome = Literal[
    "CONTAINMENT_VERIFIED",
    "POLICY_CONFORMANT",
    "FALSE_POSITIVE_DETECTED",
    "THREAT_PERSISTS_ESCALATE",
    "NO_RETEST_REQUIRED",
]

VerifiedLabelType = Literal[
    "true_positive",
    "false_positive",
    "benign_policy_drift",
    "needs_more_data",
]


class ResponsePolicyViolationError(ValueError):
    """Raised when a response plan, action, or simulation violates Topic 04 policy."""


class AutonomousTrainingForbiddenError(ResponsePolicyViolationError):
    """Raised when Stage 8 receives an unapproved or autonomous relabel/self-training request."""


@dataclass(frozen=True)
class ActionRequest:
    """Explicit response action request (used for boundary/adversarial testing and direct planning)."""

    incident_id: str
    tier: str
    environment: str
    asset_criticality: str
    target_host_token: str
    requested_actions: List[str] = field(default_factory=list)
    verified_malicious: bool = False
    policy_anomaly_only: bool = False
    corroborating_channels: List[str] = field(default_factory=list)
    severity: str = "medium"
    max_s_total: float = 0.0
    alerted_flow_count: int = 0
    attack_technique_ids: List[str] = field(default_factory=list)
    crypto_policy_checks: Dict[str, Any] = field(default_factory=dict)
    ech_only: bool = False
    fingerprint_only: bool = False
    request_production_auto_merge: bool = False
    extra_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalDecision:
    """Human-in-the-loop approval or rejection record (`SR-07`)."""

    decision: Literal["approve", "reject", "pending"] = "pending"
    approved: bool = False
    reviewer_id: str = ""
    reason: str = ""
    decided_at_utc: Optional[str] = None
    ticket_ref: Optional[str] = None
    request_production_auto_merge: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "decision": self.decision,
                "approved": self.approved,
                "reviewer_id": self.reviewer_id,
                "reason": self.reason,
                "decided_at_utc": self.decided_at_utc,
                "ticket_ref": self.ticket_ref,
                "request_production_auto_merge": self.request_production_auto_merge,
            }
        )


@dataclass(frozen=True)
class GateDecision:
    """Output of `HumanApprovalGate.evaluate()`."""

    plan_id: str
    incident_id: str
    approved: bool
    status: GateStatus
    reviewer_id: Optional[str]
    reason: str
    decided_at_utc: str
    policy_anchors: Tuple[str, ...] = ("SR-06", "SR-07")

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_anchors", tuple(str(x) for x in self.policy_anchors))

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "plan_id": self.plan_id,
                "incident_id": self.incident_id,
                "approved": self.approved,
                "status": self.status,
                "reviewer_id": self.reviewer_id,
                "reason": self.reason,
                "decided_at_utc": self.decided_at_utc,
                "policy_anchors": list(self.policy_anchors),
            }
        )


def _freeze_value(value: Any) -> Any:
    """Recursively freeze dicts/lists/sets into MappingProxyType/tuple/frozenset."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(v) for v in value)
    return copy.deepcopy(value)


def _unfreeze_value(value: Any) -> Any:
    """Recursively convert frozen MappingProxyType/tuple/frozenset back to dict/list."""
    if isinstance(value, Mapping):
        return {k: _unfreeze_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unfreeze_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [
            _unfreeze_value(v)
            for v in sorted(value, key=lambda x: (type(x).__qualname__, repr(x)))
        ]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ResponsePlan:
    """Policy-validated Stage 6a response plan produced by `ResponsePlanner`."""

    plan_id: str
    incident_id: str
    tier: TierId
    tier_name: str
    environment: str
    asset_criticality: str
    target_host_token: str
    target_host_tokens: Tuple[str, ...]
    selected_actions: Tuple[str, ...]
    allowed_actions: Tuple[str, ...]
    network_change_allowed: bool
    human_approval_required: bool
    auto_expire_minutes: Optional[int]
    rollback_required: bool
    retest_after_action: bool
    automatic_production_merge_allowed: Optional[bool]
    policy_status: PlanPolicyStatus
    denial_reasons: Tuple[str, ...]
    rationale: str
    policy_version: str = "1.0.0"
    default_posture: str = "DENY_BY_DEFAULT"
    verified_malicious: bool = False
    policy_anomaly_only: bool = False
    severity: str = "info"
    corroborating_channels: Tuple[str, ...] = ()
    attack_technique_ids: Tuple[str, ...] = ()
    crypto_policy_checks: Mapping[str, Any] = field(default_factory=dict)
    max_s_total: float = 0.0
    alerted_flow_count: int = 0

    def __post_init__(self) -> None:
        for seq_field in (
            "target_host_tokens",
            "selected_actions",
            "allowed_actions",
            "denial_reasons",
            "corroborating_channels",
            "attack_technique_ids",
        ):
            val = getattr(self, seq_field)
            object.__setattr__(self, seq_field, tuple(str(x) for x in val))
        object.__setattr__(
            self,
            "crypto_policy_checks",
            _freeze_value(dict(self.crypto_policy_checks or {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "plan_id": self.plan_id,
                "incident_id": self.incident_id,
                "tier": self.tier,
                "tier_name": self.tier_name,
                "environment": self.environment,
                "asset_criticality": self.asset_criticality,
                "target_host_token": self.target_host_token,
                "target_host_tokens": list(self.target_host_tokens),
                "selected_actions": list(self.selected_actions),
                "allowed_actions": list(self.allowed_actions),
                "network_change_allowed": self.network_change_allowed,
                "human_approval_required": self.human_approval_required,
                "auto_expire_minutes": self.auto_expire_minutes,
                "rollback_required": self.rollback_required,
                "retest_after_action": self.retest_after_action,
                "automatic_production_merge_allowed": self.automatic_production_merge_allowed,
                "policy_status": self.policy_status,
                "denial_reasons": list(self.denial_reasons),
                "rationale": self.rationale,
                "policy_version": self.policy_version,
                "default_posture": self.default_posture,
                "verified_malicious": self.verified_malicious,
                "policy_anomaly_only": self.policy_anomaly_only,
                "severity": self.severity,
                "corroborating_channels": list(self.corroborating_channels),
                "attack_technique_ids": list(self.attack_technique_ids),
                "crypto_policy_checks": _unfreeze_value(self.crypto_policy_checks),
                "max_s_total": float(self.max_s_total),
                "alerted_flow_count": int(self.alerted_flow_count),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


@dataclass(frozen=True)
class ContainmentEntry:
    """Pure in-memory simulated containment/rate-limit entry (zero OS side effects)."""

    entry_id: str
    rollback_token: str
    incident_id: str
    plan_id: str
    tier: TierId
    action: str
    environment: str
    asset_criticality: str
    target_host_token: str
    scope: str
    created_at_utc: str
    created_minute: float
    ttl_minutes: int
    expires_at_minute: float
    status: ContainmentStatus = "ACTIVE"
    reviewer_id: Optional[str] = None
    rolled_back_at_minute: Optional[float] = None
    rollback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "entry_id": self.entry_id,
                "rollback_token": self.rollback_token,
                "incident_id": self.incident_id,
                "plan_id": self.plan_id,
                "tier": self.tier,
                "action": self.action,
                "environment": self.environment,
                "asset_criticality": self.asset_criticality,
                "target_host_token": self.target_host_token,
                "scope": self.scope,
                "created_at_utc": self.created_at_utc,
                "created_minute": float(self.created_minute),
                "ttl_minutes": int(self.ttl_minutes),
                "expires_at_minute": float(self.expires_at_minute),
                "status": self.status,
                "reviewer_id": self.reviewer_id,
                "rolled_back_at_minute": self.rolled_back_at_minute,
                "rollback_reason": self.rollback_reason,
            }
        )


@dataclass(frozen=True)
class RemediationDiff:
    """Structured R4 TLS/SSH/X.509 configuration remediation diff (`automatic_production_merge_allowed=False`)."""

    diff_id: str
    rollback_token: str
    incident_id: str
    plan_id: str
    target_host_token: str
    environment: str
    policy_standard: str
    finding_codes: List[str]
    current_config_summary: Dict[str, Any]
    proposed_config_summary: Dict[str, Any]
    unified_diff_lines: List[str]
    lab_conformance_retest_status: str
    automatic_production_merge_allowed: bool = False
    applied_to_production: bool = False
    status: str = "PROPOSED_NEVER_AUTO_MERGED"

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "diff_id": self.diff_id,
                "rollback_token": self.rollback_token,
                "incident_id": self.incident_id,
                "plan_id": self.plan_id,
                "target_host_token": self.target_host_token,
                "environment": self.environment,
                "policy_standard": self.policy_standard,
                "finding_codes": list(self.finding_codes),
                "current_config_summary": dict(self.current_config_summary),
                "proposed_config_summary": dict(self.proposed_config_summary),
                "unified_diff_lines": list(self.unified_diff_lines),
                "lab_conformance_retest_status": self.lab_conformance_retest_status,
                "automatic_production_merge_allowed": self.automatic_production_merge_allowed,
                "applied_to_production": self.applied_to_production,
                "status": self.status,
            }
        )


@dataclass(frozen=True)
class SimulatedActionRecord:
    """Result of `InMemoryResponseSimulator.execute_plan()`."""

    action_record_id: str
    plan_id: str
    incident_id: str
    tier: TierId
    environment: str
    asset_criticality: str
    target_host_token: str
    outcome: ExecutionOutcome
    executed_actions: List[str]
    network_state_mutated: bool
    containment_entries: List[ContainmentEntry]
    remediation_diffs: List[RemediationDiff]
    rollback_tokens: List[str]
    gate_decision: GateDecision
    simulated_time_to_containment_ms: float
    created_at_utc: str
    logical_minute: float

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "action_record_id": self.action_record_id,
                "plan_id": self.plan_id,
                "incident_id": self.incident_id,
                "tier": self.tier,
                "environment": self.environment,
                "asset_criticality": self.asset_criticality,
                "target_host_token": self.target_host_token,
                "outcome": self.outcome,
                "executed_actions": list(self.executed_actions),
                "network_state_mutated": self.network_state_mutated,
                "containment_entries": [c.to_dict() for c in self.containment_entries],
                "remediation_diffs": [d.to_dict() for d in self.remediation_diffs],
                "rollback_tokens": list(self.rollback_tokens),
                "gate_decision": self.gate_decision.to_dict(),
                "simulated_time_to_containment_ms": float(self.simulated_time_to_containment_ms),
                "created_at_utc": self.created_at_utc,
                "logical_minute": float(self.logical_minute),
            }
        )


@dataclass(frozen=True)
class RollbackResult:
    """Outcome of an explicit or Stage-7-triggered rollback."""

    success: bool
    rollback_token: str
    entry_id: Optional[str]
    incident_id: Optional[str]
    previous_status: Optional[str]
    new_status: Optional[str]
    actor: str
    reason: str
    logical_minute: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "success": self.success,
                "rollback_token": self.rollback_token,
                "entry_id": self.entry_id,
                "incident_id": self.incident_id,
                "previous_status": self.previous_status,
                "new_status": self.new_status,
                "actor": self.actor,
                "reason": self.reason,
                "logical_minute": float(self.logical_minute),
                "timestamp_utc": self.timestamp_utc,
            }
        )


@dataclass(frozen=True)
class RetestObservation:
    """Post-action window telemetry observed by Stage 7 Re-test Verifier."""

    window_id: int
    post_max_s_total: float
    post_alerted_flow_count: int
    post_total_flow_count: int = 1
    benign_traffic_disruption_ratio: float = 0.0
    false_positive_confirmed: bool = False
    crypto_conformance_passed: Optional[bool] = None
    active_reason_codes: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "window_id": int(self.window_id),
                "post_max_s_total": float(self.post_max_s_total),
                "post_alerted_flow_count": int(self.post_alerted_flow_count),
                "post_total_flow_count": int(self.post_total_flow_count),
                "benign_traffic_disruption_ratio": float(self.benign_traffic_disruption_ratio),
                "false_positive_confirmed": self.false_positive_confirmed,
                "crypto_conformance_passed": self.crypto_conformance_passed,
                "active_reason_codes": list(self.active_reason_codes),
                "notes": self.notes,
            }
        )


@dataclass(frozen=True)
class RetestResult:
    """Stage 7 post-action re-test outcome (Section 15 Stage 7)."""

    retest_id: str
    incident_id: str
    plan_id: str
    tier: TierId
    outcome: RetestOutcome
    containment_verified: bool
    false_positive_detected: bool
    auto_rollback_triggered: bool
    rollback_results: List[RollbackResult] = field(default_factory=list)
    escalate_to_analyst: bool = False
    rationale: str = ""
    checked_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "retest_id": self.retest_id,
                "incident_id": self.incident_id,
                "plan_id": self.plan_id,
                "tier": self.tier,
                "outcome": self.outcome,
                "containment_verified": self.containment_verified,
                "false_positive_detected": self.false_positive_detected,
                "auto_rollback_triggered": self.auto_rollback_triggered,
                "rollback_results": [r.to_dict() for r in self.rollback_results],
                "escalate_to_analyst": self.escalate_to_analyst,
                "rationale": self.rationale,
                "checked_at_utc": self.checked_at_utc,
            }
        )


@dataclass(frozen=True)
class AnalystFeedbackRecord:
    """Stage 8 analyst-approved feedback intake record (`SR-07`, Section 15 Stage 8)."""

    incident_id: str
    plan_id: str
    recommended_tier: str
    analyst_approved: bool
    reviewer_id: str
    verified_label: VerifiedLabelType
    action_acceptable: bool
    notes: str = ""
    autonomous_relabel: bool = False
    self_train: bool = False
    auto_train: bool = False
    submitted_at_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "incident_id": self.incident_id,
                "plan_id": self.plan_id,
                "recommended_tier": self.recommended_tier,
                "analyst_approved": self.analyst_approved,
                "reviewer_id": self.reviewer_id,
                "verified_label": self.verified_label,
                "action_acceptable": self.action_acceptable,
                "notes": self.notes,
                "autonomous_relabel": self.autonomous_relabel,
                "self_train": self.self_train,
                "auto_train": self.auto_train,
                "submitted_at_utc": self.submitted_at_utc,
            }
        )


@dataclass(frozen=True)
class FeedbackSubmissionResult:
    """Outcome of `Stage8FeedbackManager.submit_feedback()`."""

    accepted: bool
    status: str
    incident_id: str
    reviewer_id: str
    reason: str
    record: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "accepted": self.accepted,
                "status": self.status,
                "incident_id": self.incident_id,
                "reviewer_id": self.reviewer_id,
                "reason": self.reason,
                "record": self.record,
            }
        )


@dataclass(frozen=True)
class AuditEvent:
    """Immutable-style audit trail event covering Stages 6, 7, and 8."""

    event_id: str
    sequence_no: int
    timestamp_utc: str
    logical_minute: float
    stage: str
    event_type: str
    incident_id: str
    plan_id: Optional[str]
    tier: Optional[str]
    environment: Optional[str]
    target_host_token: Optional[str]
    actor: str
    outcome: str
    rollback_token: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", copy.deepcopy(dict(self.details)))

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json_value(
            {
                "event_id": self.event_id,
                "sequence_no": int(self.sequence_no),
                "timestamp_utc": self.timestamp_utc,
                "logical_minute": float(self.logical_minute),
                "stage": self.stage,
                "event_type": self.event_type,
                "incident_id": self.incident_id,
                "plan_id": self.plan_id,
                "tier": self.tier,
                "environment": self.environment,
                "target_host_token": self.target_host_token,
                "actor": self.actor,
                "outcome": self.outcome,
                "rollback_token": self.rollback_token,
                "details": dict(self.details),
            }
        )
