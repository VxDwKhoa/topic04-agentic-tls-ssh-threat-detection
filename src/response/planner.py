"""
Policy-Constrained Response Planner & Human Approval Gate (Stage 6a, Topic 04 Week 10).

Enforces `DENY_BY_DEFAULT` from `src/response/response_policy_tiers.json`,
`ScopePolicyGuard` (`SR-01`, `SR-04`, `SR-06`, `SR-07`, `CR-03`, `CR-05`, `CR-06`),
environment/asset-criticality bounds, action allowlists, and mandatory
human-in-the-loop approval for production containment and R4 remediation diffs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from src.agent.policy_guard import (
    ScopePolicyGuard,
    ScopePolicyViolationError,
)
from src.agent.state import AgentInvestigationResult
from src.response.schemas import (
    ActionRequest,
    ApprovalDecision,
    GateDecision,
    PlanPolicyStatus,
    ResponsePlan,
    ResponsePolicyViolationError,
    TierId,
    is_human_reviewer_id,
    validate_week10_payload,
)

VALID_TIERS = frozenset({"R0", "R1", "R2", "R3", "R4"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deterministic_plan_id(incident_id: str, tier: str, env: str, crit: str, target: str, actions: Sequence[str]) -> str:
    seed = f"{incident_id}|{tier}|{env}|{crit}|{target}|{','.join(actions)}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"plan_{digest}"


class ResponsePlanner:
    """Loads `response_policy_tiers.json` and produces policy-constrained `ResponsePlan`s."""

    def __init__(
        self,
        policy_path: Optional[Union[str, Path]] = None,
        policy_guard: Optional[ScopePolicyGuard] = None,
    ) -> None:
        if policy_path is not None:
            self.policy_path = Path(policy_path)
        else:
            self.policy_path = Path(__file__).resolve().parent / "response_policy_tiers.json"

        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Response policy file not found: {self.policy_path}")

        with open(self.policy_path, "r", encoding="utf-8") as f:
            self.policy_data: Dict[str, Any] = json.load(f)

        self.policy_version: str = str(self.policy_data.get("policy_version", "1.0.0"))
        self.default_posture: str = str(self.policy_data.get("default_posture", "DENY_BY_DEFAULT"))
        if self.default_posture != "DENY_BY_DEFAULT":
            raise ResponsePolicyViolationError(
                f"Expected default_posture='DENY_BY_DEFAULT', got '{self.default_posture}'."
            )

        self.tiers_by_id: Dict[str, Dict[str, Any]] = {
            str(t["tier_id"]): dict(t) for t in self.policy_data.get("tiers", [])
        }
        for req_tier in ("R0", "R1", "R2", "R3", "R4"):
            if req_tier not in self.tiers_by_id:
                raise ValueError(f"Missing required tier '{req_tier}' in {self.policy_path}")

        self.required_metrics: List[str] = list(
            self.policy_data.get("evaluation_metrics_required", [])
        )
        self.policy_guard = policy_guard or ScopePolicyGuard()

    def _validate_target_tokens(self, primary_token: str, target_tokens: Sequence[str]) -> None:
        if not primary_token or not primary_token.strip():
            raise ScopePolicyViolationError("Target host token cannot be empty (SR-06).")
        validate_week10_payload(
            {
                "primary_host_token": primary_token,
                "target_host_tokens": list(target_tokens),
            },
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )

    def _allowed_actions_for(self, tier_id: str, environment: str) -> List[str]:
        tier_cfg = self.tiers_by_id[tier_id]
        if tier_id == "R3":
            if environment == "production":
                return list(tier_cfg.get("allowed_actions_production", []))
            return list(tier_cfg.get("allowed_actions_lab", []))
        return list(tier_cfg.get("allowed_actions", []))

    def _enforce_tier_policy(
        self,
        *,
        incident_id: str,
        tier_id: str,
        environment: str,
        asset_criticality: str,
        primary_host_token: str,
        target_host_tokens: List[str],
        verified_malicious: bool,
        policy_anomaly_only: bool,
        severity: str,
        corroborating_channels: List[str],
        attack_technique_ids: List[str],
        crypto_policy_checks: Dict[str, Any],
        max_s_total: float,
        alerted_flow_count: int,
        requested_actions: Optional[Sequence[str]] = None,
        ech_only: bool = False,
        fingerprint_only: bool = False,
        request_production_auto_merge: bool = False,
        base_rationale: str = "",
    ) -> ResponsePlan:
        ScopePolicyGuard.check_environment_scope(environment, asset_criticality)
        env = environment.strip().lower()
        crit = asset_criticality.strip().lower()
        tier = str(tier_id).strip().upper()
        if tier not in VALID_TIERS:
            raise ResponsePolicyViolationError(
                f"Unknown response tier '{tier_id}'. Allowed: {sorted(VALID_TIERS)}."
            )

        self._validate_target_tokens(primary_host_token, target_host_tokens)

        tier_cfg = self.tiers_by_id[tier]
        allowed_actions = self._allowed_actions_for(tier, env)
        if not allowed_actions:
            raise ResponsePolicyViolationError(
                f"Policy configuration has no allowed actions for tier={tier}, env={env}."
            )

        if requested_actions is not None:
            raw_actions = [str(a).strip() for a in requested_actions if str(a).strip()]
            selected_actions = list(dict.fromkeys(raw_actions))
        else:
            selected_actions = [allowed_actions[0]]

        unique_channels: List[str] = []
        seen_channels: set[str] = set()
        for ch in corroborating_channels:
            ch_clean = str(ch).strip()
            ch_key = ch_clean.lower()
            if ch_clean and ch_key not in seen_channels:
                seen_channels.add(ch_key)
                unique_channels.append(ch_clean)

        denial_reasons: List[str] = []

        # 1. Allowlist verification (DENY_BY_DEFAULT, SR-06)
        if not selected_actions:
            denial_reasons.append(
                f"Empty action list is denied under {self.default_posture} (SR-06)."
            )
        else:
            disallowed = [a for a in selected_actions if a not in allowed_actions]
            if disallowed:
                denial_reasons.append(
                    f"Action(s) {disallowed} not in allowlist {allowed_actions} for tier={tier}, env={env} ({self.default_posture})."
                )

        # 2. Tier-specific policy matrix enforcement
        network_change_allowed = bool(tier_cfg.get("network_change_allowed", False))
        auto_expire_raw = tier_cfg.get("auto_expire_minutes")
        auto_expire_minutes = int(auto_expire_raw) if auto_expire_raw is not None else None
        rollback_required = bool(tier_cfg.get("rollback_required", False))
        retest_after_action = bool(tier_cfg.get("retest_after_action", False))
        auto_merge_cfg = tier_cfg.get("automatic_production_merge_allowed")
        automatic_production_merge_allowed = (
            bool(auto_merge_cfg) if auto_merge_cfg is not None else None
        )

        if tier == "R0":
            human_approval_required = bool(tier_cfg.get("human_approval_required", False))
            network_change_allowed = False
        elif tier == "R1":
            human_approval_required = bool(tier_cfg.get("human_approval_required", False))
            network_change_allowed = False
        elif tier == "R2":
            allowed_envs = {
                str(e).strip().lower() for e in tier_cfg.get("allowed_environments", [])
            }
            human_approval_required = bool(tier_cfg.get("human_approval_required", False))
            if env not in allowed_envs:
                network_change_allowed = False
                denial_reasons.append(
                    f"R2 soft containment denied in environment='{env}'; restricted to {sorted(allowed_envs)} ({self.default_posture} / SR-06)."
                )
            if crit == "critical":
                network_change_allowed = False
                denial_reasons.append(
                    f"R2 soft containment denied on asset_criticality='critical' without escalation ({self.default_posture} / SR-06)."
                )
            if ech_only or fingerprint_only:
                network_change_allowed = False
                denial_reasons.append(
                    "CR-03/CR-05: Neutral ECH or fingerprint-only observation cannot trigger R2 containment."
                )
            elif (not verified_malicious) or len(unique_channels) < 2:
                network_change_allowed = False
                denial_reasons.append(
                    "SR-04: R2 network containment denied without verified_malicious=True and >= 2 unique corroborating channels."
                )
        elif tier == "R3":
            if env == "production":
                human_approval_required = bool(
                    tier_cfg.get("human_approval_required_in_production", True)
                )
            else:
                human_approval_required = bool(crit == "critical")
            if ech_only or fingerprint_only:
                network_change_allowed = False
                denial_reasons.append(
                    "CR-03/CR-05: Neutral ECH or fingerprint-only observation cannot trigger R3 containment."
                )
            elif (not verified_malicious) or len(unique_channels) < 2:
                network_change_allowed = False
                denial_reasons.append(
                    "SR-04: R3 hard containment denied without verified_malicious=True and >= 2 unique corroborating channels."
                )
        else:  # R4
            human_approval_required = True
            automatic_production_merge_allowed = False
            network_change_allowed = False
            if request_production_auto_merge:
                denial_reasons.append(
                    "R4 invariant violated: automatic_production_merge_allowed=False; automatic production merge is strictly prohibited."
                )

        if denial_reasons:
            policy_status: PlanPolicyStatus = "DENIED_BY_DEFAULT"
        elif human_approval_required:
            policy_status = "REQUIRES_HUMAN_APPROVAL"
        elif tier in {"R0", "R1"}:
            policy_status = "ALLOWED_NON_NETWORK"
        else:
            policy_status = "ALLOWED_BOUNDED_LAB"

        plan_id = _deterministic_plan_id(
            incident_id=incident_id,
            tier=tier,
            env=env,
            crit=crit,
            target=primary_host_token,
            actions=selected_actions,
        )

        rationale_parts = [base_rationale] if base_rationale else []
        if denial_reasons:
            rationale_parts.append("DENIED: " + " | ".join(denial_reasons))
        elif human_approval_required:
            rationale_parts.append(
                f"Plan requires explicit human approval before execution ({self.default_posture}, SR-07)."
            )
        else:
            rationale_parts.append(
                f"Plan authorized within bounded policy ({tier} in {env}, criticality={crit})."
            )

        plan = ResponsePlan(
            plan_id=plan_id,
            incident_id=incident_id,
            tier=tier,  # type: ignore[arg-type]
            tier_name=str(tier_cfg["name"]),
            environment=env,
            asset_criticality=crit,
            target_host_token=primary_host_token,
            target_host_tokens=tuple(target_host_tokens),
            selected_actions=tuple(selected_actions),
            allowed_actions=tuple(allowed_actions),
            network_change_allowed=network_change_allowed,
            human_approval_required=human_approval_required,
            auto_expire_minutes=auto_expire_minutes,
            rollback_required=rollback_required,
            retest_after_action=retest_after_action,
            automatic_production_merge_allowed=automatic_production_merge_allowed,
            policy_status=policy_status,
            denial_reasons=tuple(denial_reasons),
            rationale=" ".join(rationale_parts).strip(),
            policy_version=self.policy_version,
            default_posture=self.default_posture,
            verified_malicious=verified_malicious,
            policy_anomaly_only=policy_anomaly_only,
            severity=severity,
            corroborating_channels=tuple(unique_channels),
            attack_technique_ids=tuple(attack_technique_ids),
            crypto_policy_checks=dict(crypto_policy_checks),
            max_s_total=float(max_s_total),
            alerted_flow_count=int(alerted_flow_count),
        )
        validate_week10_payload(
            plan.to_dict(),
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )
        return plan

    def plan_from_investigation(
        self,
        investigation: Union[AgentInvestigationResult, Mapping[str, Any]],
        environment: Optional[str] = None,
        asset_criticality: Optional[str] = None,
        requested_actions: Optional[Sequence[str]] = None,
        target_host_token: Optional[str] = None,
        tier_override: Optional[str] = None,
        request_production_auto_merge: bool = False,
    ) -> ResponsePlan:
        """Build a policy-enforced `ResponsePlan` from a Week 9 `AgentInvestigationResult` or serialized dict."""
        if isinstance(investigation, AgentInvestigationResult):
            raw_dict = investigation.to_dict()
        elif isinstance(investigation, Mapping):
            raw_dict = dict(investigation)
        else:
            raise TypeError(
                f"Expected AgentInvestigationResult or Mapping, got {type(investigation).__name__}."
            )

        sanitized, _ = validate_week10_payload(
            raw_dict,
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )

        incident_id = str(sanitized.get("incident_id", "")).strip()
        if not incident_id:
            raise ResponsePolicyViolationError("Investigation payload is missing 'incident_id'.")

        rec = sanitized.get("policy_recommendation") or {}
        ver = sanitized.get("verification") or {}
        triage = sanitized.get("triage_summary") or {}
        enrichment = sanitized.get("enrichment") or {}
        mappings = sanitized.get("attack_mappings") or []

        env = str(
            environment if environment is not None else rec.get("environment", "production")
        ).strip().lower()
        crit = str(
            asset_criticality
            if asset_criticality is not None
            else rec.get("asset_criticality", "medium")
        ).strip().lower()
        tier = str(
            tier_override if tier_override is not None else rec.get("recommended_tier", "R0")
        ).strip().upper()

        primary_host = str(
            target_host_token
            if target_host_token is not None
            else triage.get("primary_host_token", "")
        ).strip()
        targets = [str(t) for t in triage.get("target_host_tokens", [])]
        if not targets and primary_host:
            targets = [primary_host]

        det_checks = ver.get("deterministic_checks") or {}
        blocked_esc = [str(b) for b in ver.get("blocked_escalations", [])]
        ech_only = any("CR-03" in b for b in blocked_esc) and not bool(ver.get("verified_malicious"))
        fp_only = any("CR-05" in b for b in blocked_esc) and not bool(ver.get("verified_malicious"))
        if not det_checks.get("ech_neutrality_cr03", True):
            ech_only = True

        return self._enforce_tier_policy(
            incident_id=incident_id,
            tier_id=tier,
            environment=env,
            asset_criticality=crit,
            primary_host_token=primary_host,
            target_host_tokens=targets,
            verified_malicious=bool(ver.get("verified_malicious", False)),
            policy_anomaly_only=bool(ver.get("policy_anomaly_only", False)),
            severity=str(ver.get("severity", "info")),
            corroborating_channels=[str(c) for c in ver.get("corroborating_channels", [])],
            attack_technique_ids=[str(m.get("technique_id", "")) for m in mappings if m.get("technique_id")],
            crypto_policy_checks=dict(enrichment.get("crypto_policy_checks", {})),
            max_s_total=float(triage.get("max_s_total", 0.0)),
            alerted_flow_count=int(triage.get("alerted_flow_count", 0)),
            requested_actions=requested_actions,
            ech_only=ech_only,
            fingerprint_only=fp_only,
            request_production_auto_merge=request_production_auto_merge,
            base_rationale=str(rec.get("rationale", "")),
        )

    def plan_from_request(self, request: ActionRequest) -> ResponsePlan:
        """Build a policy-enforced `ResponsePlan` from an explicit `ActionRequest`."""
        if request.extra_payload:
            validate_week10_payload(
                dict(request.extra_payload),
                strict=True,
                strip_ground_truth=True,
                policy_guard=self.policy_guard,
            )
        validate_week10_payload(
            {
                "incident_id": request.incident_id,
                "tier": request.tier,
                "environment": request.environment,
                "asset_criticality": request.asset_criticality,
                "primary_host_token": request.target_host_token,
                "requested_actions": list(request.requested_actions),
                "corroborating_channels": list(request.corroborating_channels),
                "severity": request.severity,
                "attack_technique_ids": list(request.attack_technique_ids),
                "crypto_policy_checks": dict(request.crypto_policy_checks),
            },
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )
        return self._enforce_tier_policy(
            incident_id=request.incident_id,
            tier_id=request.tier,
            environment=request.environment,
            asset_criticality=request.asset_criticality,
            primary_host_token=request.target_host_token,
            target_host_tokens=[request.target_host_token],
            verified_malicious=request.verified_malicious,
            policy_anomaly_only=request.policy_anomaly_only,
            severity=request.severity,
            corroborating_channels=list(request.corroborating_channels),
            attack_technique_ids=list(request.attack_technique_ids),
            crypto_policy_checks=dict(request.crypto_policy_checks),
            max_s_total=request.max_s_total,
            alerted_flow_count=request.alerted_flow_count,
            requested_actions=request.requested_actions if request.requested_actions else None,
            ech_only=request.ech_only,
            fingerprint_only=request.fingerprint_only,
            request_production_auto_merge=request.request_production_auto_merge,
            base_rationale="Explicit ActionRequest evaluated against DENY_BY_DEFAULT policy matrix.",
        )


class HumanApprovalGate:
    """Mandatory Stage 6a human-in-the-loop gate (`SR-06`, `SR-07`)."""

    def __init__(self, policy_guard: Optional[ScopePolicyGuard] = None) -> None:
        self.policy_guard = policy_guard or ScopePolicyGuard()

    def evaluate(
        self,
        plan: ResponsePlan,
        approval: Optional[ApprovalDecision] = None,
    ) -> GateDecision:
        decided_at = (
            approval.decided_at_utc
            if approval is not None and approval.decided_at_utc
            else _utc_now_iso()
        )

        validate_week10_payload(
            plan.to_dict(),
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )

        if approval is not None:
            validate_week10_payload(
                approval.to_dict(),
                strict=True,
                strip_ground_truth=True,
                policy_guard=self.policy_guard,
            )

        env = str(plan.environment).strip().lower()
        crit = str(plan.asset_criticality).strip().lower()
        tier = str(plan.tier).strip().upper()
        unique_channels = {
            str(c).strip().lower() for c in plan.corroborating_channels if str(c).strip()
        }

        # 1. Policy-denied plans (or forged plans violating tier/SR-04 invariants) can NEVER be approved
        if plan.policy_status == "DENIED_BY_DEFAULT" or plan.denial_reasons:
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=False,
                status="DENIED_BY_DEFAULT",
                reviewer_id=approval.reviewer_id.strip() if approval and approval.reviewer_id else None,
                reason=(
                    "Plan failed policy matrix checks under DENY_BY_DEFAULT: "
                    + ("; ".join(plan.denial_reasons) or "policy violation")
                ),
                decided_at_utc=decided_at,
            )

        selected_actions = [str(a).strip() for a in plan.selected_actions if str(a).strip()]
        allowed_actions = {str(a).strip() for a in plan.allowed_actions if str(a).strip()}
        disallowed_actions = [a for a in selected_actions if a not in allowed_actions]
        if not selected_actions or disallowed_actions:
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=False,
                status="DENIED_BY_DEFAULT",
                reviewer_id=approval.reviewer_id.strip() if approval and approval.reviewer_id else None,
                reason=(
                    "Plan has empty selected_actions under DENY_BY_DEFAULT (SR-06)."
                    if not selected_actions
                    else f"Action(s) {disallowed_actions} not in plan.allowed_actions under DENY_BY_DEFAULT (SR-06)."
                ),
                decided_at_utc=decided_at,
            )

        if tier == "R2" and (env not in {"lab", "staging"} or crit == "critical"):
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=False,
                status="DENIED_BY_DEFAULT",
                reviewer_id=approval.reviewer_id.strip() if approval and approval.reviewer_id else None,
                reason=f"R2 soft containment denied in env='{env}', criticality='{crit}' under DENY_BY_DEFAULT.",
                decided_at_utc=decided_at,
            )

        if tier in {"R2", "R3"} and (not plan.verified_malicious or len(unique_channels) < 2):
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=False,
                status="DENIED_BY_DEFAULT",
                reviewer_id=approval.reviewer_id.strip() if approval and approval.reviewer_id else None,
                reason="SR-04: Containment denied without verified_malicious=True and >= 2 unique corroborating channels.",
                decided_at_utc=decided_at,
            )

        # 2. R4 automatic production merge is unconditionally forbidden
        if (
            tier == "R4"
            and approval is not None
            and approval.request_production_auto_merge
        ):
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=False,
                status="DENIED_BY_DEFAULT",
                reviewer_id=approval.reviewer_id.strip() or None,
                reason="R4 automatic_production_merge_allowed=False cannot be overridden by approval.",
                decided_at_utc=decided_at,
            )

        effective_human_required = bool(
            plan.human_approval_required
            or (tier == "R3" and (env == "production" or crit == "critical"))
            or tier == "R4"
        )

        # 3. Non-human-gated plans (R0, R1, non-critical lab/staging R2, non-critical lab/staging R3)
        if not effective_human_required:
            if approval is not None and (
                approval.decision == "reject"
                or (approval.decision == "approve" and not approval.approved)
            ):
                return GateDecision(
                    plan_id=plan.plan_id,
                    incident_id=plan.incident_id,
                    approved=False,
                    status="REJECTED_BY_ANALYST",
                    reviewer_id=approval.reviewer_id.strip() or None,
                    reason=approval.reason or "Explicitly rejected by analyst.",
                    decided_at_utc=decided_at,
                )
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=True,
                status="APPROVED_BY_POLICY",
                reviewer_id="policy_auto_bounded",
                reason=(
                    f"Tier {tier} in '{env}' (criticality={crit}) "
                    "is policy-authorized without production human gate."
                ),
                decided_at_utc=decided_at,
            )

        # 4. Human-gated plans (R3 in production or critical, R4) -> DENY_BY_DEFAULT unless explicitly approved
        if approval is None or approval.decision == "pending":
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=False,
                status="DENIED_BY_DEFAULT",
                reviewer_id=None,
                reason=(
                    f"Human approval required for {tier} in '{env}' "
                    f"(criticality={crit}); blocked under DENY_BY_DEFAULT (SR-06/SR-07)."
                ),
                decided_at_utc=decided_at,
            )

        if approval.decision == "reject" or not approval.approved:
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=False,
                status="REJECTED_BY_ANALYST",
                reviewer_id=approval.reviewer_id.strip() or None,
                reason=approval.reason or "Proposal rejected by human analyst.",
                decided_at_utc=decided_at,
            )

        if not is_human_reviewer_id(approval.reviewer_id):
            return GateDecision(
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                approved=False,
                status="DENIED_BY_DEFAULT",
                reviewer_id=None,
                reason=(
                    "Human approval rejected: non-empty, non-automated human reviewer_id is mandatory (SR-07)."
                ),
                decided_at_utc=decided_at,
            )

        reviewer = str(approval.reviewer_id).strip()
        return GateDecision(
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            approved=True,
            status="APPROVED_BY_HUMAN",
            reviewer_id=reviewer,
            reason=approval.reason or f"Approved by analyst '{reviewer}' (SR-07).",
            decided_at_utc=decided_at,
        )
