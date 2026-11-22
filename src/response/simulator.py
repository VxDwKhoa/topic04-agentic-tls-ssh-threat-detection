"""
Pure In-Memory Response Simulator / Bounded Lab Adapter (Stage 6b, Topic 04 Week 10).

Provides 100% offline, deterministic, in-memory containment state transitions,
logical-clock TTL/auto-expiry, deterministic rollback tokens, structured R4
TLS/SSH/X.509 configuration remediation diffs (`automatic_production_merge_allowed=False`),
and an immutable-style audit trail (`ScopePolicyGuard`-enforced).

Strictly generates and executes ZERO OS-level firewall or network commands.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.agent.policy_guard import (
    IPV4_REGEX,
    ScopePolicyGuard,
    ScopePolicyViolationError,
)
from src.hunting.schemas import sanitize_json_value
from src.response.schemas import (
    AuditEvent,
    ContainmentEntry,
    ExecutionOutcome,
    GateDecision,
    RemediationDiff,
    ResponsePlan,
    ResponsePolicyViolationError,
    RollbackResult,
    SimulatedActionRecord,
    is_human_reviewer_id,
    validate_week10_payload,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deterministic_token(prefix: str, seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


class InMemoryResponseSimulator:
    """
    Pure in-memory bounded response simulator with logical-clock TTL,
    deterministic rollback tokens, structured R4 diffs, and full audit trail.
    """

    def __init__(
        self,
        policy_guard: Optional[ScopePolicyGuard] = None,
        initial_minute: float = 0.0,
        policy_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.policy_guard = policy_guard or ScopePolicyGuard()
        if isinstance(initial_minute, bool):
            raise ValueError("initial_minute must be a finite numeric value >= 0, not bool.")
        init_min = float(initial_minute)
        if not math.isfinite(init_min) or init_min < 0:
            raise ValueError("initial_minute must be finite and >= 0.")
        self._policy_path = (
            Path(policy_path)
            if policy_path is not None
            else Path(__file__).resolve().parent / "response_policy_tiers.json"
        )
        if not self._policy_path.is_file():
            raise FileNotFoundError(f"Response policy file not found: {self._policy_path}")
        with open(self._policy_path, "r", encoding="utf-8") as f:
            policy_data = json.load(f)
        self._tiers_by_id: Dict[str, Dict[str, Any]] = {
            str(t["tier_id"]).strip().upper(): dict(t) for t in policy_data.get("tiers", [])
        }
        self._clock_minutes: float = init_min
        self._containment_table: Dict[str, ContainmentEntry] = {}
        self._diffs: Dict[str, RemediationDiff] = {}
        self._token_index: Dict[str, Tuple[str, str]] = {}
        self._audit_trail: List[AuditEvent] = []
        self._action_counter: int = 0
        self._audit_counter: int = 0

    def _authoritative_allowed_actions(self, tier_id: str, environment: str) -> List[str]:
        tier_cfg = self._tiers_by_id.get(tier_id)
        if tier_cfg is None:
            return []
        if tier_id == "R3":
            if environment == "production":
                return [str(a) for a in tier_cfg.get("allowed_actions_production", [])]
            return [str(a) for a in tier_cfg.get("allowed_actions_lab", [])]
        return [str(a) for a in tier_cfg.get("allowed_actions", [])]

    @property
    def current_minute(self) -> float:
        return self._clock_minutes

    @property
    def audit_trail(self) -> List[AuditEvent]:
        return [replace(e, details=copy.deepcopy(e.details)) for e in self._audit_trail]

    @property
    def all_containments(self) -> List[ContainmentEntry]:
        return list(self._containment_table.values())

    @property
    def active_containments(self) -> List[ContainmentEntry]:
        return [e for e in self._containment_table.values() if e.status == "ACTIVE"]

    @staticmethod
    def _clone_remediation_diff(diff: RemediationDiff) -> RemediationDiff:
        return replace(
            diff,
            finding_codes=list(diff.finding_codes),
            current_config_summary=copy.deepcopy(diff.current_config_summary),
            proposed_config_summary=copy.deepcopy(diff.proposed_config_summary),
            unified_diff_lines=list(diff.unified_diff_lines),
        )

    @property
    def remediation_diffs(self) -> List[RemediationDiff]:
        return [self._clone_remediation_diff(d) for d in self._diffs.values()]

    def update_remediation_diff_retest_status(
        self, diff_id: str, retest_status: str
    ) -> Optional[RemediationDiff]:
        """Update `lab_conformance_retest_status` on an existing staged R4 diff after Stage 7."""
        diff_obj = self._diffs.get(diff_id)
        if diff_obj is None:
            return None
        updated = replace(diff_obj, lab_conformance_retest_status=str(retest_status))
        self._diffs[diff_id] = updated
        return self._clone_remediation_diff(updated)

    def get_active_containments(
        self,
        incident_id: Optional[str] = None,
        target_host_token: Optional[str] = None,
    ) -> List[ContainmentEntry]:
        results = self.active_containments
        if incident_id is not None:
            results = [e for e in results if e.incident_id == incident_id]
        if target_host_token is not None:
            results = [e for e in results if e.target_host_token == target_host_token]
        return results

    def _validate_text_fields(self, *values: Optional[str]) -> None:
        for val in values:
            if not val:
                continue
            if IPV4_REGEX.search(str(val)):
                raise ScopePolicyViolationError(
                    f"Raw IPv4 address detected in simulator text '{val}'; pseudonymized tokens required."
                )

    def record_audit_event(
        self,
        *,
        stage: str,
        event_type: str,
        incident_id: str,
        actor: str,
        outcome: str,
        plan_id: Optional[str] = None,
        tier: Optional[str] = None,
        environment: Optional[str] = None,
        target_host_token: Optional[str] = None,
        rollback_token: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        audit_input = {
            "stage": stage,
            "event_type": event_type,
            "incident_id": incident_id,
            "plan_id": plan_id,
            "tier": tier,
            "environment": environment,
            "target_host_token": target_host_token,
            "actor": actor,
            "outcome": outcome,
            "rollback_token": rollback_token,
            "details": copy.deepcopy(dict(details or {})),
        }
        sanitized_input, _ = validate_week10_payload(
            audit_input,
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )
        safe_details = dict(sanitized_input.get("details") or {})
        json.dumps(safe_details, allow_nan=False)

        self._audit_counter += 1
        event_id = _deterministic_token(
            "aud",
            f"{self._audit_counter}|{stage}|{event_type}|{incident_id}|{plan_id}|{self._clock_minutes}",
        )
        event = AuditEvent(
            event_id=event_id,
            sequence_no=self._audit_counter,
            timestamp_utc=_utc_now_iso(),
            logical_minute=self._clock_minutes,
            stage=stage,
            event_type=event_type,
            incident_id=incident_id,
            plan_id=plan_id,
            tier=tier,
            environment=environment,
            target_host_token=target_host_token,
            actor=actor,
            outcome=outcome,
            rollback_token=rollback_token,
            details=safe_details,
        )
        self._audit_trail.append(event)
        return replace(event, details=copy.deepcopy(event.details))

    def _build_remediation_diff(
        self,
        plan: ResponsePlan,
        diff_id: str,
        rollback_token: str,
    ) -> RemediationDiff:
        checks = plan.crypto_policy_checks or {}
        finding_codes: List[str] = []
        standards: List[str] = []
        current_cfg: Dict[str, Any] = {}
        proposed_cfg: Dict[str, Any] = {}
        diff_lines: List[str] = [
            f"--- a/crypto_policy/{plan.target_host_token}.conf",
            f"+++ b/crypto_policy/{plan.target_host_token}.conf",
        ]

        if checks.get("CR-01_deprecated_tls"):
            finding_codes.append("CR-01_DEPRECATED_TLS")
            standards.append("RFC 9846 / NIST SP 800-52 Rev.2")
            current_cfg["tls_min_version"] = "TLSv1.0"
            current_cfg["tls_ciphers"] = "TLS_RSA_WITH_AES_128_CBC_SHA"
            proposed_cfg["tls_min_version"] = "TLSv1.3"
            proposed_cfg["tls_ciphers"] = "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256"
            diff_lines.extend(
                [
                    "@@ TLS Profile (CR-01) @@",
                    "-tls_min_version = TLSv1.0",
                    "+tls_min_version = TLSv1.3",
                    "-tls_ciphers = TLS_RSA_WITH_AES_128_CBC_SHA",
                    "+tls_ciphers = TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256",
                ]
            )

        if checks.get("CR-04_weak_ssh_kex"):
            finding_codes.append("CR-04_WEAK_SSH_KEX")
            standards.append("RFC 9142")
            current_cfg["ssh_kex_algorithms"] = "diffie-hellman-group1-sha1"
            current_cfg["ssh_macs"] = "hmac-sha1"
            proposed_cfg["ssh_kex_algorithms"] = "mlkem768x25519-sha256,curve25519-sha256"
            proposed_cfg["ssh_macs"] = "hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com"
            diff_lines.extend(
                [
                    "@@ SSH KEX Profile (CR-04) @@",
                    "-KexAlgorithms diffie-hellman-group1-sha1",
                    "+KexAlgorithms mlkem768x25519-sha256,curve25519-sha256",
                    "-MACs hmac-sha1",
                    "+MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com",
                ]
            )

        if checks.get("CR-02_cert_trust_anomaly"):
            finding_codes.append("CR-02_CERT_TRUST_ANOMALY")
            standards.append("RFC 5280 / RFC 9162")
            current_cfg["cert_trust_mode"] = "self_signed_or_expired"
            current_cfg["ocsp_stapling"] = False
            proposed_cfg["cert_trust_mode"] = "internal_pkix_ca_chain_ct_verified"
            proposed_cfg["ocsp_stapling"] = True
            diff_lines.extend(
                [
                    "@@ X.509 Certificate Profile (CR-02) @@",
                    "-cert_trust_mode = self_signed_or_expired",
                    "+cert_trust_mode = internal_pkix_ca_chain_ct_verified",
                    "-ocsp_stapling = false",
                    "+ocsp_stapling = true",
                ]
            )

        if not finding_codes:
            finding_codes.append("CR-01_CR-04_BASELINE_HARDENING")
            standards.append("RFC 9846 / RFC 9142")
            current_cfg["crypto_profile"] = "legacy_compatibility"
            proposed_cfg["crypto_profile"] = "strict_tls13_aead_and_rfc9142_ssh"
            diff_lines.extend(
                [
                    "@@ Crypto Baseline Profile @@",
                    "-crypto_profile = legacy_compatibility",
                    "+crypto_profile = strict_tls13_aead_and_rfc9142_ssh",
                ]
            )

        return RemediationDiff(
            diff_id=diff_id,
            rollback_token=rollback_token,
            incident_id=plan.incident_id,
            plan_id=plan.plan_id,
            target_host_token=plan.target_host_token,
            environment=plan.environment,
            policy_standard="; ".join(standards),
            finding_codes=finding_codes,
            current_config_summary=current_cfg,
            proposed_config_summary=proposed_cfg,
            unified_diff_lines=diff_lines,
            lab_conformance_retest_status="PENDING_STAGE7_RETEST",
            automatic_production_merge_allowed=False,
            applied_to_production=False,
            status="PROPOSED_NEVER_AUTO_MERGED",
        )

    def execute_plan(
        self,
        plan: ResponsePlan,
        gate_decision: GateDecision,
    ) -> SimulatedActionRecord:
        if plan.plan_id != gate_decision.plan_id:
            raise ResponsePolicyViolationError(
                f"Mismatched plan_id '{plan.plan_id}' vs gate_decision.plan_id '{gate_decision.plan_id}'."
            )
        if plan.incident_id != gate_decision.incident_id:
            raise ResponsePolicyViolationError(
                f"Mismatched incident_id '{plan.incident_id}' vs gate_decision.incident_id '{gate_decision.incident_id}'."
            )

        validate_week10_payload(
            plan.to_dict(),
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )
        validate_week10_payload(
            gate_decision.to_dict(),
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )

        approved_statuses = {"APPROVED_BY_POLICY", "APPROVED_BY_HUMAN"}
        if gate_decision.approved and gate_decision.status not in approved_statuses:
            raise ResponsePolicyViolationError(
                f"Contradictory GateDecision: approved=True cannot have status='{gate_decision.status}'."
            )
        if not gate_decision.approved and gate_decision.status in approved_statuses:
            raise ResponsePolicyViolationError(
                f"Contradictory GateDecision: approved=False cannot have status='{gate_decision.status}'."
            )

        env = str(plan.environment).strip().lower()
        crit = str(plan.asset_criticality).strip().lower()
        tier = str(plan.tier).strip().upper()

        self._action_counter += 1
        now_utc = _utc_now_iso()
        record_id = _deterministic_token(
            "act", f"{self._action_counter}|{plan.plan_id}|{plan.incident_id}"
        )

        self.record_audit_event(
            stage="stage6_plan",
            event_type="PLAN_EVALUATED",
            incident_id=plan.incident_id,
            plan_id=plan.plan_id,
            tier=tier,
            environment=env,
            target_host_token=plan.target_host_token,
            actor="response_planner",
            outcome=plan.policy_status,
            details={
                "selected_actions": list(plan.selected_actions),
                "human_approval_required": plan.human_approval_required,
                "denial_reasons": list(plan.denial_reasons),
            },
        )
        self.record_audit_event(
            stage="stage6_gate",
            event_type="HUMAN_GATE_EVALUATED",
            incident_id=plan.incident_id,
            plan_id=plan.plan_id,
            tier=tier,
            environment=env,
            target_host_token=plan.target_host_token,
            actor=gate_decision.reviewer_id or "gate_deny_by_default",
            outcome=gate_decision.status,
            details={"approved": gate_decision.approved, "reason": gate_decision.reason},
        )

        if not gate_decision.approved:
            blocked_outcome: ExecutionOutcome = (
                "REJECTED_BY_ANALYST"
                if gate_decision.status == "REJECTED_BY_ANALYST"
                else "DENIED_BY_DEFAULT"
            )
            self.record_audit_event(
                stage="stage6_simulate",
                event_type="EXECUTION_BLOCKED",
                incident_id=plan.incident_id,
                plan_id=plan.plan_id,
                tier=tier,
                environment=env,
                target_host_token=plan.target_host_token,
                actor=gate_decision.reviewer_id or "deny_by_default_guard",
                outcome="BLOCKED_NO_EXECUTION",
                details={"gate_status": gate_decision.status, "reason": gate_decision.reason},
            )
            return SimulatedActionRecord(
                action_record_id=record_id,
                plan_id=plan.plan_id,
                incident_id=plan.incident_id,
                tier=tier,  # type: ignore[arg-type]
                environment=env,
                asset_criticality=crit,
                target_host_token=plan.target_host_token,
                outcome=blocked_outcome,
                executed_actions=[],
                network_state_mutated=False,
                containment_entries=[],
                remediation_diffs=[],
                rollback_tokens=[],
                gate_decision=gate_decision,
                simulated_time_to_containment_ms=0.0,
                created_at_utc=now_utc,
                logical_minute=self._clock_minutes,
            )

        # Defense-in-depth invariants before any simulation execution
        ScopePolicyGuard.check_environment_scope(env, crit)
        if tier not in self._tiers_by_id:
            raise ResponsePolicyViolationError(f"Unknown response tier '{plan.tier}'.")

        if plan.policy_status == "DENIED_BY_DEFAULT" or plan.denial_reasons:
            raise ResponsePolicyViolationError(
                f"Cannot execute denied plan '{plan.plan_id}': {plan.denial_reasons}"
            )

        actions_to_execute = list(
            dict.fromkeys(str(a).strip() for a in plan.selected_actions if str(a).strip())
        )
        if not actions_to_execute:
            raise ResponsePolicyViolationError(
                f"Cannot execute plan '{plan.plan_id}' with empty selected_actions."
            )

        auth_allowed = self._authoritative_allowed_actions(tier, env)
        plan_allowed = set(str(a).strip() for a in plan.allowed_actions)
        disallowed = [
            a for a in actions_to_execute if a not in plan_allowed or a not in auth_allowed
        ]
        if disallowed:
            raise ResponsePolicyViolationError(
                f"Defense-in-depth block: action(s) {disallowed} outside plan/policy allowlist for tier={tier}, env={env}."
            )

        if tier in {"R2", "R3"}:
            unique_channels = {
                str(c).strip().lower() for c in plan.corroborating_channels if str(c).strip()
            }
            if not plan.verified_malicious or len(unique_channels) < 2:
                raise ResponsePolicyViolationError(
                    "Defense-in-depth block (SR-04): R2/R3 containment requires verified_malicious=True and >= 2 unique corroborating channels."
                )

        if tier == "R2" and (env not in {"lab", "staging"} or crit == "critical"):
            raise ResponsePolicyViolationError(
                "Defense-in-depth block: R2 cannot execute in production or on critical assets."
            )

        requires_human = bool(
            plan.human_approval_required
            or (tier == "R3" and (env == "production" or crit == "critical"))
            or tier == "R4"
        )
        if requires_human:
            if gate_decision.status != "APPROVED_BY_HUMAN" or not is_human_reviewer_id(
                gate_decision.reviewer_id
            ):
                raise ResponsePolicyViolationError(
                    f"Defense-in-depth block (SR-07): tier={tier}, env={env}, crit={crit} requires APPROVED_BY_HUMAN with valid human reviewer_id."
                )

        if tier == "R4":
            if plan.automatic_production_merge_allowed is not False:
                raise ResponsePolicyViolationError(
                    "Defense-in-depth block: R4 automatic_production_merge_allowed must be False."
                )
            if plan.network_change_allowed:
                raise ResponsePolicyViolationError(
                    "Defense-in-depth block: R4 network_change_allowed must be False."
                )

        if tier in {"R0", "R1"} and plan.network_change_allowed:
            raise ResponsePolicyViolationError(
                f"Defense-in-depth block: {tier} cannot permit network_change_allowed=True."
            )

        entries: List[ContainmentEntry] = []
        diffs: List[RemediationDiff] = []
        rollback_tokens: List[str] = []

        if tier == "R0":
            outcome: ExecutionOutcome = "OBSERVED_ONLY"
            network_mutated = False
            ttc_ms = 5.0
        elif tier == "R1":
            outcome = "TICKET_CREATED"
            network_mutated = False
            ttc_ms = 12.0
        elif tier in {"R2", "R3"}:
            default_ttl = 30 if tier == "R2" else 60
            raw_ttl: Any = (
                default_ttl if plan.auto_expire_minutes is None else plan.auto_expire_minutes
            )
            if (
                isinstance(raw_ttl, bool)
                or not isinstance(raw_ttl, (int, float))
                or not math.isfinite(float(raw_ttl))
                or float(raw_ttl) <= 0
            ):
                raise ResponsePolicyViolationError(
                    f"Invalid auto_expire_minutes '{plan.auto_expire_minutes}' for {tier}; must be finite and > 0."
                )
            ttl = int(raw_ttl)
            if ttl <= 0:
                raise ResponsePolicyViolationError(
                    f"Invalid auto_expire_minutes '{plan.auto_expire_minutes}' for {tier}; must be >= 1."
                )
            scope = (
                "bounded_lab_staging"
                if env in {"lab", "staging"}
                else "simulated_production_human_approved"
            )
            for idx, action in enumerate(actions_to_execute):
                entry_id = _deterministic_token(
                    "cnt",
                    f"{self._action_counter}|{idx}|{plan.plan_id}|{plan.target_host_token}|{action}",
                )
                rb_token = _deterministic_token(
                    "rb",
                    f"{self._action_counter}|{idx}|{plan.incident_id}|{plan.target_host_token}|{action}|{self._clock_minutes}",
                )
                entry = ContainmentEntry(
                    entry_id=entry_id,
                    rollback_token=rb_token,
                    incident_id=plan.incident_id,
                    plan_id=plan.plan_id,
                    tier=tier,  # type: ignore[arg-type]
                    action=action,
                    environment=env,
                    asset_criticality=crit,
                    target_host_token=plan.target_host_token,
                    scope=scope,
                    created_at_utc=now_utc,
                    created_minute=self._clock_minutes,
                    ttl_minutes=ttl,
                    expires_at_minute=self._clock_minutes + ttl,
                    status="ACTIVE",
                    reviewer_id=gate_decision.reviewer_id,
                )
                self._containment_table[entry_id] = entry
                self._token_index[rb_token] = ("containment", entry_id)
                entries.append(entry)
                rollback_tokens.append(rb_token)
            outcome = "EXECUTED_SIMULATED"
            network_mutated = True
            ttc_ms = 25.0 if tier == "R2" else 45.0
        else:  # R4
            diff_id = _deterministic_token(
                "diff", f"{self._action_counter}|{plan.plan_id}|{plan.target_host_token}"
            )
            rb_token = _deterministic_token(
                "rb", f"{self._action_counter}|r4|{plan.incident_id}|{plan.target_host_token}"
            )
            diff_obj = self._build_remediation_diff(plan, diff_id=diff_id, rollback_token=rb_token)
            self._diffs[diff_id] = diff_obj
            self._token_index[rb_token] = ("diff", diff_id)
            diffs.append(self._clone_remediation_diff(diff_obj))
            rollback_tokens.append(rb_token)
            outcome = "PROPOSED_REMEDIATION_DIFF"
            network_mutated = False
            ttc_ms = 35.0

        self.record_audit_event(
            stage="stage6_simulate",
            event_type="ACTION_EXECUTED_IN_SIMULATOR",
            incident_id=plan.incident_id,
            plan_id=plan.plan_id,
            tier=tier,
            environment=env,
            target_host_token=plan.target_host_token,
            actor=gate_decision.reviewer_id or "in_memory_simulator",
            outcome=outcome,
            rollback_token=rollback_tokens[0] if rollback_tokens else None,
            details={
                "executed_actions": list(actions_to_execute),
                "network_state_mutated": network_mutated,
                "rollback_tokens": rollback_tokens,
                "os_commands_generated": 0,
            },
        )

        return SimulatedActionRecord(
            action_record_id=record_id,
            plan_id=plan.plan_id,
            incident_id=plan.incident_id,
            tier=tier,  # type: ignore[arg-type]
            environment=env,
            asset_criticality=crit,
            target_host_token=plan.target_host_token,
            outcome=outcome,
            executed_actions=list(actions_to_execute),
            network_state_mutated=network_mutated,
            containment_entries=entries,
            remediation_diffs=diffs,
            rollback_tokens=rollback_tokens,
            gate_decision=gate_decision,
            simulated_time_to_containment_ms=ttc_ms,
            created_at_utc=now_utc,
            logical_minute=self._clock_minutes,
        )

    def advance_time(self, minutes: float) -> List[ContainmentEntry]:
        """Advance logical simulator clock and auto-expire containment entries past TTL."""
        if isinstance(minutes, bool):
            raise ValueError("minutes must be a finite numeric value >= 0, not bool.")
        delta = float(minutes)
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("minutes must be finite and >= 0.")
        self._clock_minutes += delta

        expired_now: List[ContainmentEntry] = []
        for entry_id, entry in list(self._containment_table.items()):
            if entry.status == "ACTIVE" and self._clock_minutes >= entry.expires_at_minute:
                updated = replace(entry, status="EXPIRED")
                self._containment_table[entry_id] = updated
                expired_now.append(updated)
                self.record_audit_event(
                    stage="stage6_ttl",
                    event_type="CONTAINMENT_TTL_EXPIRED",
                    incident_id=entry.incident_id,
                    plan_id=entry.plan_id,
                    tier=entry.tier,
                    environment=entry.environment,
                    target_host_token=entry.target_host_token,
                    actor="logical_clock_ttl",
                    outcome="EXPIRED",
                    rollback_token=entry.rollback_token,
                    details={
                        "entry_id": entry.entry_id,
                        "ttl_minutes": entry.ttl_minutes,
                        "expires_at_minute": entry.expires_at_minute,
                        "current_minute": self._clock_minutes,
                    },
                )
        return expired_now

    def rollback(
        self,
        rollback_token: str,
        actor: str = "analyst",
        reason: str = "manual_rollback",
    ) -> RollbackResult:
        """Atomically revert an active containment entry or staged R4 diff using its rollback token."""
        validate_week10_payload(
            {"rollback_token": rollback_token, "actor": actor, "reason": reason},
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )
        now_utc = _utc_now_iso()

        loc = self._token_index.get(rollback_token)
        if loc is None:
            self.record_audit_event(
                stage="stage6_rollback",
                event_type="ROLLBACK_FAILED_UNKNOWN_TOKEN",
                incident_id="unknown",
                actor=actor,
                outcome="FAILED",
                rollback_token=rollback_token,
                details={"reason": reason},
            )
            return RollbackResult(
                success=False,
                rollback_token=rollback_token,
                entry_id=None,
                incident_id=None,
                previous_status=None,
                new_status=None,
                actor=actor,
                reason=f"Unknown rollback_token '{rollback_token}'.",
                logical_minute=self._clock_minutes,
                timestamp_utc=now_utc,
            )

        kind, item_id = loc
        if kind == "containment":
            entry = self._containment_table[item_id]
            if entry.status != "ACTIVE":
                self.record_audit_event(
                    stage="stage6_rollback",
                    event_type="ROLLBACK_SKIPPED_NOT_ACTIVE",
                    incident_id=entry.incident_id,
                    plan_id=entry.plan_id,
                    tier=entry.tier,
                    environment=entry.environment,
                    target_host_token=entry.target_host_token,
                    actor=actor,
                    outcome="FAILED",
                    rollback_token=rollback_token,
                    details={"current_status": entry.status, "reason": reason},
                )
                return RollbackResult(
                    success=False,
                    rollback_token=rollback_token,
                    entry_id=entry.entry_id,
                    incident_id=entry.incident_id,
                    previous_status=entry.status,
                    new_status=entry.status,
                    actor=actor,
                    reason=f"Containment '{entry.entry_id}' is '{entry.status}', not ACTIVE.",
                    logical_minute=self._clock_minutes,
                    timestamp_utc=now_utc,
                )

            updated = replace(
                entry,
                status="ROLLED_BACK",
                rolled_back_at_minute=self._clock_minutes,
                rollback_reason=reason,
            )
            self._containment_table[item_id] = updated
            self.record_audit_event(
                stage="stage6_rollback",
                event_type="CONTAINMENT_ROLLED_BACK",
                incident_id=entry.incident_id,
                plan_id=entry.plan_id,
                tier=entry.tier,
                environment=entry.environment,
                target_host_token=entry.target_host_token,
                actor=actor,
                outcome="ROLLED_BACK",
                rollback_token=rollback_token,
                details={"entry_id": entry.entry_id, "reason": reason},
            )
            return RollbackResult(
                success=True,
                rollback_token=rollback_token,
                entry_id=entry.entry_id,
                incident_id=entry.incident_id,
                previous_status="ACTIVE",
                new_status="ROLLED_BACK",
                actor=actor,
                reason=reason,
                logical_minute=self._clock_minutes,
                timestamp_utc=now_utc,
            )

        # R4 staged remediation diff rollback
        diff_obj = self._diffs[item_id]
        if diff_obj.status == "ROLLED_BACK":
            return RollbackResult(
                success=False,
                rollback_token=rollback_token,
                entry_id=diff_obj.diff_id,
                incident_id=diff_obj.incident_id,
                previous_status=diff_obj.status,
                new_status=diff_obj.status,
                actor=actor,
                reason=f"Remediation diff '{diff_obj.diff_id}' already rolled back.",
                logical_minute=self._clock_minutes,
                timestamp_utc=now_utc,
            )

        prev_status = diff_obj.status
        updated_diff = replace(diff_obj, status="ROLLED_BACK")
        self._diffs[item_id] = updated_diff
        self.record_audit_event(
            stage="stage6_rollback",
            event_type="REMEDIATION_DIFF_ROLLED_BACK",
            incident_id=diff_obj.incident_id,
            plan_id=diff_obj.plan_id,
            tier="R4",
            environment=diff_obj.environment,
            target_host_token=diff_obj.target_host_token,
            actor=actor,
            outcome="ROLLED_BACK",
            rollback_token=rollback_token,
            details={"diff_id": diff_obj.diff_id, "reason": reason},
        )
        return RollbackResult(
            success=True,
            rollback_token=rollback_token,
            entry_id=diff_obj.diff_id,
            incident_id=diff_obj.incident_id,
            previous_status=prev_status,
            new_status="ROLLED_BACK",
            actor=actor,
            reason=reason,
            logical_minute=self._clock_minutes,
            timestamp_utc=now_utc,
        )

    def export_audit_json(self, output_path: Union[str, Path]) -> Path:
        """Export sanitized audit trail to local JSON (`allow_nan=False`)."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = sanitize_json_value(
            {
                "schema_version": "1.0.0",
                "exported_at_utc": _utc_now_iso(),
                "logical_minute": self._clock_minutes,
                "audit_event_count": len(self._audit_trail),
                "active_containment_count": len(self.active_containments),
                "remediation_diff_count": len(self._diffs),
                "audit_trail": [e.to_dict() for e in self._audit_trail],
            }
        )
        sanitized, _ = validate_week10_payload(
            payload,
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )
        text = json.dumps(sanitized, indent=2, allow_nan=False)
        out.write_text(text, encoding="utf-8")
        return out
