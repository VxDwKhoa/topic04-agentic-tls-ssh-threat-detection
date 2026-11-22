"""
Stage 7 Post-Action Re-test Verifier & Stage 8 Analyst-Approved Feedback Manager
(Topic 04 Week 10).

Implements Section 15 Stages 7 & 8:
- Stage 7 (`Stage7RetestVerifier`): observes subsequent telemetry windows after
  simulated containment or R4 conformance re-test; automatically invokes
  `simulator.rollback(...)` when false positives or excessive benign disruption
  are detected.
- Stage 8 (`Stage8FeedbackManager`): records ONLY analyst-approved feedback
  (`analyst_approved=True` with non-empty `reviewer_id`), strictly blocking any
  autonomous relabeling or self-training (`AutonomousTrainingForbiddenError`),
  enforcing `ScopePolicyGuard` (`SR-01`/`CR-06`), and supporting JSON/JSONL export.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.agent.policy_guard import ScopePolicyGuard
from src.hunting.schemas import sanitize_json_value
from src.response.schemas import (
    AnalystFeedbackRecord,
    AutonomousTrainingForbiddenError,
    FeedbackSubmissionResult,
    RetestObservation,
    RetestOutcome,
    RetestResult,
    RollbackResult,
    SimulatedActionRecord,
    is_human_reviewer_id,
    validate_week10_payload,
)
from src.response.simulator import InMemoryResponseSimulator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Stage7RetestVerifier:
    """Stage 7 post-action re-test verifier with automatic false-positive rollback."""

    def __init__(
        self,
        disruption_threshold: float = 0.25,
        containment_score_threshold: float = 0.45,
        policy_guard: Optional[ScopePolicyGuard] = None,
    ) -> None:
        self.disruption_threshold = float(disruption_threshold)
        self.containment_score_threshold = float(containment_score_threshold)
        self.policy_guard = policy_guard or ScopePolicyGuard()
        self._retest_counter: int = 0

    def verify(
        self,
        action_record: SimulatedActionRecord,
        observation: RetestObservation,
        simulator: InMemoryResponseSimulator,
    ) -> RetestResult:
        if not math.isfinite(float(observation.benign_traffic_disruption_ratio)):
            raise ValueError(
                "RetestObservation.benign_traffic_disruption_ratio must be finite (not NaN/Inf)."
            )
        if not math.isfinite(float(observation.post_max_s_total)):
            raise ValueError(
                "RetestObservation.post_max_s_total must be finite (not NaN/Inf)."
            )
        obs_dict = observation.to_dict()
        validate_week10_payload(
            obs_dict,
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )

        self._retest_counter += 1
        retest_digest = hashlib.sha256(
            f"{self._retest_counter}|{action_record.action_record_id}|{observation.window_id}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
        retest_id = f"ret_{retest_digest}"
        now_utc = _utc_now_iso()

        if (
            action_record.outcome in {"DENIED_BY_DEFAULT", "REJECTED_BY_ANALYST"}
            or action_record.tier in {"R0", "R1"}
        ):
            result = RetestResult(
                retest_id=retest_id,
                incident_id=action_record.incident_id,
                plan_id=action_record.plan_id,
                tier=action_record.tier,
                outcome="NO_RETEST_REQUIRED",
                containment_verified=False,
                false_positive_detected=False,
                auto_rollback_triggered=False,
                rollback_results=[],
                escalate_to_analyst=False,
                rationale=(
                    f"No active containment or remediation diff to re-test (tier={action_record.tier}, "
                    f"outcome={action_record.outcome})."
                ),
                checked_at_utc=now_utc,
            )
            simulator.record_audit_event(
                stage="stage7_retest",
                event_type="RETEST_SKIPPED_NO_MUTATION",
                incident_id=action_record.incident_id,
                plan_id=action_record.plan_id,
                tier=action_record.tier,
                environment=action_record.environment,
                target_host_token=action_record.target_host_token,
                actor="stage7_retest_verifier",
                outcome=result.outcome,
                details={"retest_id": retest_id, "window_id": observation.window_id},
            )
            return result

        fp_detected = bool(
            observation.false_positive_confirmed
            or observation.benign_traffic_disruption_ratio > self.disruption_threshold
        )

        rollback_results: List[RollbackResult] = []
        auto_rollback = False
        containment_verified = False
        escalate = False

        if fp_detected:
            outcome: RetestOutcome = "FALSE_POSITIVE_DETECTED"
            for tok in action_record.rollback_tokens:
                rb_res = simulator.rollback(
                    tok,
                    actor="stage7_retest_verifier",
                    reason=(
                        f"Auto-rollback triggered on Stage 7 re-test "
                        f"(false_positive_confirmed={observation.false_positive_confirmed}, "
                        f"disruption_ratio={observation.benign_traffic_disruption_ratio:.3f})."
                    ),
                )
                rollback_results.append(rb_res)
                if rb_res.success:
                    auto_rollback = True
            if action_record.tier == "R4":
                for idx, d in enumerate(action_record.remediation_diffs):
                    updated_d = simulator.update_remediation_diff_retest_status(
                        d.diff_id, "FAILED_LAB_CONFORMANCE"
                    )
                    if updated_d is not None:
                        action_record.remediation_diffs[idx] = updated_d
            if auto_rollback:
                rationale = (
                    "Stage 7 re-test detected false positive or excessive benign traffic disruption; "
                    "automatic rollback executed."
                )
            else:
                rationale = (
                    "Stage 7 re-test detected false positive or excessive benign traffic disruption; "
                    "no active containment remained to roll back (already expired or inactive)."
                )
        elif action_record.tier == "R4":
            if observation.crypto_conformance_passed is True:
                outcome = "POLICY_CONFORMANT"
                containment_verified = True
                new_diff_status = "PASSED_IN_LAB_SIMULATION"
                rationale = (
                    "Stage 7 lab conformance re-test passed (RFC 9846 / RFC 9142 / RFC 5280); "
                    "automatic_production_merge_allowed remains False."
                )
            elif observation.crypto_conformance_passed is False:
                outcome = "THREAT_PERSISTS_ESCALATE"
                escalate = True
                new_diff_status = "FAILED_LAB_CONFORMANCE"
                rationale = (
                    "Stage 7 lab conformance re-test failed for R4 remediation diff; "
                    "escalating to analyst before any staged rollout."
                )
            else:
                outcome = "THREAT_PERSISTS_ESCALATE"
                escalate = True
                new_diff_status = "UNVERIFIED_LAB_CONFORMANCE"
                rationale = (
                    "Stage 7 lab conformance result missing (crypto_conformance_passed=None) for R4 remediation diff; "
                    "escalating to analyst as unverified."
                )
            for idx, d in enumerate(action_record.remediation_diffs):
                updated_d = simulator.update_remediation_diff_retest_status(
                    d.diff_id, new_diff_status
                )
                if updated_d is not None:
                    action_record.remediation_diffs[idx] = updated_d
        else:
            # R2 / R3 active containment verification
            if (
                observation.post_alerted_flow_count == 0
                and observation.post_max_s_total < self.containment_score_threshold
                and not observation.active_reason_codes
            ):
                outcome = "CONTAINMENT_VERIFIED"
                containment_verified = True
                rationale = (
                    f"Subsequent window {observation.window_id} confirmed containment: "
                    f"alerted_flows=0, post_max_s_total={observation.post_max_s_total:.3f}."
                )
            else:
                outcome = "THREAT_PERSISTS_ESCALATE"
                escalate = True
                rationale = (
                    f"Threat signals persisted in window {observation.window_id} "
                    f"(alerted_flows={observation.post_alerted_flow_count}, "
                    f"post_max_s_total={observation.post_max_s_total:.3f}); escalating to analyst."
                )

        result = RetestResult(
            retest_id=retest_id,
            incident_id=action_record.incident_id,
            plan_id=action_record.plan_id,
            tier=action_record.tier,
            outcome=outcome,
            containment_verified=containment_verified,
            false_positive_detected=fp_detected,
            auto_rollback_triggered=auto_rollback,
            rollback_results=rollback_results,
            escalate_to_analyst=escalate,
            rationale=rationale,
            checked_at_utc=now_utc,
        )
        simulator.record_audit_event(
            stage="stage7_retest",
            event_type="RETEST_COMPLETED",
            incident_id=action_record.incident_id,
            plan_id=action_record.plan_id,
            tier=action_record.tier,
            environment=action_record.environment,
            target_host_token=action_record.target_host_token,
            actor="stage7_retest_verifier",
            outcome=outcome,
            rollback_token=(
                action_record.rollback_tokens[0] if action_record.rollback_tokens else None
            ),
            details={
                "retest_id": retest_id,
                "window_id": observation.window_id,
                "containment_verified": containment_verified,
                "false_positive_detected": fp_detected,
                "auto_rollback_triggered": auto_rollback,
                "escalate_to_analyst": escalate,
            },
        )
        return result


class Stage8FeedbackManager:
    """
    Stage 8 Analyst-Approved Feedback Manager.

    Strictly enforces Section 15 Stage 8 & Section 8:
    - Analyst-approved labels only (`analyst_approved=True` + non-empty `reviewer_id`).
    - Autonomous relabeling or self-training (`autonomous_relabel=True`, `self_train=True`,
      `auto_train=True`) is unconditionally rejected.
    - All stored/exported feedback passes `ScopePolicyGuard` (zero L7, secrets, or raw IPv4).
    """

    def __init__(self, policy_guard: Optional[ScopePolicyGuard] = None) -> None:
        self.policy_guard = policy_guard or ScopePolicyGuard()
        self._records: List[Dict[str, Any]] = []
        self._rejected_attempts: List[Dict[str, Any]] = []

    @property
    def records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._records)

    @property
    def rejected_attempts(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._rejected_attempts)

    def submit_feedback(
        self,
        feedback: AnalystFeedbackRecord,
        simulator: Optional[InMemoryResponseSimulator] = None,
        raise_on_forbidden: bool = True,
    ) -> FeedbackSubmissionResult:
        raw_dict = feedback.to_dict()
        validate_week10_payload(
            raw_dict,
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )

        reviewer = str(feedback.reviewer_id or "").strip()
        timestamp = feedback.submitted_at_utc or _utc_now_iso()

        if feedback.autonomous_relabel or feedback.self_train or feedback.auto_train:
            reason = (
                "Autonomous relabeling and automatic self-training are strictly forbidden "
                "(Section 8, Section 15 Stage 8)."
            )
            self._rejected_attempts.append(
                sanitize_json_value(
                    {
                        "incident_id": feedback.incident_id,
                        "plan_id": feedback.plan_id,
                        "reviewer_id": reviewer,
                        "status": "REJECTED_AUTONOMOUS_TRAINING_FORBIDDEN",
                        "reason": reason,
                        "timestamp_utc": timestamp,
                    }
                )
            )
            if simulator is not None:
                simulator.record_audit_event(
                    stage="stage8_feedback",
                    event_type="AUTONOMOUS_TRAINING_BLOCKED",
                    incident_id=feedback.incident_id,
                    plan_id=feedback.plan_id,
                    tier=feedback.recommended_tier,
                    actor=reviewer or "autonomous_loop_attempt",
                    outcome="REJECTED_AUTONOMOUS_TRAINING_FORBIDDEN",
                    details={"reason": reason},
                )
            if raise_on_forbidden:
                raise AutonomousTrainingForbiddenError(reason)
            return FeedbackSubmissionResult(
                accepted=False,
                status="REJECTED_AUTONOMOUS_TRAINING_FORBIDDEN",
                incident_id=feedback.incident_id,
                reviewer_id=reviewer,
                reason=reason,
                record=None,
            )

        if not feedback.analyst_approved or not is_human_reviewer_id(reviewer):
            reason = (
                "Stage 8 feedback requires analyst_approved=True and a valid non-automated human reviewer_id; "
                "unapproved or automated labels cannot enter feedback dataset."
            )
            self._rejected_attempts.append(
                sanitize_json_value(
                    {
                        "incident_id": feedback.incident_id,
                        "plan_id": feedback.plan_id,
                        "reviewer_id": reviewer,
                        "status": "REJECTED_UNAPPROVED_OR_INVALID",
                        "reason": reason,
                        "timestamp_utc": timestamp,
                    }
                )
            )
            if simulator is not None:
                simulator.record_audit_event(
                    stage="stage8_feedback",
                    event_type="UNAPPROVED_FEEDBACK_BLOCKED",
                    incident_id=feedback.incident_id,
                    plan_id=feedback.plan_id,
                    tier=feedback.recommended_tier,
                    actor=reviewer or "unapproved_submitter",
                    outcome="REJECTED_UNAPPROVED_OR_INVALID",
                    details={"reason": reason},
                )
            if raise_on_forbidden:
                raise AutonomousTrainingForbiddenError(reason)
            return FeedbackSubmissionResult(
                accepted=False,
                status="REJECTED_UNAPPROVED_OR_INVALID",
                incident_id=feedback.incident_id,
                reviewer_id=reviewer,
                reason=reason,
                record=None,
            )

        stored_record = sanitize_json_value(
            {
                "feedback_id": "fb_"
                + hashlib.sha256(
                    f"{len(self._records) + 1}|{feedback.incident_id}|{feedback.plan_id}|{reviewer}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:12],
                "incident_id": feedback.incident_id,
                "plan_id": feedback.plan_id,
                "recommended_tier": feedback.recommended_tier,
                "analyst_approved": True,
                "reviewer_id": reviewer,
                "verified_label": feedback.verified_label,
                "action_acceptable": bool(feedback.action_acceptable),
                "notes": feedback.notes,
                "autonomous_relabel": False,
                "self_train": False,
                "self_training_executed": False,
                "training_status": "ANALYST_APPROVED_OFFLINE_DATASET_ONLY",
                "submitted_at_utc": timestamp,
            }
        )
        validate_week10_payload(
            stored_record,
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )
        self._records.append(stored_record)

        if simulator is not None:
            simulator.record_audit_event(
                stage="stage8_feedback",
                event_type="ANALYST_FEEDBACK_RECORDED",
                incident_id=feedback.incident_id,
                plan_id=feedback.plan_id,
                tier=feedback.recommended_tier,
                actor=reviewer,
                outcome="ACCEPTED_ANALYST_APPROVED_FEEDBACK",
                details={
                    "feedback_id": stored_record["feedback_id"],
                    "verified_label": feedback.verified_label,
                    "action_acceptable": feedback.action_acceptable,
                    "self_training_executed": False,
                },
            )

        return FeedbackSubmissionResult(
            accepted=True,
            status="ACCEPTED_ANALYST_APPROVED_FEEDBACK",
            incident_id=feedback.incident_id,
            reviewer_id=reviewer,
            reason="Analyst-approved feedback recorded offline; autonomous self-training disabled.",
            record=copy.deepcopy(stored_record),
        )

    def export_json(self, output_path: Union[str, Path]) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = sanitize_json_value(
            {
                "schema_version": "1.0.0",
                "exported_at_utc": _utc_now_iso(),
                "autonomous_training_blocked": True,
                "accepted_count": len(self._records),
                "rejected_count": len(self._rejected_attempts),
                "feedback_records": copy.deepcopy(self._records),
                "rejected_attempts": copy.deepcopy(self._rejected_attempts),
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

    def export_jsonl(self, output_path: Union[str, Path]) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        for rec in self._records:
            sanitized, _ = validate_week10_payload(
                rec,
                strict=True,
                strip_ground_truth=True,
                policy_guard=self.policy_guard,
            )
            line = json.dumps(sanitized, allow_nan=False)
            lines.append(line)
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return out
