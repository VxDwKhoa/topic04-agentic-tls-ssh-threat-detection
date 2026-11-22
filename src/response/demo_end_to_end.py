"""
End-to-End Offline Demo CLI for Topic 04 (Week 10, R5).

Connects:
  NF-UQ-NIDS-v2 (or Week 9 report / L5-L6 fixtures)
  -> Privacy Validator & ScopePolicyGuard (SR-01, CR-06)
  -> Weeks 5-8 Models & Calibrated Fusion (Week8ScoringAdapter)
  -> Week 9 ThreatHuntingAgent (Stages 1-5)
  -> Week 10 ResponsePlanner + HumanApprovalGate + InMemoryResponseSimulator
  -> Stage 7 Re-test + Rollback + Stage 8 Analyst-Approved Feedback

Supports both non-interactive mode (defaulting to `DENY_BY_DEFAULT` when no
approval flag is passed, plus explicit CLI flags and preset profiles) and
interactive analyst prompt mode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

import pandas as pd

from src.agent.policy_guard import (
    ScopePolicyGuard,
)
from src.agent.state import AgentInvestigationResult
from src.agent.workflow import ThreatHuntingAgent
from src.features.data_loader import NFUQDataLoader
from src.hunting.pipeline import Week8ScoringAdapter
from src.hunting.schemas import sanitize_json_value
from src.response.evaluator import _build_synthetic_r2_r4_investigations
from src.response.planner import HumanApprovalGate, ResponsePlanner
from src.response.retest_learn import Stage7RetestVerifier, Stage8FeedbackManager
from src.response.schemas import (
    AnalystFeedbackRecord,
    ApprovalDecision,
    RetestObservation,
    validate_week10_payload,
)
from src.response.simulator import InMemoryResponseSimulator
from src.utils.config import PathResolver

PROFILES = (
    "deny_by_default",
    "approve_r3_verify_retest",
    "reject_fp_rollback",
    "lab_r2_soft_containment",
    "r4_crypto_remediation",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EndToEndDemoRunner:
    """Orchestrates the offline Weeks 5-10 end-to-end pipeline."""

    def __init__(
        self,
        checkpoint_root: Optional[Path] = None,
        data_path: Optional[Path] = None,
        week9_report_path: Optional[Path] = None,
        policy_path: Optional[Path] = None,
        output_stream: Optional[TextIO] = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.checkpoint_root = checkpoint_root
        self.data_path = data_path
        self.week9_report_path = week9_report_path
        self.policy_guard = ScopePolicyGuard()
        self.planner = ResponsePlanner(policy_path=policy_path, policy_guard=self.policy_guard)
        self.gate = HumanApprovalGate(policy_guard=self.policy_guard)
        self.simulator = InMemoryResponseSimulator(policy_guard=self.policy_guard)
        self.verifier = Stage7RetestVerifier(policy_guard=self.policy_guard)
        self.feedback = Stage8FeedbackManager(policy_guard=self.policy_guard)
        self.out: TextIO = output_stream or sys.stdout

    def _emit(self, line: str) -> None:
        safe_line = line.encode("ascii", errors="replace").decode("ascii")
        self.out.write(safe_line + "\n")

    def _load_investigations(
        self,
        *,
        profile: str,
        environment: str,
        asset_criticality: str,
        max_incidents: int,
        include_low_signal_clusters: bool = True,
    ) -> tuple[List[Any], str]:
        if profile == "lab_r2_soft_containment":
            syn = _build_synthetic_r2_r4_investigations()
            return [syn["syn_r2_lab_scan"]], "synthetic_r2_lab_fixture_via_ThreatHuntingAgent"

        if profile == "r4_crypto_remediation":
            syn = _build_synthetic_r2_r4_investigations()
            return [syn["syn_r4_cr01_prod"], syn["syn_r4_cr04_staging"]], "synthetic_r4_zeek_l5_l6_fixture_via_ThreatHuntingAgent"

        if self.week9_report_path is not None:
            with open(self.week9_report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            incidents = list(data.get("incidents", []))
            if max_incidents > 0:
                incidents = incidents[:max_incidents]
            return incidents, f"week9_report:{self.week9_report_path}"

        # Live offline pipeline: NF-UQ-NIDS-v2 -> ScopePolicyGuard -> Weeks 5-8 -> Week 9
        if self.data_path is not None:
            df_raw = pd.read_parquet(self.data_path)
            source_desc = f"parquet:{self.data_path}"
        else:
            resolved = PathResolver.resolve(
                checkpoint_dir=str(self.checkpoint_root) if self.checkpoint_root else None,
                prefer_sample=True,
            )
            loader = NFUQDataLoader(resolved, strict_privacy=True)
            splits = loader.load_tabular_splits()
            df_raw = splits["test"]
            source_desc = f"nfuq_sample_test_split(n={len(df_raw)},mode={resolved.mode})"
            if self.checkpoint_root is None and resolved.checkpoint_dir is not None:
                self.checkpoint_root = resolved.checkpoint_dir

        # Explicit privacy gate before Weeks 5-8 scoring (SR-01 / CR-06)
        df_clean, _ = self.policy_guard.validate_dataframe(
            df_raw, strip_ground_truth=True, strict=True
        )

        ckpt_root = self.checkpoint_root
        if ckpt_root is None:
            ckpt_root = self.repo_root.parent / "Notebook" / "Checkpoint"

        if not Path(ckpt_root).is_dir():
            # Fallback to Week 9 saved report if external checkpoints are absent
            fallback_report = self.repo_root / "report" / "week9_incident_evidence_graphs.json"
            if fallback_report.is_file():
                with open(fallback_report, "r", encoding="utf-8") as f:
                    data = json.load(f)
                incidents = list(data.get("incidents", []))
                if max_incidents > 0:
                    incidents = incidents[:max_incidents]
                return incidents, f"week9_report_fallback:{fallback_report}"
            raise FileNotFoundError(
                f"Checkpoint directory not found at '{ckpt_root}' and no Week 9 fallback report exists."
            )

        adapter = Week8ScoringAdapter.from_checkpoint_root(ckpt_root, device="cpu")
        agent = ThreatHuntingAgent(scoring_adapter=adapter, policy_guard=self.policy_guard)
        results: List[AgentInvestigationResult] = agent.investigate_dataframe(
            df_clean,
            environment=environment,
            asset_criticality=asset_criticality,
            include_low_signal_clusters=include_low_signal_clusters,
        )
        if max_incidents > 0:
            results = results[:max_incidents]
        return results, source_desc

    def run(
        self,
        *,
        interactive: bool = False,
        profile: str = "deny_by_default",
        environment: Optional[str] = None,
        asset_criticality: Optional[str] = None,
        max_incidents: int = 5,
        approval_decision: Optional[str] = None,
        reviewer_id: str = "analyst_cli_demo",
        simulate_fp_retest: bool = False,
        simulate_rollback: bool = False,
        advance_ttl_minutes: float = 0.0,
        record_feedback: bool = False,
        export_dir: Optional[Path] = None,
        output_json: Optional[Path] = None,
        input_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Expected one of {PROFILES}.")

        # Resolve environment/criticality defaults per profile
        if environment is None:
            if profile == "lab_r2_soft_containment":
                env = "lab"
            elif profile == "r4_crypto_remediation":
                env = None  # preserve fixture env ("production"/"staging")
            else:
                env = "production"
        else:
            env = environment

        if asset_criticality is None:
            if profile == "lab_r2_soft_containment":
                crit = "medium"
            elif profile == "r4_crypto_remediation":
                crit = None
            else:
                crit = "high"
        else:
            crit = asset_criticality

        investigations, source_desc = self._load_investigations(
            profile=profile,
            environment=env or "production",
            asset_criticality=crit or "medium",
            max_incidents=max_incidents,
        )

        # Profile-driven behavioral defaults in non-interactive mode
        effective_decision = approval_decision
        if effective_decision is None:
            if profile in {"approve_r3_verify_retest", "reject_fp_rollback", "r4_crypto_remediation"}:
                effective_decision = "approve"
            else:
                effective_decision = "deny"

        effective_fp_retest = simulate_fp_retest or (profile == "reject_fp_rollback")
        effective_ttl = float(advance_ttl_minutes)
        if not math.isfinite(effective_ttl) or effective_ttl < 0.0:
            raise ValueError("advance_ttl_minutes must be finite and >= 0.")
        if profile == "lab_r2_soft_containment" and effective_ttl == 0.0:
            effective_ttl = 30.0
        effective_feedback = record_feedback or (
            profile in {"approve_r3_verify_retest", "reject_fp_rollback", "r4_crypto_remediation"}
        )

        self._emit("=" * 72)
        self._emit("Topic 04 End-to-End Offline Demo (Metadata -> Hunting -> Response)")
        self._emit(
            f"Mode={'INTERACTIVE' if interactive else 'NON-INTERACTIVE'} | "
            f"Profile={profile} | DefaultPosture=DENY_BY_DEFAULT"
        )
        self._emit(f"Source={source_desc} | IncidentsLoaded={len(investigations)}")
        self._emit("=" * 72)

        prompt_fn = input_fn or input
        incident_summaries: List[Dict[str, Any]] = []
        stop_interactive = False

        for idx, item in enumerate(investigations, start=1):
            plan = self.planner.plan_from_investigation(
                item,
                environment=env,
                asset_criticality=crit,
            )
            self._emit(
                f"\n[{idx}/{len(investigations)}] Incident={plan.incident_id} | "
                f"Host={plan.target_host_token} | Tier={plan.tier} ({plan.tier_name}) | "
                f"Env={plan.environment}/{plan.asset_criticality} | "
                f"HumanGate={plan.human_approval_required}"
            )
            self._emit(
                f"  ProposedActions={list(plan.selected_actions)} | "
                f"PolicyStatus={plan.policy_status}"
            )

            approval_obj: Optional[ApprovalDecision] = None
            do_manual_rollback = simulate_rollback
            do_fp_retest = effective_fp_retest
            do_feedback = effective_feedback

            if interactive:
                if not stop_interactive and plan.human_approval_required:
                    self._emit(
                        "  [HUMAN GATE SR-07] Enter [a]pprove / [r]eject / [d]eny(default) / [q]uit:"
                    )
                    try:
                        choice = str(prompt_fn("  decision> ")).strip().lower()
                    except EOFError:
                        choice = "d"
                    if choice in {"q", "quit"}:
                        stop_interactive = True
                        approval_obj = None
                    elif choice in {"a", "approve"}:
                        approval_obj = ApprovalDecision(
                            decision="approve",
                            approved=True,
                            reviewer_id=reviewer_id,
                            reason="Interactive analyst approval.",
                        )
                    elif choice in {"r", "reject"}:
                        approval_obj = ApprovalDecision(
                            decision="reject",
                            approved=False,
                            reviewer_id=reviewer_id,
                            reason="Interactive analyst rejection.",
                        )
                    else:
                        approval_obj = None
                elif not stop_interactive:
                    self._emit("  [POLICY AUTO-BOUNDED] No production human gate required.")
                else:
                    approval_obj = None
            else:
                if plan.human_approval_required:
                    if effective_decision == "approve":
                        approval_obj = ApprovalDecision(
                            decision="approve",
                            approved=True,
                            reviewer_id=reviewer_id,
                            reason=f"CLI profile '{profile}' simulated analyst approval.",
                        )
                    elif effective_decision == "reject":
                        approval_obj = ApprovalDecision(
                            decision="reject",
                            approved=False,
                            reviewer_id=reviewer_id,
                            reason=f"CLI profile '{profile}' simulated analyst rejection.",
                        )
                    else:
                        approval_obj = None  # DENY_BY_DEFAULT

            gate_dec = self.gate.evaluate(plan, approval=approval_obj)
            action_rec = self.simulator.execute_plan(plan, gate_dec)
            self._emit(
                f"  Gate={gate_dec.status} | Outcome={action_rec.outcome} | "
                f"NetworkMutated={action_rec.network_state_mutated}"
            )

            # Interactive post-execution options (rollback / Stage 7 FP / Stage 8 feedback)
            if interactive and not stop_interactive and action_rec.rollback_tokens:
                self._emit(
                    "  [POST-ACTION] Enter [c]ontinue(verify) / [b]rollback / [f]alse-positive-retest / [q]uit:"
                )
                try:
                    post_choice = str(prompt_fn("  post> ")).strip().lower()
                except EOFError:
                    post_choice = "c"
                if post_choice in {"q", "quit"}:
                    stop_interactive = True
                elif post_choice in {"b", "rollback"}:
                    do_manual_rollback = True
                elif post_choice in {"f", "fp"}:
                    do_fp_retest = True

            rollback_res_list: List[Dict[str, Any]] = []
            if do_manual_rollback and action_rec.rollback_tokens:
                for tok in action_rec.rollback_tokens:
                    rb = self.simulator.rollback(
                        tok,
                        actor=reviewer_id,
                        reason="CLI manual rollback.",
                    )
                    rollback_res_list.append(rb.to_dict())
                    self._emit(f"  [ROLLBACK] token={tok} success={rb.success}")

            # Stage 7 Re-test
            if do_fp_retest:
                obs = RetestObservation(
                    window_id=999,
                    post_max_s_total=0.14,
                    post_alerted_flow_count=0,
                    benign_traffic_disruption_ratio=0.40,
                    false_positive_confirmed=True,
                    notes="Simulated false positive in subsequent window.",
                )
            elif plan.tier == "R4":
                obs = RetestObservation(
                    window_id=999,
                    post_max_s_total=0.02,
                    post_alerted_flow_count=0,
                    crypto_conformance_passed=True,
                )
            else:
                obs = RetestObservation(
                    window_id=999,
                    post_max_s_total=0.10 if action_rec.network_state_mutated else plan.max_s_total,
                    post_alerted_flow_count=0 if action_rec.network_state_mutated else plan.alerted_flow_count,
                )

            retest_res = self.verifier.verify(action_rec, obs, self.simulator)
            self._emit(
                f"  Stage7Retest={retest_res.outcome} | "
                f"AutoRollback={retest_res.auto_rollback_triggered}"
            )

            # Stage 8 Analyst-Approved Feedback
            fb_status = "not_recorded"
            if interactive:
                if not stop_interactive:
                    self._emit("  [STAGE 8] Record analyst-approved feedback? [y/N]:")
                    try:
                        fb_choice = str(prompt_fn("  feedback> ")).strip().lower()
                    except EOFError:
                        fb_choice = "n"
                    if fb_choice in {"q", "quit"}:
                        stop_interactive = True
                        do_feedback = False
                    else:
                        do_feedback = fb_choice in {"y", "yes"}
                else:
                    do_feedback = False

            if do_feedback and (
                (approval_obj is not None and approval_obj.decision in {"approve", "reject"})
                or not plan.human_approval_required
            ):
                verified_lbl = (
                    "false_positive"
                    if (retest_res.false_positive_detected or gate_dec.status == "REJECTED_BY_ANALYST")
                    else ("benign_policy_drift" if plan.tier == "R4" else "true_positive")
                )
                fb_res = self.feedback.submit_feedback(
                    AnalystFeedbackRecord(
                        incident_id=plan.incident_id,
                        plan_id=plan.plan_id,
                        recommended_tier=plan.tier,
                        analyst_approved=True,
                        reviewer_id=reviewer_id,
                        verified_label=verified_lbl,  # type: ignore[arg-type]
                        action_acceptable=(
                            not retest_res.false_positive_detected
                            and gate_dec.status != "REJECTED_BY_ANALYST"
                        ),
                        notes=f"Demo CLI feedback (profile={profile}).",
                    ),
                    simulator=self.simulator,
                )
                fb_status = fb_res.status
                self._emit(f"  Stage8Feedback={fb_status}")

            incident_summaries.append(
                {
                    "incident_id": plan.incident_id,
                    "tier": plan.tier,
                    "environment": plan.environment,
                    "asset_criticality": plan.asset_criticality,
                    "target_host_token": plan.target_host_token,
                    "policy_status": plan.policy_status,
                    "gate_status": gate_dec.status,
                    "outcome": action_rec.outcome,
                    "network_state_mutated": action_rec.network_state_mutated,
                    "rollback_tokens": list(action_rec.rollback_tokens),
                    "manual_rollbacks": rollback_res_list,
                    "retest_outcome": retest_res.outcome,
                    "auto_rollback_triggered": retest_res.auto_rollback_triggered,
                    "remediation_diffs": [d.to_dict() for d in action_rec.remediation_diffs],
                    "stage8_feedback_status": fb_status,
                }
            )
            if stop_interactive:
                self._emit("  [INTERACTIVE QUIT] Halting remaining incidents under DENY_BY_DEFAULT.")
                break

        expired_entries = []
        if effective_ttl > 0.0:
            expired = self.simulator.advance_time(effective_ttl)
            expired_entries = [e.to_dict() for e in expired]
            self._emit(
                f"\n[TTL CLOCK] Advanced logical clock by {effective_ttl:.1f}m -> "
                f"ExpiredEntries={len(expired_entries)}"
            )

        summary: Dict[str, Any] = sanitize_json_value(
            {
                "schema_version": "1.0.0",
                "generated_at_utc": _utc_now_iso(),
                "mode": "interactive" if interactive else "non_interactive",
                "profile": profile,
                "default_posture": "DENY_BY_DEFAULT",
                "source": source_desc,
                "incidents_processed": len(incident_summaries),
                "active_containment_count": len(self.simulator.active_containments),
                "expired_on_ttl_count": len(expired_entries),
                "remediation_diff_count": len(self.simulator.remediation_diffs),
                "audit_event_count": len(self.simulator.audit_trail),
                "stage8_feedback_count": len(self.feedback.records),
                "os_commands_generated_or_executed": 0,
                "incident_summaries": incident_summaries,
                "expired_entries": expired_entries,
            }
        )

        sanitized_summary, _ = validate_week10_payload(
            summary,
            strict=True,
            strip_ground_truth=True,
            policy_guard=self.policy_guard,
        )
        json.dumps(sanitized_summary, allow_nan=False)

        if export_dir is not None:
            exp = Path(export_dir)
            exp.mkdir(parents=True, exist_ok=True)
            (exp / "demo_summary.json").write_text(
                json.dumps(sanitized_summary, indent=2, allow_nan=False), encoding="utf-8"
            )
            self.simulator.export_audit_json(exp / "audit_trail.json")
            self.feedback.export_jsonl(exp / "stage8_feedback.jsonl")
            self._emit(f"[OK] Exported demo artifacts to: {exp}")

        if output_json is not None:
            out_path = Path(output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(sanitized_summary, indent=2, allow_nan=False), encoding="utf-8"
            )
            self._emit(f"[OK] Saved demo summary JSON to: {out_path}")

        self._emit(
            f"\n[DONE] Incidents={len(incident_summaries)} | "
            f"ActiveContainments={len(self.simulator.active_containments)} | "
            f"AuditEvents={len(self.simulator.audit_trail)} | "
            f"Stage8Feedback={len(self.feedback.records)} | OSCommands=0"
        )
        return sanitized_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Topic 04 Week 10 End-to-End Offline Demo CLI (Metadata -> Hunting -> Response)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive analyst decision prompts.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run deterministically without interactive stdin prompts (default).",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="deny_by_default",
        help="Preset offline demo scenario profile (default: deny_by_default).",
    )
    parser.add_argument(
        "--environment",
        choices=("lab", "staging", "production"),
        default=None,
        help="Override target environment scope.",
    )
    parser.add_argument(
        "--asset-criticality",
        choices=("low", "medium", "high", "critical"),
        default=None,
        help="Override asset criticality.",
    )
    parser.add_argument(
        "--approval-decision",
        choices=("deny", "approve", "reject"),
        default=None,
        help="Explicit simulated analyst decision for human-gated plans (defaults to DENY_BY_DEFAULT).",
    )
    parser.add_argument(
        "--reviewer-id",
        default="analyst_cli_demo",
        help="Pseudonymized/role reviewer ID for simulated analyst decisions.",
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        default=5,
        help="Maximum incidents to process (0 = all).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional Parquet path (defaults to sample NF-UQ-NIDS-v2 test split).",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Optional Weeks 5-8 checkpoint directory.",
    )
    parser.add_argument(
        "--week9-report",
        type=Path,
        default=None,
        help="Optional Week 9 incident evidence graph JSON to replay directly.",
    )
    parser.add_argument(
        "--simulate-fp-retest",
        action="store_true",
        help="Simulate a false-positive Stage 7 observation window triggering auto-rollback.",
    )
    parser.add_argument(
        "--simulate-rollback",
        action="store_true",
        help="Execute explicit analyst rollback after simulated containment.",
    )
    parser.add_argument(
        "--advance-ttl-minutes",
        type=float,
        default=0.0,
        help="Advance simulator logical clock in minutes to test TTL auto-expiry.",
    )
    parser.add_argument(
        "--record-feedback",
        action="store_true",
        help="Record analyst-approved Stage 8 feedback.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional directory to export demo_summary.json, audit_trail.json, and stage8_feedback.jsonl.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save demo summary JSON.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    interactive = bool(args.interactive and not args.non_interactive)
    runner = EndToEndDemoRunner(
        checkpoint_root=args.checkpoint_root,
        data_path=args.data_path,
        week9_report_path=args.week9_report,
    )
    runner.run(
        interactive=interactive,
        profile=args.profile,
        environment=args.environment,
        asset_criticality=args.asset_criticality,
        max_incidents=args.max_incidents,
        approval_decision=args.approval_decision,
        reviewer_id=args.reviewer_id,
        simulate_fp_retest=args.simulate_fp_retest,
        simulate_rollback=args.simulate_rollback,
        advance_ttl_minutes=args.advance_ttl_minutes,
        record_feedback=args.record_feedback,
        export_dir=args.export_dir,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
