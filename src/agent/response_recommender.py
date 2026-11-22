"""
Read-Only Response Tier Recommender (`R0`-`R4`) for Topic 04 Week 9.

Loads `src/response/response_policy_tiers.json` (`DENY_BY_DEFAULT`), evaluates
verified findings against environment (`lab`, `staging`, `production`) and asset
criticality (`low`, `medium`, `high`, `critical`), and attaches read-only
recommendations (`execution_status="PROPOSED_ONLY_NOT_EXECUTED"`).
Week 10 execution and re-test/learn actions remain strictly out of scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.agent.policy_guard import ScopePolicyGuard
from src.agent.state import PolicyRecommendation, VerificationResult
from src.hunting.schemas import (
    AttackMapping,
    IncidentCluster,
    flow_is_cr01,
    flow_is_cr02,
    flow_is_cr04,
)

HIGH_IMPACT_TECHNIQUES = frozenset({"T1557", "T1573", "T1071.001", "T1021.004", "T1041"})
SOFT_CONTAINMENT_TECHNIQUES = frozenset({"T1110", "T1046", "T1498"})


class ReadOnlyResponseRecommender:
    """Maps verified findings to read-only R0-R4 policy recommendations."""

    def __init__(self, policy_path: Optional[Union[str, Path]] = None) -> None:
        if policy_path is not None:
            self.policy_path = Path(policy_path)
        else:
            self.policy_path = (
                Path(__file__).resolve().parents[1]
                / "response"
                / "response_policy_tiers.json"
            )
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Response policy file not found: {self.policy_path}")

        with open(self.policy_path, "r", encoding="utf-8") as f:
            self.policy_data: Dict[str, Any] = json.load(f)

        self.policy_version: str = str(self.policy_data.get("policy_version", "1.0.0"))
        self.default_posture: str = str(self.policy_data.get("default_posture", "DENY_BY_DEFAULT"))
        self.tiers_by_id: Dict[str, Dict[str, Any]] = {
            str(t["tier_id"]): t for t in self.policy_data.get("tiers", [])
        }
        for req in ("R0", "R1", "R2", "R3", "R4"):
            if req not in self.tiers_by_id:
                raise ValueError(f"Missing required policy tier '{req}' in {self.policy_path}")

    def _build_recommendation(
        self,
        tier_id: str,
        environment: str,
        asset_criticality: str,
        rationale: str,
        constraints_applied: List[str],
    ) -> PolicyRecommendation:
        tier = self.tiers_by_id[tier_id]
        env = environment.lower()
        crit = asset_criticality.lower()

        if tier_id == "R3":
            if env == "production":
                actions = list(tier.get("allowed_actions_production", []))
                human_approval = bool(tier.get("human_approval_required_in_production", True))
            else:
                actions = list(tier.get("allowed_actions_lab", []))
                human_approval = bool(crit == "critical")
        else:
            actions = list(tier.get("allowed_actions", []))
            human_approval = bool(tier.get("human_approval_required", False))

        # R4 always requires human approval and forbids automatic production merge
        if tier_id == "R4":
            human_approval = True

        network_change = bool(tier.get("network_change_allowed", False))
        auto_expire = tier.get("auto_expire_minutes")
        rollback_req = bool(tier.get("rollback_required", False))
        retest_req = bool(tier.get("retest_after_action", False))
        auto_merge = tier.get("automatic_production_merge_allowed")

        return PolicyRecommendation(
            recommended_tier=tier_id,  # type: ignore[arg-type]
            tier_name=str(tier["name"]),
            environment=env,
            asset_criticality=crit,
            allowed_actions=actions,
            proposed_actions=actions,
            network_change_allowed=network_change,
            human_approval_required=human_approval,
            auto_expire_minutes=int(auto_expire) if auto_expire is not None else None,
            rollback_required=rollback_req,
            retest_after_action=retest_req,
            automatic_production_merge_allowed=(
                bool(auto_merge) if auto_merge is not None else None
            ),
            rationale=rationale,
            policy_version=self.policy_version,
            default_posture=self.default_posture,
            execution_status="PROPOSED_ONLY_NOT_EXECUTED",
            policy_constraints_applied=constraints_applied,
        )

    def recommend(
        self,
        verification: VerificationResult,
        cluster: IncidentCluster,
        attack_mappings: Sequence[AttackMapping],
        environment: str = "production",
        asset_criticality: str = "medium",
    ) -> PolicyRecommendation:
        ScopePolicyGuard.check_environment_scope(environment, asset_criticality)
        env = environment.strip().lower()
        crit = asset_criticality.strip().lower()

        constraints: List[str] = [
            f"default_posture={self.default_posture}",
            "read_only_week9_no_network_execution",
        ]
        mapped_ids = {m.technique_id for m in attack_mappings}
        active_set = set(cluster.active_reason_codes)
        has_crypto_policy_rule = bool(
            (
                active_set
                & {
                    "RULE_CR01_DEPRECATED_TLS_VERSION",
                    "RULE_CR02_CERT_TRUST_ANOMALY",
                    "RULE_CR04_WEAK_SSH_KEX",
                }
            )
            or any(flow_is_cr01(f) or flow_is_cr02(f) or flow_is_cr04(f) for f in cluster.flows)
        )
        threshold = max(0.45, max((f.threshold for f in cluster.flows), default=0.45))

        # 1. Crypto policy regression without active verified exploitation -> R4
        if verification.policy_anomaly_only or (
            has_crypto_policy_rule and not verification.verified_malicious
        ):
            constraints.append("R4_human_approval_required_and_auto_merge_disabled")
            return self._build_recommendation(
                tier_id="R4",
                environment=env,
                asset_criticality=crit,
                rationale=(
                    "Cryptographic policy violation (CR-01/CR-02/CR-04) detected without confirmed "
                    "active multi-channel compromise; proposing R4 config remediation diff and lab conformance retest."
                ),
                constraints_applied=constraints,
            )

        # 2. Verified malicious attack -> R3 or R2 (constrained by environment/criticality)
        if verification.verified_malicious:
            needs_r3 = bool(
                (mapped_ids & HIGH_IMPACT_TECHNIQUES)
                or (
                    verification.severity == "critical"
                    and len(verification.corroborating_channels) >= 3
                    and env == "production"
                )
                or (
                    verification.severity in {"high", "critical"}
                    and len(verification.corroborating_channels) >= 2
                    and (
                        cluster.max_s_total >= 0.85
                        or crit in {"high", "critical"}
                        or env == "production"
                    )
                    and not (mapped_ids and mapped_ids.issubset(SOFT_CONTAINMENT_TECHNIQUES))
                )
            )
            if needs_r3:
                if env == "production":
                    constraints.append("R3_production_requires_human_approval_SR07")
                else:
                    constraints.append("R3_lab_staging_bounded_proposal")
                return self._build_recommendation(
                    tier_id="R3",
                    environment=env,
                    asset_criticality=crit,
                    rationale=(
                        f"Verified multi-channel threat (severity={verification.severity}, "
                        f"techniques={sorted(mapped_ids) or ['multi_channel_anomaly']}, "
                        f"channels={verification.corroborating_channels}); proposing R3 hard containment "
                        f"with rollback and retest."
                    ),
                    constraints_applied=constraints,
                )

            # Bounded brute-force / scan / DoS -> R2 in lab/staging, constrained in production
            r2_allowed_envs = {
                str(e).lower() for e in self.tiers_by_id["R2"].get("allowed_environments", [])
            }
            if env in r2_allowed_envs and crit != "critical":
                constraints.append("R2_soft_containment_allowed_in_lab_staging_with_auto_expire_30m")
                return self._build_recommendation(
                    tier_id="R2",
                    environment=env,
                    asset_criticality=crit,
                    rationale=(
                        f"Verified bounded threat ({sorted(mapped_ids) or verification.corroborating_channels}) "
                        f"in '{env}' environment (criticality={crit}); proposing R2 soft containment with 30m auto-expire."
                    ),
                    constraints_applied=constraints,
                )

            constraints.append(
                f"R2_denied_in_{env}_or_critical_asset_per_DENY_BY_DEFAULT_constrained_to_R1"
            )
            return self._build_recommendation(
                tier_id="R1",
                environment=env,
                asset_criticality=crit,
                rationale=(
                    f"R2 soft containment is restricted to {sorted(r2_allowed_envs)} (and non-critical assets) "
                    f"under DENY_BY_DEFAULT; constraining bounded finding in '{env}' (criticality={crit}) to R1 Alert/Ticket."
                ),
                constraints_applied=constraints,
            )

        # 3. Unverified / single-channel / moderate anomaly above threshold -> R1 Alert / Ticket
        if cluster.max_s_total >= threshold or active_set:
            constraints.append("R1_alert_only_for_unverified_or_single_channel_anomaly")
            return self._build_recommendation(
                tier_id="R1",
                environment=env,
                asset_criticality=crit,
                rationale=(
                    f"Anomaly or single-channel signal above threshold (max_s_total={cluster.max_s_total:.3f}) "
                    "without multi-channel verification; creating R1 analyst ticket without network changes."
                ),
                constraints_applied=constraints,
            )

        # 4. Benign / ECH-only (CR-03) / Fingerprint-only (CR-05) / below threshold -> R0 Observe
        constraints.append("R0_observe_only_benign_or_neutral_privacy_fingerprint")
        return self._build_recommendation(
            tier_id="R0",
            environment=env,
            asset_criticality=crit,
            rationale=(
                "No verified malicious behavior or policy violation (including neutral ECH CR-03 / "
                "fingerprint-only CR-05 checks); maintaining R0 Observe posture."
            ),
            constraints_applied=constraints,
        )
