"""
Offline-First Narrative Generation & Pluggable LLM Adapter for Topic 04 Week 9.

Default mode uses `OfflineTemplateNarrativeGenerator` (100% deterministic, zero
external API keys). Optional `CallableLLMNarrativeAdapter` allows plugging in a
mock or future LLM callable guarded by `ScopePolicyGuard` pre-validation so no
raw L7 payload, raw IP, or secret material can leave the boundary (`SR-01`, `CR-06`).
LLM narrative output is strictly informational and NEVER alters Stage 4 severity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from src.agent.policy_guard import ScopePolicyGuard


class BaseNarrativeGenerator(ABC):
    """Abstract interface for Stage 5 incident narrative generation."""

    @abstractmethod
    def generate(self, sanitized_context: Dict[str, Any]) -> Dict[str, Any]:
        """Return structured narrative metadata dict from sanitized investigation context."""


class OfflineTemplateNarrativeGenerator(BaseNarrativeGenerator):
    """100% offline, deterministic template-based explainer citing verified observables only."""

    def generate(self, sanitized_context: Dict[str, Any]) -> Dict[str, Any]:
        incident_id = sanitized_context.get("incident_id", "unknown")
        host = sanitized_context.get("primary_host_token", "unknown")
        window_id = sanitized_context.get("window_id", 0)
        triage = sanitized_context.get("triage_summary", {})
        enrichment = sanitized_context.get("enrichment", {})
        verification = sanitized_context.get("verification", {})
        mappings = sanitized_context.get("attack_mappings", [])
        unmapped_rationale = sanitized_context.get("unmapped_rationale")
        policy = sanitized_context.get("policy_recommendation", {})

        max_s = float(triage.get("max_s_total", 0.0))
        dom_sig = triage.get("dominant_signal", "unknown")
        reasons = triage.get("active_reason_codes", [])
        l5_avail = enrichment.get("l5_l6_availability", {})
        l5_mode = "Zeek L5/L6 enriched" if l5_avail.get("available") else "L3/L4 metadata-only"
        ech_note = " ECH observed and treated neutrally per CR-03." if l5_avail.get("ech_observed_any") else ""

        if mappings:
            tech_str = ", ".join(f"{m['technique_id']} ({m['technique_name']})" for m in mappings)
        else:
            tech_str = f"None ({unmapped_rationale or 'insufficient behavioral evidence'})"

        narrative = (
            f"Incident {incident_id} on host {host} (window {window_id}, {l5_mode}): "
            f"max fusion score s_total={max_s:.3f} (dominant_signal={dom_sig}, "
            f"active_reason_codes={reasons or ['none']}).{ech_note} "
            f"Stage 4 deterministic verification set verified_malicious={verification.get('verified_malicious', False)}, "
            f"severity={verification.get('severity', 'info')}, "
            f"corroborating_channels={verification.get('corroborating_channels', [])}. "
            f"MITRE ATT&CK mapping(s): {tech_str}. "
            f"Recommended read-only response tier: {policy.get('recommended_tier', 'R0')} "
            f"({policy.get('tier_name', 'Observe')}), "
            f"human_approval_required={policy.get('human_approval_required', False)}, "
            f"execution_status={policy.get('execution_status', 'PROPOSED_ONLY_NOT_EXECUTED')}."
        )
        return {
            "narrative": narrative,
            "provider": "offline_template",
            "model": "deterministic-v1",
            "offline": True,
            "llm_enabled": False,
            "severity_override_blocked": True,
        }


class CallableLLMNarrativeAdapter(BaseNarrativeGenerator):
    """
    Optional pluggable LLM narrative adapter.
    Disabled/offline-fallback by default when `llm_callable is None`.
    All contexts pass through `ScopePolicyGuard` before invoking `llm_callable`.
    """

    def __init__(
        self,
        llm_callable: Optional[Callable[[Dict[str, Any]], str]] = None,
        policy_guard: Optional[ScopePolicyGuard] = None,
        provider_name: str = "pluggable_callable",
        model_name: str = "custom",
        fallback_generator: Optional[BaseNarrativeGenerator] = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.policy_guard = policy_guard or ScopePolicyGuard()
        self.provider_name = provider_name
        self.model_name = model_name
        self.fallback_generator = fallback_generator or OfflineTemplateNarrativeGenerator()

    def generate(self, sanitized_context: Dict[str, Any]) -> Dict[str, Any]:
        clean_ctx, audit = self.policy_guard.validate_and_sanitize_payload(
            sanitized_context, strict=True, strip_ground_truth=True
        )
        if self.llm_callable is None:
            result = self.fallback_generator.generate(clean_ctx)
            result["guard_audit"] = audit.to_dict()
            return result

        try:
            raw_text = str(self.llm_callable(clean_ctx))
        except Exception as exc:
            result = self.fallback_generator.generate(clean_ctx)
            result["fallback_used"] = True
            clean_exc, _ = self.policy_guard.redact_secrets_in_text(str(exc))
            result["fallback_reason"] = f"{type(exc).__name__}: {clean_exc}"
            result["guard_audit"] = audit.to_dict()
            return result

        clean_text, _ = self.policy_guard.redact_secrets_in_text(raw_text)
        return {
            "narrative": clean_text,
            "provider": self.provider_name,
            "model": self.model_name,
            "offline": False,
            "llm_enabled": True,
            "fallback_used": False,
            "severity_override_blocked": True,
            "guard_audit": audit.to_dict(),
        }
