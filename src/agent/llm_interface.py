"""
Offline-First Narrative Generation & Pluggable LLM Adapter for Topic 04 Week 9.

Default mode uses `OfflineTemplateNarrativeGenerator` (100% deterministic, zero
external API keys). Optional `CallableLLMNarrativeAdapter` allows plugging in a
mock or future LLM callable guarded by `ScopePolicyGuard` pre-validation so no
raw L7 payload, raw IP, or secret material can leave the boundary (`SR-01`, `CR-06`).
LLM narrative output is strictly informational and NEVER alters Stage 4 severity.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.agent.policy_guard import ScopePolicyGuard

DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"


def load_dotenv_file(dotenv_path: Optional[Path] = None, *, override: bool = False) -> Dict[str, str]:
    """
    Lightweight local `.env` loader (zero external dependency on `python-dotenv`).
    Loads key=value pairs into `os.environ` only when present on local disk.
    Never logs or returns secret values in exception messages.
    """
    target = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    loaded: Dict[str, str] = {}
    if not target.is_file():
        return loaded

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        if not key or not val:
            continue
        if override or not os.environ.get(key):
            os.environ[key] = val
            loaded[key] = "[SET]"
    return loaded


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


class GeminiNarrativeAdapter(BaseNarrativeGenerator):
    """
    Guardrailed Gemini API adapter using the official `google-genai` Interactions API.

    Security & Privacy Guarantees (Topic 04 SR-01, SR-04, SR-06, SR-07, CR-06):
    - Reads API key strictly from environment (`GEMINI_API_KEY`) or local ignored `.env`.
    - Pre-validates & sanitizes context via `ScopePolicyGuard(strict=True, strip_ground_truth=True)`
      before any outbound API call (zero raw L7 payload, zero raw IPs, zero secrets).
    - Sets `store=False` so incident metadata is not persisted remotely.
    - Strictly informational: `severity_override_blocked=True`; NEVER alters Stage 4 severity
      or R0-R4 response policy tier.
    - Gracefully falls back to `OfflineTemplateNarrativeGenerator` if API key is missing,
      `google-genai` is not installed, or an API/network error occurs.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL,
        *,
        api_key_env: str = "GEMINI_API_KEY",
        language: str = "vi",
        max_output_tokens: int = 450,
        dotenv_path: Optional[Path] = None,
        policy_guard: Optional[ScopePolicyGuard] = None,
        fallback_generator: Optional[BaseNarrativeGenerator] = None,
        client_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.model_name = (model_name or DEFAULT_GEMINI_MODEL).strip()
        self.api_key_env = api_key_env
        self.language = "en" if str(language).strip().lower() == "en" else "vi"
        self.max_output_tokens = max(64, int(max_output_tokens))
        self.dotenv_path = dotenv_path
        self.policy_guard = policy_guard or ScopePolicyGuard()
        self.fallback_generator = fallback_generator or OfflineTemplateNarrativeGenerator()
        self._client_factory = client_factory

    def _resolve_api_key(self) -> Optional[str]:
        key = os.environ.get(self.api_key_env, "").strip()
        if key:
            return key
        # Check explicit dotenv_path or repo root `.env`
        candidate_env = self.dotenv_path
        if candidate_env is None:
            candidate_env = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv_file(candidate_env, override=False)
        key = os.environ.get(self.api_key_env, "").strip()
        return key or None

    def _build_system_instruction(self) -> str:
        lang_rule = (
            "Respond concisely in Vietnamese (3-5 bullet points)."
            if self.language == "vi"
            else "Respond concisely in English (3-5 bullet points)."
        )
        return (
            "You are a SOC Threat-Hunting Assistant for a metadata-only TLS/SSH detection system "
            "(Topic 04). Summarize ONLY the verified L3/L4 and L5/L6 metadata evidence provided in JSON. "
            "Strict rules: "
            "(1) Never infer or claim raw Layer-7 application payload content (HTTP body/URLs, SSH commands, credentials). "
            "(2) Never override or change the deterministic Stage 4 severity or R0-R4 policy tier. "
            "(3) Treat ECH (ech_observed=true) as a normal privacy feature, never as malicious evidence. "
            "(4) Highlight why human approval is or is not required and suggest 1-2 safe follow-up hunting checks. "
            f"{lang_rule}"
        )

    def _build_prompt(self, clean_ctx: Dict[str, Any]) -> str:
        serialized = json.dumps(clean_ctx, ensure_ascii=False, indent=2, default=str)
        return (
            "Analyze the following sanitized incident context and produce a brief SOC analyst briefing:\n"
            f"{serialized}"
        )

    def _fallback(
        self,
        clean_ctx: Dict[str, Any],
        audit: Any,
        reason: str,
    ) -> Dict[str, Any]:
        result = self.fallback_generator.generate(clean_ctx)
        clean_reason, _ = self.policy_guard.redact_secrets_in_text(reason)
        result["fallback_used"] = True
        result["fallback_reason"] = clean_reason
        result["requested_provider"] = "gemini"
        result["requested_model"] = self.model_name
        result["severity_override_blocked"] = True
        result["guard_audit"] = audit.to_dict()
        return result

    def generate(self, sanitized_context: Dict[str, Any]) -> Dict[str, Any]:
        clean_ctx, audit = self.policy_guard.validate_and_sanitize_payload(
            sanitized_context, strict=True, strip_ground_truth=True
        )
        api_key = self._resolve_api_key()
        if not api_key:
            return self._fallback(
                clean_ctx,
                audit,
                f"Missing {self.api_key_env} environment variable; using offline template.",
            )

        try:
            if self._client_factory is not None:
                client = self._client_factory(api_key)
            else:
                from google import genai  # type: ignore[import-not-found]

                client = genai.Client(api_key=api_key)

            interaction = client.interactions.create(
                model=self.model_name,
                system_instruction=self._build_system_instruction(),
                input=self._build_prompt(clean_ctx),
                store=False,
                generation_config={"max_output_tokens": self.max_output_tokens},
            )
            raw_text = str(getattr(interaction, "output_text", "") or "").strip()
            if not raw_text:
                return self._fallback(clean_ctx, audit, "Empty output_text returned by Gemini API.")
        except ImportError:
            return self._fallback(
                clean_ctx,
                audit,
                "Package 'google-genai' is not installed (run `pip install -U google-genai`).",
            )
        except Exception as exc:
            return self._fallback(clean_ctx, audit, f"{type(exc).__name__}: {exc}")

        clean_text, _ = self.policy_guard.redact_secrets_in_text(raw_text)
        return {
            "narrative": clean_text,
            "provider": "gemini",
            "model": self.model_name,
            "language": self.language,
            "offline": False,
            "llm_enabled": True,
            "fallback_used": False,
            "severity_override_blocked": True,
            "guard_audit": audit.to_dict(),
        }
