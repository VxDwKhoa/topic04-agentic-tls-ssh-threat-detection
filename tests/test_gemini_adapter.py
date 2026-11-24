"""Tests for Guardrailed GeminiNarrativeAdapter and API-key safety controls."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.agent.llm_interface import (
    DEFAULT_GEMINI_MODEL,
    GeminiNarrativeAdapter,
    load_dotenv_file,
)
from src.agent.policy_guard import ScopePolicyGuard, ScopePolicyViolationError
from src.response.demo_end_to_end import EndToEndDemoRunner


def _sample_sanitized_context() -> Dict[str, Any]:
    return {
        "incident_id": "inc_test_gemini_01",
        "primary_host_token": "host_aaaaaaaaaaaaaaaa",
        "window_id": 12,
        "triage_summary": {
            "max_s_total": 0.94,
            "dominant_signal": "classifier",
            "active_reason_codes": ["RULE_HIGH_FLOW_RATE", "RF_HIGH_PROB"],
        },
        "enrichment": {
            "l5_l6_availability": {"available": False, "ech_observed_any": False},
        },
        "verification": {
            "verified_malicious": True,
            "severity": "high",
            "corroborating_channels": ["classifier", "sequence"],
        },
        "attack_mappings": [
            {"technique_id": "T1573", "technique_name": "Encrypted Channel"}
        ],
        "unmapped_rationale": None,
        "policy_recommendation": {
            "recommended_tier": "R3",
            "tier_name": "Hard Containment Proposal",
            "human_approval_required": True,
            "execution_status": "PROPOSED_ONLY_NOT_EXECUTED",
        },
    }


def test_gemini_adapter_falls_back_cleanly_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adapter = GeminiNarrativeAdapter(dotenv_path=tmp_path / ".env")
    res = adapter.generate(_sample_sanitized_context())

    assert res["offline"] is True
    assert res["fallback_used"] is True
    assert res["severity_override_blocked"] is True
    assert res["requested_provider"] == "gemini"
    assert res["requested_model"] == DEFAULT_GEMINI_MODEL
    assert "Missing GEMINI_API_KEY" in res["fallback_reason"]
    assert "inc_test_gemini_01" in res["narrative"]


def test_gemini_adapter_calls_interactions_api_with_store_false_and_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_key = "AIza" + "SyFakeTestKey1234567890abcdef"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    captured_calls: List[Dict[str, Any]] = []

    class _FakeInteractions:
        def create(self, **kwargs: Any) -> Any:
            captured_calls.append(kwargs)
            return SimpleNamespace(
                output_text=(
                    f"- Su co muc do cao tren host_aaaaaaaaaaaaaaaa. "
                    f"Khong de lo key {fake_key}."
                )
            )

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.interactions = _FakeInteractions()

    ctx = _sample_sanitized_context()
    ctx["Label"] = 1  # Ground-truth field must be stripped before outbound prompt

    adapter = GeminiNarrativeAdapter(
        model_name="gemini-4-hydrogen",
        language="vi",
        client_factory=lambda k: _FakeClient(k),
    )
    res = adapter.generate(ctx)

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["model"] == "gemini-4-hydrogen"
    assert call["store"] is False
    assert "system_instruction" in call
    assert "Label" not in call["input"]
    assert res["provider"] == "gemini"
    assert res["offline"] is False
    assert res["llm_enabled"] is True
    assert res["fallback_used"] is False
    assert res["severity_override_blocked"] is True
    assert fake_key not in res["narrative"]
    assert "[REDACTED_GOOGLE_API_KEY]" in res["narrative"]


def test_gemini_adapter_blocks_raw_l7_and_raw_ip_before_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "SyFakeTestKey1234567890abcdef")
    called = False

    def _boom_client(_key: str) -> Any:
        nonlocal called
        called = True
        raise AssertionError("Gemini client must not be invoked when ScopePolicyGuard blocks payload.")

    adapter = GeminiNarrativeAdapter(client_factory=_boom_client)

    bad_l7 = _sample_sanitized_context()
    bad_l7["http_body"] = "GET /secret HTTP/1.1"
    with pytest.raises(ScopePolicyViolationError):
        adapter.generate(bad_l7)
    assert called is False

    bad_ip = _sample_sanitized_context()
    bad_ip["primary_host_token"] = "192.168.1.50"
    with pytest.raises(ScopePolicyViolationError):
        adapter.generate(bad_ip)
    assert called is False


def test_gemini_adapter_redacts_api_key_in_exception_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_key = "AIza" + "SySecretLeakTest1234567890xyz"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)

    def _failing_client(_key: str) -> Any:
        raise RuntimeError(f"Auth error with key {fake_key}")

    adapter = GeminiNarrativeAdapter(client_factory=_failing_client)
    res = adapter.generate(_sample_sanitized_context())

    assert res["fallback_used"] is True
    assert res["offline"] is True
    assert fake_key not in res["fallback_reason"]
    assert "[REDACTED_GOOGLE_API_KEY]" in res["fallback_reason"]


def test_load_dotenv_file_respects_existing_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nGEMINI_API_KEY=from_file_value\nGEMINI_MODEL=gemini-4-hydrogen\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "from_shell_value")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    loaded = load_dotenv_file(env_file, override=False)
    assert "GEMINI_API_KEY" not in loaded
    assert loaded.get("GEMINI_MODEL") == "[SET]"
    assert ScopePolicyGuard().redact_secrets_in_text("AIza" + "SyFakeTestKey1234567890abcdef")[0] == "[REDACTED_GOOGLE_API_KEY]"


def test_demo_cli_gemini_mode_runs_cleanly_offline_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    buf = io.StringIO()
    runner = EndToEndDemoRunner(
        output_stream=buf,
        llm_provider="gemini",
        llm_model="gemini-4-hydrogen",
        llm_language="vi",
    )
    summary = runner.run(
        interactive=False,
        profile="lab_r2_soft_containment",
        max_incidents=1,
    )
    output = buf.getvalue()
    assert summary["os_commands_generated_or_executed"] == 0
    assert "[AI ANALYST BRIEFING |" in output
    assert "[FALLBACK_OFFLINE]" in output
