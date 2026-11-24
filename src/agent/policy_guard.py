"""
Scope & Policy Guard for Topic 04 Week 9 Threat-Hunting Agent.

Enforces SR-01, SR-06, SR-07, CR-06, and `data/schemas/schema_allowlist.json` +
`data/schemas/privacy_validator.py` semantics before any agent processing or
LLM adapter invocation:
- Rejects forbidden raw Layer-7 fields (`http_*`, `ssh_command*`, `decrypted_*`, etc.).
- Permits valid L5/L6 Zeek metadata (`tls_version`, `cert_status`, `ja4`, `hassh`,
  `hassh_or_ja4ssh`, `ech_observed`, `ssh_kex_class`, etc.).
- Rejects raw IPv4 addresses in `src_host_token` / `dst_host_token`.
- Rejects or redacts decryption keys, PEM blocks, SSLKEYLOGFILE/CLIENT_RANDOM,
  Bearer tokens, and API keys (`CR-06`).
- Strips ground-truth columns before hunting/agent reasoning.
- Validates target environment and asset criticality scopes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from src.hunting.schemas import GROUND_TRUTH_COLUMNS, L5_L6_OBSERVABLE_FIELDS, FlowEvidence

ALLOWED_ENVIRONMENTS: FrozenSet[str] = frozenset({"lab", "staging", "production"})
ALLOWED_ASSET_CRITICALITIES: FrozenSet[str] = frozenset({"low", "medium", "high", "critical"})

IPV4_REGEX = re.compile(
    r"(?<![0-9.])(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?![0-9.])"
)

FORBIDDEN_FIELD_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^http_",
        r"^raw_http_",
        r"^ssh_command",
        r"^ssh_terminal",
        r"^ssh_keystroke",
        r"^ssh_auth_password",
        r"^ssh_password",
        r"^decrypted_",
        r"^tls_session_key",
        r"^private_key",
        r"^raw_pcap",
        r"^payload",
        r"^l7_proto$",
        r"^dns_query_",
        r"^dns_ttl_",
        r"^ftp_command_",
        r"^ipv4_src_addr$",
        r"^ipv4_dst_addr$",
        r"^src_ip$",
        r"^dst_ip$",
        r"^raw_sni$",
        r"^sni$",
        r"^raw_san$",
    )
)

SECRET_VALUE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("PEM_KEY_BLOCK", re.compile(r"-----BEGIN\s+[A-Z0-9 ]*KEY-----", re.IGNORECASE)),
    ("SSLKEYLOG_CLIENT_RANDOM", re.compile(r"\bCLIENT_RANDOM\s+[0-9a-fA-F]{16,}")),
    ("SSLKEYLOGFILE_REF", re.compile(r"\bSSLKEYLOGFILE\b")),
    ("BEARER_TOKEN", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{10,}")),
    ("OPENAI_STYLE_KEY", re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b")),
    ("INLINE_API_OR_SECRET_KEY", re.compile(r"(?i)\b(?:api_key|secret_key|tls_session_key|private_key)\s*[:=]\s*\S+")),
)


class ScopePolicyViolationError(ValueError):
    """Raised when telemetry or context violates Topic 04 No-L7 / No-Secret / Scope policy."""


@dataclass(frozen=True)
class PolicyAuditResult:
    """Auditable outcome of a ScopePolicyGuard check."""

    passed: bool
    violations: List[str] = field(default_factory=list)
    redactions_applied: List[str] = field(default_factory=list)
    stripped_ground_truth_columns: List[str] = field(default_factory=list)
    l5_l6_fields_present: List[str] = field(default_factory=list)
    compliance_anchors: List[str] = field(
        default_factory=lambda: ["SR-01", "SR-06", "SR-07", "CR-06", "Section 8", "Section 11"]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": list(self.violations),
            "redactions_applied": list(self.redactions_applied),
            "stripped_ground_truth_columns": list(self.stripped_ground_truth_columns),
            "l5_l6_fields_present": list(self.l5_l6_fields_present),
            "compliance_anchors": list(self.compliance_anchors),
        }


class ScopePolicyGuard:
    """Pre-stage security, privacy, and scope policy guard."""

    def __init__(self, schema_path: Optional[Union[str, Path]] = None) -> None:
        if schema_path is not None:
            self.schema_path: Optional[Path] = Path(schema_path)
        else:
            default_schema = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "schemas"
                / "schema_allowlist.json"
            )
            self.schema_path = default_schema if default_schema.is_file() else None

        self.denylist: FrozenSet[str] = self._load_denylist(self.schema_path)

    @staticmethod
    def _load_denylist(schema_path: Optional[Path]) -> FrozenSet[str]:
        base = {
            "http_body",
            "http_request_uri",
            "http_request_path",
            "http_headers_raw",
            "http_cookie",
            "http_authorization",
            "http_user_agent_raw",
            "ssh_commands",
            "ssh_terminal_keystrokes",
            "ssh_auth_password",
            "decrypted_tls_payload",
            "tls_session_keys",
            "private_keys",
            "raw_pcap_payload_bytes",
        }
        if schema_path is not None and schema_path.is_file():
            with open(schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            base.update(str(c).lower() for c in data.get("denylist_strict_no_L7", []))
        return frozenset(base)

    @staticmethod
    def check_environment_scope(environment: str, asset_criticality: str) -> None:
        env = str(environment).strip().lower()
        crit = str(asset_criticality).strip().lower()
        if env not in ALLOWED_ENVIRONMENTS:
            raise ScopePolicyViolationError(
                f"Out-of-scope environment '{environment}'. Allowed: {sorted(ALLOWED_ENVIRONMENTS)} (SR-06)."
            )
        if crit not in ALLOWED_ASSET_CRITICALITIES:
            raise ScopePolicyViolationError(
                f"Invalid asset_criticality '{asset_criticality}'. Allowed: {sorted(ALLOWED_ASSET_CRITICALITIES)}."
            )

    def _is_forbidden_field_name(self, name: str) -> Optional[str]:
        col_lower = str(name).strip().lower()
        if col_lower in self.denylist:
            return f"Field '{name}' is explicitly listed in denylist_strict_no_L7."
        for pat in FORBIDDEN_FIELD_PATTERNS:
            if pat.search(col_lower):
                return f"Field '{name}' matches forbidden L7/raw-identity pattern '{pat.pattern}'."
        return None

    @staticmethod
    def _detect_secrets(text: str) -> List[str]:
        hits: List[str] = []
        for label, pat in SECRET_VALUE_PATTERNS:
            if pat.search(text):
                hits.append(label)
        return hits

    @staticmethod
    def redact_secrets_in_text(text: str) -> Tuple[str, List[str]]:
        redacted = text
        applied: List[str] = []
        for label, pat in SECRET_VALUE_PATTERNS:
            if pat.search(redacted):
                redacted = pat.sub(f"[REDACTED_{label}]", redacted)
                applied.append(label)
        return redacted, applied

    def _scan_cell_for_violations(
        self,
        val: Any,
        col_name: str,
        seen_strs: set[str],
    ) -> Optional[str]:
        if isinstance(val, Mapping):
            for k, v in val.items():
                msg = self._is_forbidden_field_name(str(k))
                if msg:
                    return f"Column '{col_name}' contains forbidden nested field '{k}': {msg}"
                nested = self._scan_cell_for_violations(v, col_name, seen_strs)
                if nested:
                    return nested
            return None
        if isinstance(val, (list, tuple, set)):
            for item in val:
                nested = self._scan_cell_for_violations(item, col_name, seen_strs)
                if nested:
                    return nested
            return None
        s = str(val)
        if s in seen_strs:
            return None
        seen_strs.add(s)
        hits = self._detect_secrets(s)
        if hits:
            return f"Column '{col_name}' contains prohibited secret/key material ({', '.join(hits)})."
        return None

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        strip_ground_truth: bool = True,
        strict: bool = True,
    ) -> Tuple[pd.DataFrame, PolicyAuditResult]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

        violations: List[str] = []
        for col in df.columns:
            msg = self._is_forbidden_field_name(str(col))
            if msg:
                violations.append(msg)

        # Full-column check for raw or embedded IPv4 addresses in host tokens
        for host_col in ("src_host_token", "dst_host_token"):
            if host_col in df.columns:
                unique_vals = df[host_col].dropna().astype(str).unique()
                raw_ips = [v for v in unique_vals if IPV4_REGEX.search(v.strip())]
                if raw_ips:
                    violations.append(
                        f"Column '{host_col}' contains raw un-pseudonymized IPv4 address (e.g. '{raw_ips[0]}')."
                    )

        # Scan all unique values across object, string, and Categorical columns (CR-06 & nested L7 cells)
        scan_cols = [
            c
            for c in df.columns
            if (
                pd.api.types.is_object_dtype(df[c].dtype)
                or pd.api.types.is_string_dtype(df[c].dtype)
                or isinstance(df[c].dtype, pd.CategoricalDtype)
            )
        ]
        for col in scan_cols:
            seen_strs: set[str] = set()
            series = df[col].dropna()
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                for val in series.unique():
                    err = self._scan_cell_for_violations(val, str(col), seen_strs)
                    if err:
                        violations.append(err)
                        break
                continue

            try:
                unique_vals = series.unique()
                for val in unique_vals:
                    err = self._scan_cell_for_violations(val, str(col), seen_strs)
                    if err:
                        violations.append(err)
                        break
            except TypeError:
                # Object column contains unhashable dict/list cells; inspect every non-null cell
                for val in series:
                    err = self._scan_cell_for_violations(val, str(col), seen_strs)
                    if err:
                        violations.append(err)
                        break

        stripped_gt = [str(c) for c in df.columns if str(c).lower() in GROUND_TRUTH_COLUMNS]
        l5_l6_present = sorted(
            str(c) for c in df.columns if str(c).lower() in set(L5_L6_OBSERVABLE_FIELDS)
        )

        audit = PolicyAuditResult(
            passed=len(violations) == 0,
            violations=violations,
            stripped_ground_truth_columns=stripped_gt if strip_ground_truth else [],
            l5_l6_fields_present=l5_l6_present,
        )

        if strict and not audit.passed:
            raise ScopePolicyViolationError(f"ScopePolicyGuard rejected DataFrame: {violations}")

        if strip_ground_truth and stripped_gt:
            keep_cols = [c for c in df.columns if str(c).lower() not in GROUND_TRUTH_COLUMNS]
            df_out = df[keep_cols].copy()
        else:
            df_out = df.copy()

        return df_out, audit

    def validate_flows(
        self,
        flows: Sequence[FlowEvidence],
        strict: bool = True,
    ) -> PolicyAuditResult:
        violations: List[str] = []
        l5_l6_present_set: set[str] = set()

        for idx, flow in enumerate(flows):
            for field_name, token in (
                ("src_host_token", flow.src_host_token),
                ("dst_host_token", flow.dst_host_token),
            ):
                if IPV4_REGEX.search(str(token).strip()):
                    violations.append(
                        f"Flow[{idx}] {field_name}='{token}' contains raw IPv4 address; pseudonymized token required."
                    )
            str_fields: List[Tuple[str, Optional[str]]] = [
                ("src_host_token", flow.src_host_token),
                ("dst_host_token", flow.dst_host_token),
                ("session_id", flow.session_id),
                ("event_id", flow.event_id),
                ("ssh_kex_class", flow.ssh_kex_class),
                ("ssh_cipher_class", flow.ssh_cipher_class),
                ("ssh_mac_class", flow.ssh_mac_class),
                ("cert_issuer_class", flow.cert_issuer_class),
                ("tls_cipher_class", flow.tls_cipher_class),
                ("security_proto", flow.security_proto),
                ("ts_start", flow.ts_start),
                ("ja4", flow.ja4),
                ("ja3_optional", flow.ja3_optional),
                ("hassh", flow.hassh),
                ("cert_spki_token", flow.cert_spki_token),
                ("tls_version", flow.tls_version),
                ("cert_status", flow.cert_status),
                ("dominant_signal", flow.dominant_signal),
            ]
            for rc in flow.reason_codes:
                str_fields.append(("reason_codes", rc))
            for rc in flow.active_reason_codes:
                str_fields.append(("active_reason_codes", rc))

            for field_name, str_val in str_fields:
                if str_val:
                    hits = self._detect_secrets(str(str_val))
                    if hits:
                        violations.append(
                            f"Flow[{idx}].{field_name} contains prohibited secret/key material ({', '.join(hits)})."
                        )
            l5_l6_present_set.update(flow.present_l5_l6_fields())

        audit = PolicyAuditResult(
            passed=len(violations) == 0,
            violations=violations,
            l5_l6_fields_present=sorted(l5_l6_present_set),
        )
        if strict and not audit.passed:
            raise ScopePolicyViolationError(f"ScopePolicyGuard rejected flows: {violations}")
        return audit

    def validate_and_sanitize_payload(
        self,
        payload: Any,
        strict: bool = True,
        strip_ground_truth: bool = True,
    ) -> Tuple[Any, PolicyAuditResult]:
        """
        Recursively inspects dict/list/scalar context before agent/LLM calls:
        - Rejects forbidden L7 field keys (`SR-01`).
        - Rejects raw or embedded IPv4s in host-token keys.
        - Rejects (strict=True) or redacts (strict=False) key/secret material (`CR-06`).
        - Strips ground-truth leakage keys.
        """
        violations: List[str] = []
        redactions: List[str] = []
        stripped_gt: List[str] = []
        host_token_keys = {
            "src_host_token",
            "dst_host_token",
            "primary_host_token",
            "target_host_tokens",
            "ip",
            "src_ip",
            "dst_ip",
        }

        def _walk(obj: Any, path: str = "root") -> Any:
            if isinstance(obj, Mapping):
                cleaned: Dict[str, Any] = {}
                for k, v in obj.items():
                    key_str = str(k)
                    key_lower = key_str.lower()
                    if strip_ground_truth and key_lower in GROUND_TRUTH_COLUMNS:
                        if key_str not in stripped_gt:
                            stripped_gt.append(key_str)
                        continue
                    forbidden_msg = self._is_forbidden_field_name(key_str)
                    if forbidden_msg:
                        violations.append(f"{path}.{key_str}: {forbidden_msg}")
                        continue
                    if key_lower in host_token_keys:
                        if isinstance(v, str) and IPV4_REGEX.search(v.strip()):
                            violations.append(
                                f"{path}.{key_str}: raw IPv4 '{v}' is forbidden; pseudonymized token required."
                            )
                            continue
                        if isinstance(v, (list, tuple, set, frozenset)):
                            bad_item = next(
                                (
                                    item
                                    for item in v
                                    if isinstance(item, str) and IPV4_REGEX.search(item.strip())
                                ),
                                None,
                            )
                            if bad_item is not None:
                                violations.append(
                                    f"{path}.{key_str}: raw IPv4 '{bad_item}' is forbidden; pseudonymized token required."
                                )
                                continue
                    cleaned[key_str] = _walk(v, f"{path}.{key_str}")
                return cleaned
            if isinstance(obj, (list, tuple)):
                return [_walk(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            if isinstance(obj, (set, frozenset)):
                sanitized_items = [
                    _walk(item, f"{path}[{i}]")
                    for i, item in enumerate(
                        sorted(obj, key=lambda x: (type(x).__qualname__, repr(x)))
                    )
                ]
                try:
                    return type(obj)(sanitized_items)
                except TypeError:
                    return sanitized_items
            if isinstance(obj, str):
                hits = self._detect_secrets(obj)
                if hits:
                    if strict:
                        violations.append(f"{path}: prohibited secret/key material detected ({', '.join(hits)}).")
                    else:
                        new_str, applied = self.redact_secrets_in_text(obj)
                        redactions.extend(applied)
                        return new_str
                return obj
            return obj

        sanitized = _walk(payload)
        audit = PolicyAuditResult(
            passed=len(violations) == 0,
            violations=violations,
            redactions_applied=sorted(set(redactions)),
            stripped_ground_truth_columns=stripped_gt,
        )
        if strict and not audit.passed:
            raise ScopePolicyViolationError(f"ScopePolicyGuard rejected payload: {violations}")
        return sanitized, audit
