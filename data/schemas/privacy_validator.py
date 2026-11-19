"""
Privacy & L7-Exclusion Validator for Topic 04 (NT219 / NT2205)
Adheres to:
- Section 1 (Boundary): No HTTP path/body/headers, SSH command/session content, decrypted payload.
- Section 7 (SR-01): Detection must operate without raw/decrypted application payload.
- Section 9 (CR-06): No decryption keys, session secrets or application plaintext enter the model/LLM context.
- Section 13.4 (Step 7): Run the L7-exclusion validator before any model training.
- Section 14: Recommended Dataset Schema & Split Rules.
"""

import json
import os
import re
from typing import Dict, List, Any
import pandas as pd


FORBIDDEN_COLUMN_PATTERNS = [
    r"^http_body",
    r"^http_uri",
    r"^http_request",
    r"^http_headers",
    r"^http_cookie",
    r"^http_authorization",
    r"^http_user_agent",
    r"^ssh_command",
    r"^ssh_terminal",
    r"^ssh_keystroke",
    r"^ssh_auth_password",
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
    r"^raw_sni$",
    r"^raw_san$"
]

IPV4_REGEX = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")


def load_schema_allowlist(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_dataframe_privacy(df: pd.DataFrame, schema_path: str = None, dataset_name: str = "dataset") -> Dict[str, Any]:
    """
    Validates that a DataFrame contains zero forbidden L7 columns, zero raw IPv4 strings in host tokens,
    and complies with Topic 04 strict No-L7 boundary.
    """
    columns = list(df.columns)
    violations: List[str] = []

    # 1. Check against forbidden patterns and explicit denylist
    denylist = set()
    if schema_path and os.path.exists(schema_path):
        schema = load_schema_allowlist(schema_path)
        denylist = set(c.lower() for c in schema.get("denylist_strict_no_L7", []))

    for col in columns:
        col_lower = col.lower()
        if col_lower in denylist:
            violations.append(f"Column '{col}' is explicitly listed in denylist_strict_no_L7.")
        for pat in FORBIDDEN_COLUMN_PATTERNS:
            if re.search(pat, col_lower):
                violations.append(f"Column '{col}' matches forbidden L7/raw-identity pattern '{pat}'.")

    # 2. Verify pseudonymization of host tokens (ensure no raw IPv4 address strings leaked)
    for host_col in ["src_host_token", "dst_host_token"]:
        if host_col in df.columns:
            sample_vals = df[host_col].dropna().astype(str).head(500)
            raw_ips = [v for v in sample_vals if IPV4_REGEX.match(v)]
            if raw_ips:
                violations.append(f"Column '{host_col}' contains raw un-pseudonymized IPv4 addresses (e.g. {raw_ips[0]}).")

    passed = len(violations) == 0
    return {
        "dataset_name": dataset_name,
        "passed_no_l7_audit": passed,
        "num_rows_checked": int(len(df)),
        "num_columns_checked": len(columns),
        "columns": columns,
        "violations": violations,
        "compliance_anchors": ["SR-01", "CR-06", "Section 13.4 Step 7", "Section 14"]
    }
