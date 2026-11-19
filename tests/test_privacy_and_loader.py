"""
Unit Tests for Topic 04 Privacy Validator & Multi-Model Dataset Loader
Verifies compliance with SR-01, CR-06, Section 13.4 Step 7, and Section 14.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data" / "schemas"))
sys.path.insert(0, str(REPO_ROOT / "data" / "public_loaders"))

from privacy_validator import validate_dataframe_privacy
from preprocess_and_split_nfuq import build_lstm_sequences, L3_L4_FEATURE_COLUMNS


def test_privacy_validator_blocks_l7_and_raw_ip():
    bad_df = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "src_host_token": ["192.168.1.10", "10.0.0.5"],  # Raw IP leakage!
        "http_body": ["secret_payload", "password=123"],  # Forbidden L7 column!
        "in_bytes": [100, 200]
    })
    res = validate_dataframe_privacy(bad_df)
    assert res["passed_no_l7_audit"] is False, "Validator should fail when raw IPs or L7 columns exist!"
    assert len(res["violations"]) >= 2
    print("[PASS] test_privacy_validator_blocks_l7_and_raw_ip")


def test_repo_sample_compliance_and_dual_model_loading():
    sample_path = REPO_ROOT / "data" / "public_loaders" / "sample_data" / "nfuq_sample_1500.parquet"
    if not sample_path.exists():
        print("[SKIP] Sample file not generated yet.")
        return

    df = pd.read_parquet(sample_path)
    schema_path = str(REPO_ROOT / "data" / "schemas" / "schema_allowlist.json")
    audit = validate_dataframe_privacy(df, schema_path=schema_path)
    assert audit["passed_no_l7_audit"] is True, f"Sample failed privacy audit: {audit['violations']}"

    # 1. Test Tabular shape for Random Forest / XGBoost
    X_tab = df[L3_L4_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_tab = df["label"].to_numpy(dtype=np.int64)
    assert X_tab.shape == (len(df), len(L3_L4_FEATURE_COLUMNS))
    assert len(y_tab) == len(df)

    # 2. Test 3D Sequence generation for LSTM / 1D-CNN
    X_seq, y_seq = build_lstm_sequences(df, seq_len=4, stride=2)
    assert X_seq.ndim == 3 and X_seq.shape[1] == 4 and X_seq.shape[2] == len(L3_L4_FEATURE_COLUMNS)
    assert len(X_seq) == len(y_seq)
    print(f"[PASS] test_repo_sample_compliance_and_dual_model_loading | Tabular: {X_tab.shape}, LSTM Sequences: {X_seq.shape}")


if __name__ == "__main__":
    test_privacy_validator_blocks_l7_and_raw_ip()
    test_repo_sample_compliance_and_dual_model_loading()
