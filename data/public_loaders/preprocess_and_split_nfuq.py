"""
NF-UQ-NIDS-v2 Preprocessing, Privacy Normalizer & Multi-Model Dataset Splitter
Topic 04 — Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response
Course: NT219 / NT2205 — Cryptography and Applications

Adheres strictly to:
1. Section 12 (Dataset Plan): Stratified subsets from NF-UQ-NIDS-v2 (~75.99M records); split by original dataset/time, not random rows only.
2. Section 4, 5, 8, 13.4 (Privacy Normalizer & No-L7):
   - Drop L7/application columns: L7_PROTO, DNS_QUERY_ID, DNS_QUERY_TYPE, DNS_TTL_ANSWER, FTP_COMMAND_RET_CODE.
   - Pseudonymize IPV4_SRC_ADDR -> src_host_token and IPV4_DST_ADDR -> dst_host_token via keyed HMAC-SHA256.
   - Map L4_SRC_PORT -> src_port_class (0: well_known, 1: registered, 2: ephemeral) and retain dst_port.
3. Section 14 (Recommended Dataset Schema & Split Rules):
   - Temporal / Sequential Order Split (Early windows -> Train 70%, Middle -> Val 15%, Later -> Test 15%).
   - Host / Device Holdout: Dedicated set of src_host_token reserved exclusively for Test.
   - Attack-Family Holdout (Unseen Attack): Dedicated attack classes held out from supervised Train to evaluate Anomaly Detection (Isolation Forest).
   - Dual-Model Readiness:
     * Tabular format (`flows_tabular.parquet`) for Random Forest / XGBoost / Isolation Forest.
     * Temporal sequence format (`sequences_temporal/window_*.parquet`) sorted by `(window_id, src_host_token, flow_order_id)` for LSTM / 1D-CNN.
     * By-label subdirectories (`by_label/<Attack_Label>/data.parquet`) for per-class inspection.
"""

import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Allow importing privacy_validator from data/schemas
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "schemas"))
from privacy_validator import validate_dataframe_privacy

HMAC_SECRET_KEY = b"topic04-nt219-pseudonymization-secret-key-2026"

# Explicit L7 / application-semantic columns in NF-UQ-NIDS-v2 that MUST be dropped (SR-01, CR-06)
L7_COLUMNS_TO_DROP = [
    "L7_PROTO",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE"
]

# Pure L3/L4 numerical flow features retained for Tabular ML (Random Forest/XGBoost) & Sequence ML (LSTM)
L3_L4_FEATURE_COLUMNS = [
    "src_port_class",
    "dst_port",
    "proto",
    "in_bytes",
    "in_pkts",
    "out_bytes",
    "out_pkts",
    "tcp_flags",
    "client_tcp_flags",
    "server_tcp_flags",
    "flow_duration_ms",
    "duration_in",
    "duration_out",
    "min_ttl",
    "max_ttl",
    "longest_flow_pkt",
    "shortest_flow_pkt",
    "min_ip_pkt_len",
    "max_ip_pkt_len",
    "src_to_dst_second_bytes",
    "dst_to_src_second_bytes",
    "retransmitted_in_bytes",
    "retransmitted_in_pkts",
    "retransmitted_out_bytes",
    "retransmitted_out_pkts",
    "src_to_dst_avg_throughput",
    "dst_to_src_avg_throughput",
    "num_pkts_up_to_128_bytes",
    "num_pkts_128_to_256_bytes",
    "num_pkts_256_to_512_bytes",
    "num_pkts_512_to_1024_bytes",
    "num_pkts_1024_to_1514_bytes",
    "tcp_win_max_in",
    "tcp_win_max_out",
    "icmp_type",
    "icmp_ipv4_type",
    "is_tls_port",
    "is_ssh_port",
    "bytes_ratio",
    "pkts_ratio"
]

# Attack families held out from supervised training to test Unseen-Attack Anomaly Detection (Section 14)
UNSEEN_ATTACK_FAMILIES = {"Worms", "Shellcode", "Ransomware"}

# MITRE ATT&CK behavioral context mapping for NF-UQ-NIDS-v2 attack classes (Phase 3 / Section 6)
MITRE_ATTACK_MAPPING = {
    "Benign": "NONE",
    "Brute Force": "T1110 (Brute Force)",
    "Bot": "T1573 / T1071.001 (Encrypted Channel / C2 Beacon Context)",
    "Infilteration": "T1021 / T1071 (Lateral Movement / C2)",
    "MITM": "T1557 (Adversary-in-the-Middle)",
    "scanning": "T1046 (Network Service Discovery)",
    "Reconnaissance": "T1046 (Network Service Discovery)",
    "DoS": "T1498 (Network Denial of Service)",
    "DDoS": "T1498 (Network Denial of Service)",
    "Backdoor": "T1573 / T1071 (Command and Control)",
    "Analysis": "T1046 (Network Service Discovery)",
    "Fuzzers": "T1499 (Endpoint Denial of Service / Fuzzing)",
    "Exploits": "T1190 (Exploit Public-Facing Application)",
    "Generic": "T1190 (Exploit / Generic Intrusion)",
    "Theft": "T1041 (Exfiltration Over C2 Channel)",
    "Worms": "T1210 (Exploitation of Remote Services - Unseen Holdout)",
    "Shellcode": "T1059 (Command and Scripting Interpreter - Unseen Holdout)",
    "Ransomware": "T1486 (Data Encrypted for Impact - Unseen Holdout)"
}


def pseudonymize_ip(ip_series: pd.Series) -> pd.Series:
    """Deterministic keyed HMAC-SHA256 pseudonymization for unique IPs (fast vectorized mapping)."""
    unique_ips = ip_series.dropna().unique()
    mapping = {}
    for ip in unique_ips:
        digest = hmac.new(HMAC_SECRET_KEY, str(ip).encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        mapping[ip] = f"host_{digest}"
    return ip_series.map(mapping).fillna("host_unknown")


def classify_src_port(port_series: pd.Series) -> pd.Series:
    """Maps raw source port to RFC 6335 port classes: 0=well_known (0-1023), 1=registered (1024-49151), 2=ephemeral (49152+)."""
    ports = pd.to_numeric(port_series, errors="coerce").fillna(0).astype(np.int64)
    conds = [ports < 1024, (ports >= 1024) & (ports < 49152)]
    choices = [0, 1]
    return pd.Series(np.select(conds, choices, default=2), index=port_series.index, dtype=np.int8)


def sanitize_label_name(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(label)).strip("_")


def stream_and_sample_nfuq(
    raw_csv_path: str,
    target_total_rows: int = 650000,
    chunksize: int = 1000000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Streams the 13.7GB NF-UQ-NIDS-v2.csv in chunks, prioritizing TLS (443/8443), SSH (22/2222),
    and balanced stratified representation across all attack families and temporal windows.
    """
    rng = np.random.default_rng(random_seed)
    sampled_chunks: List[pd.DataFrame] = []
    class_counts: Dict[str, int] = {}

    # Per-class quota to prevent majority classes (DDoS/DoS/Benign) from drowning out rare attacks (MITM, Bot, Brute Force)
    max_per_major_class = 160000  # For Benign
    max_per_attack_class = 45000  # For each attack category

    global_row_offset = 0
    chunk_idx = 0
    t0 = time.time()

    print(f"[INFO] Streaming raw dataset from: {raw_csv_path}")
    for chunk in pd.read_csv(raw_csv_path, chunksize=chunksize, low_memory=False):
        chunk_idx += 1
        chunk_len = len(chunk)
        # Preserve original sequential/temporal flow order ID and temporal window ID
        chunk["flow_order_id"] = np.arange(global_row_offset, global_row_offset + chunk_len, dtype=np.int64)
        chunk["window_id"] = np.int32(chunk_idx)
        global_row_offset += chunk_len

        # Prioritize TLS/SSH ports (22, 443, 8443, 2222) + stratified sample per Attack class
        selected_indices = []
        for attack_name, group in chunk.groupby("Attack", observed=True):
            attack_str = str(attack_name)
            current_cnt = class_counts.get(attack_str, 0)
            quota = max_per_major_class if attack_str == "Benign" else max_per_attack_class
            remaining_quota = max(0, quota - current_cnt)
            if remaining_quota == 0:
                continue

            # Always prioritize TLS (443, 8443) and SSH (22, 2222) flows within each class
            dst_p = pd.to_numeric(group["L4_DST_PORT"], errors="coerce").fillna(0)
            src_p = pd.to_numeric(group["L4_SRC_PORT"], errors="coerce").fillna(0)
            crypto_mask = dst_p.isin([22, 443, 8443, 2222, 993, 995]) | src_p.isin([22, 443, 8443, 2222])

            crypto_group = group[crypto_mask]
            other_group = group[~crypto_mask]

            # Take up to per-chunk cap so samples span across all temporal windows (early, middle, late)
            per_chunk_cap = max(500, quota // 18)
            take_n = min(len(group), remaining_quota, per_chunk_cap)

            if len(crypto_group) >= take_n:
                chosen = crypto_group.sample(n=take_n, random_state=int(rng.integers(0, 1_000_000)))
            else:
                need_more = take_n - len(crypto_group)
                if len(other_group) > need_more:
                    extra = other_group.sample(n=need_more, random_state=int(rng.integers(0, 1_000_000)))
                    chosen = pd.concat([crypto_group, extra], axis=0)
                else:
                    chosen = group

            selected_indices.append(chosen)
            class_counts[attack_str] = current_cnt + len(chosen)

        if selected_indices:
            chunk_sample = pd.concat(selected_indices, axis=0)
            sampled_chunks.append(chunk_sample)

        total_collected = sum(class_counts.values())
        elapsed = time.time() - t0
        print(f"  -> Chunk {chunk_idx:02d} ({global_row_offset:,} rows scanned in {elapsed:.1f}s) | Collected: {total_collected:,} rows across {len(class_counts)} classes")

        if total_collected >= target_total_rows and chunk_idx >= 25:
            # Ensure we scanned enough chunks for rich temporal windows
            break

    df = pd.concat(sampled_chunks, axis=0, ignore_index=True)
    # Sort strictly by flow_order_id to preserve true sequential order for LSTM and temporal split
    df.sort_values("flow_order_id", inplace=True, ignore_index=True)
    print(f"[INFO] Completed streaming. Total stratified subset rows: {len(df):,}")
    return df


def normalize_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Topic 04 Privacy Normalizer & L3/L4 Feature Engineering:
    1. Drops L7 columns (L7_PROTO, DNS_*, FTP_*).
    2. Pseudonymizes IPV4_SRC_ADDR & IPV4_DST_ADDR into src_host_token & dst_host_token.
    3. Computes derived L3/L4 features aligned with Section 14 schema.
    """
    print("[INFO] Running Privacy Normalizer & L7-Exclusion Filter...")
    # 1. Drop L7 columns
    cols_to_drop = [c for c in L7_COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Pseudonymize raw IPv4 addresses into host tokens
    df["src_host_token"] = pseudonymize_ip(df["IPV4_SRC_ADDR"])
    df["dst_host_token"] = pseudonymize_ip(df["IPV4_DST_ADDR"])
    df = df.drop(columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])

    # 3. Classify source port and standardize column names to Section 14 schema
    df["src_port_class"] = classify_src_port(df["L4_SRC_PORT"])
    df["dst_port"] = pd.to_numeric(df["L4_DST_PORT"], errors="coerce").fillna(0).astype(np.int32)
    df = df.drop(columns=["L4_SRC_PORT", "L4_DST_PORT"])

    rename_map = {
        "PROTOCOL": "proto",
        "IN_BYTES": "in_bytes",
        "IN_PKTS": "in_pkts",
        "OUT_BYTES": "out_bytes",
        "OUT_PKTS": "out_pkts",
        "TCP_FLAGS": "tcp_flags",
        "CLIENT_TCP_FLAGS": "client_tcp_flags",
        "SERVER_TCP_FLAGS": "server_tcp_flags",
        "FLOW_DURATION_MILLISECONDS": "flow_duration_ms",
        "DURATION_IN": "duration_in",
        "DURATION_OUT": "duration_out",
        "MIN_TTL": "min_ttl",
        "MAX_TTL": "max_ttl",
        "LONGEST_FLOW_PKT": "longest_flow_pkt",
        "SHORTEST_FLOW_PKT": "shortest_flow_pkt",
        "MIN_IP_PKT_LEN": "min_ip_pkt_len",
        "MAX_IP_PKT_LEN": "max_ip_pkt_len",
        "SRC_TO_DST_SECOND_BYTES": "src_to_dst_second_bytes",
        "DST_TO_SRC_SECOND_BYTES": "dst_to_src_second_bytes",
        "RETRANSMITTED_IN_BYTES": "retransmitted_in_bytes",
        "RETRANSMITTED_IN_PKTS": "retransmitted_in_pkts",
        "RETRANSMITTED_OUT_BYTES": "retransmitted_out_bytes",
        "RETRANSMITTED_OUT_PKTS": "retransmitted_out_pkts",
        "SRC_TO_DST_AVG_THROUGHPUT": "src_to_dst_avg_throughput",
        "DST_TO_SRC_AVG_THROUGHPUT": "dst_to_src_avg_throughput",
        "NUM_PKTS_UP_TO_128_BYTES": "num_pkts_up_to_128_bytes",
        "NUM_PKTS_128_TO_256_BYTES": "num_pkts_128_to_256_bytes",
        "NUM_PKTS_256_TO_512_BYTES": "num_pkts_256_to_512_bytes",
        "NUM_PKTS_512_TO_1024_BYTES": "num_pkts_512_to_1024_bytes",
        "NUM_PKTS_1024_TO_1514_BYTES": "num_pkts_1024_to_1514_bytes",
        "TCP_WIN_MAX_IN": "tcp_win_max_in",
        "TCP_WIN_MAX_OUT": "tcp_win_max_out",
        "ICMP_TYPE": "icmp_type",
        "ICMP_IPV4_TYPE": "icmp_ipv4_type",
        "Label": "label",
        "Attack": "attack_family",
        "Dataset": "ground_truth_source"
    }
    df = df.rename(columns=rename_map)

    # Clean numeric columns (replace inf/NaN with 0)
    num_cols = [c for c in rename_map.values() if c not in ("attack_family", "ground_truth_source")]
    for col in num_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0)
            df[col] = s.astype(np.float32 if "second_bytes" in col or "throughput" in col else np.int64)

    # Derived cryptographic-port & flow-symmetry indicators (L3/L4 context)
    df["is_tls_port"] = df["dst_port"].isin([443, 8443, 993, 995]).astype(np.int8)
    df["is_ssh_port"] = df["dst_port"].isin([22, 2222]).astype(np.int8)
    df["bytes_ratio"] = ((df["in_bytes"] + 1.0) / (df["out_bytes"] + 1.0)).astype(np.float32)
    df["pkts_ratio"] = ((df["in_pkts"] + 1.0) / (df["out_pkts"] + 1.0)).astype(np.float32)

    # Session ID & MITRE ATT&CK mapping (Section 14)
    df["session_id"] = "nfuq_" + df["flow_order_id"].astype(str)
    df["attack_label"] = df["attack_family"].astype(str)
    df["attack_mapping"] = df["attack_family"].map(MITRE_ATTACK_MAPPING).fillna("T1190 (Network Intrusion)")
    df["attack_layers"] = np.where(df["is_tls_port"] == 1, "L3_L4_TLS_PORT", np.where(df["is_ssh_port"] == 1, "L3_L4_SSH_PORT", "L3_L4_FLOW"))

    # Reorder columns cleanly
    ordered_cols = (
        ["session_id", "flow_order_id", "window_id", "src_host_token", "dst_host_token"]
        + L3_L4_FEATURE_COLUMNS
        + ["label", "attack_family", "attack_label", "attack_layers", "attack_mapping", "ground_truth_source"]
    )
    df = df[[c for c in ordered_cols if c in df.columns]]
    return df


def split_dataset_multimodel(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Executes the 3 mandatory split rules from Section 14:
    1. Attack-Family Holdout (Unseen Attacks): Exclude UNSEEN_ATTACK_FAMILIES from supervised Train/Val and place in `unseen_attack_holdout` + Test.
    2. Host Holdout: Hold out ~10% of unique `src_host_token` exclusively for Test.
    3. Temporal Split (within each attack class & source dataset):
       - Early 70% flow_order_id -> Train
       - Middle 15% flow_order_id -> Val
       - Late 15% flow_order_id -> Test
    Preserves sequential ordering for LSTM while providing tabular sets for Random Forest / XGBoost.
    """
    print("[INFO] Applying Section 14 Split Rules (Temporal + Host Holdout + Unseen-Attack Holdout)...")

    # 1. Separate Unseen Attack Holdout
    unseen_mask = df["attack_family"].isin(UNSEEN_ATTACK_FAMILIES)
    df_unseen = df[unseen_mask].copy()
    df_known = df[~unseen_mask].copy()

    # 2. Identify Host Holdout tokens (~8% of hosts reserved strictly for Test)
    unique_hosts = df_known["src_host_token"].unique()
    # Deterministic hash selection of hosts for holdout
    holdout_hosts = set(h for h in unique_hosts if int(hashlib.md5(h.encode()).hexdigest()[:4], 16) % 12 == 0)

    train_parts, val_parts, test_parts = [], [], []

    # 3. Within each known attack family, sort by flow_order_id (temporal order) and split 70 / 15 / 15
    for attack_name, group in df_known.groupby("attack_family", observed=True):
        group = group.sort_values("flow_order_id")
        host_holdout_group = group[group["src_host_token"].isin(holdout_hosts)]
        regular_group = group[~group["src_host_token"].isin(holdout_hosts)]

        n = len(regular_group)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_slice = regular_group.iloc[:n_train]
        val_slice = regular_group.iloc[n_train:n_train + n_val]
        test_slice = pd.concat([regular_group.iloc[n_train + n_val:], host_holdout_group], axis=0)

        train_parts.append(train_slice)
        val_parts.append(val_slice)
        test_parts.append(test_slice)

    df_train = pd.concat(train_parts, axis=0).sort_values("flow_order_id").reset_index(drop=True)
    df_val = pd.concat(val_parts, axis=0).sort_values("flow_order_id").reset_index(drop=True)
    df_test = pd.concat(test_parts + [df_unseen], axis=0).sort_values("flow_order_id").reset_index(drop=True)

    # Re-assign contiguous sequence window bins (10 windows per split) for easy LSTM batch loading
    for split_df, prefix in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        split_df["time_bucket"] = pd.qcut(
            split_df["flow_order_id"].rank(method="first"),
            q=min(10, max(1, len(split_df) // 5000)),
            labels=False
        ).astype(np.int16)

    split_summary = {
        "total_rows": int(len(df)),
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "test_rows": int(len(df_test)),
        "unseen_attack_holdout_rows": int(len(df_unseen)),
        "unseen_attack_families": sorted(list(UNSEEN_ATTACK_FAMILIES)),
        "num_holdout_hosts_in_test": len(holdout_hosts),
        "train_class_distribution": df_train["attack_family"].value_counts().to_dict(),
        "val_class_distribution": df_val["attack_family"].value_counts().to_dict(),
        "test_class_distribution": df_test["attack_family"].value_counts().to_dict()
    }
    return df_train, df_val, df_test, df_unseen, split_summary


def save_split_directory(split_df: pd.DataFrame, split_dir: Path, split_name: str) -> None:
    """
    Saves each split (`train/`, `val/`, `test/`) in 3 ways to serve both Random Forest and LSTM:
    1. `flows_tabular.parquet`: Combined table for Random Forest / XGBoost / Isolation Forest.
    2. `by_label/<Attack_Name>/flows.parquet`: Subfolders by attack label.
    3. `sequences_temporal/window_XX/sequences.parquet`: Subfolders by temporal window sorted by `(src_host_token, flow_order_id)` for LSTM sliding windows.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combined Tabular file (Random Forest / XGBoost / Isolation Forest)
    tabular_path = split_dir / "flows_tabular.parquet"
    split_df.to_parquet(tabular_path, index=False)

    # 2. Subfolders by Attack Label (`by_label/<Attack_Name>/flows.parquet`)
    by_label_root = split_dir / "by_label"
    for label_name, group in split_df.groupby("attack_family", observed=True):
        safe_label = sanitize_label_name(str(label_name))
        label_dir = by_label_root / safe_label
        label_dir.mkdir(parents=True, exist_ok=True)
        group.to_parquet(label_dir / "flows.parquet", index=False)

    # 3. Subfolders by Temporal Window (`sequences_temporal/window_XX/sequences.parquet`) sorted for LSTM
    seq_root = split_dir / "sequences_temporal"
    for bucket_id, group in split_df.groupby("time_bucket", observed=True):
        win_dir = seq_root / f"window_{int(bucket_id):02d}"
        win_dir.mkdir(parents=True, exist_ok=True)
        # Sort by (src_host_token, flow_order_id) so consecutive rows belong to the same host's temporal sequence
        seq_group = group.sort_values(["src_host_token", "flow_order_id"]).reset_index(drop=True)
        seq_group.to_parquet(win_dir / "sequences.parquet", index=False)

    print(f"  -> Saved [{split_name.upper()}] ({len(split_df):,} rows) to {split_dir}")


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    seq_len: int = 8,
    stride: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper utility for Week 7 (Temporal Sequence Model - LSTM / 1D-CNN / GRU):
    Converts the preprocessed DataFrame into 3D sequences `(N_sequences, seq_len, N_features)`
    grouped by `(src_host_token, time_bucket)` in chronological `flow_order_id` order.
    """
    if feature_cols is None:
        feature_cols = L3_L4_FEATURE_COLUMNS

    df_sorted = df.sort_values(["src_host_token", "flow_order_id"])
    X_seqs, y_seqs = [], []

    for _, group in df_sorted.groupby("src_host_token", observed=True):
        vals = group[feature_cols].to_numpy(dtype=np.float32)
        labels = group["label"].to_numpy(dtype=np.int64)
        n = len(vals)
        if n < seq_len:
            continue
        for start in range(0, n - seq_len + 1, stride):
            X_seqs.append(vals[start:start + seq_len])
            # Sequence label: 1 if any flow in the window is malicious, else 0
            y_seqs.append(int(np.max(labels[start:start + seq_len])))

    if not X_seqs:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_seqs, axis=0), np.array(y_seqs, dtype=np.int64)


def run_pipeline(
    raw_csv_path: str,
    output_dir: str,
    repo_sample_dir: str,
    target_rows: int = 650000
) -> Dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Stream & stratified sample
    df_raw = stream_and_sample_nfuq(raw_csv_path, target_total_rows=target_rows)

    # Step 2: Privacy Normalizer & L7 Exclusion
    df_clean = normalize_and_engineer_features(df_raw)

    # Step 3: Mandatory L7-Exclusion Privacy Validator Audit (Section 13.4 Step 7)
    schema_path = str(REPO_ROOT / "data" / "schemas" / "schema_allowlist.json")
    audit_report = validate_dataframe_privacy(df_clean, schema_path=schema_path, dataset_name="NF-UQ-NIDS-v2_Preprocessed")
    if not audit_report["passed_no_l7_audit"]:
        raise ValueError(f"Privacy Validator FAILED! Violations: {audit_report['violations']}")
    print("[PASS] L7-Exclusion Privacy Validator passed with 0 violations!")

    # Step 4: Execute Multi-Model Split (Temporal + Host Holdout + Unseen Attack Holdout)
    df_train, df_val, df_test, df_unseen, split_summary = split_dataset_multimodel(df_clean)

    # Step 5: Compute StandardScaler parameters on TRAIN set only (prevent data leakage)
    scaler_stats = {}
    for col in L3_L4_FEATURE_COLUMNS:
        mean_val = float(df_train[col].mean())
        std_val = float(df_train[col].std())
        scaler_stats[col] = {"mean": mean_val, "std": std_val if std_val > 1e-9 else 1.0}

    # Step 6: Save train/, val/, test/ directory structure
    save_split_directory(df_train, out_path / "train", "train")
    save_split_directory(df_val, out_path / "val", "val")
    save_split_directory(df_test, out_path / "test", "test")

    # Save dedicated Unseen-Attack Holdout file inside test/ for Anomaly Detection evaluation
    if len(df_unseen) > 0:
        df_unseen.to_parquet(out_path / "test" / "unseen_attack_holdout.parquet", index=False)

    # Step 7: Save manifest, scaler stats, and privacy audit report
    manifest = {
        "dataset": "NF-UQ-NIDS-v2 (Topic 04 Preprocessed & Multi-Model Split)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_columns_L3_L4": L3_L4_FEATURE_COLUMNS,
        "num_features": len(L3_L4_FEATURE_COLUMNS),
        "dropped_L7_columns": L7_COLUMNS_TO_DROP,
        "pseudonymization": "HMAC-SHA256 (16-hex prefix) on IPV4_SRC_ADDR and IPV4_DST_ADDR",
        "split_summary": split_summary,
        "privacy_audit": audit_report,
        "train_scaler_stats": scaler_stats
    }

    with open(out_path / "metadata_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Step 8: Save a lightweight 1,500-row sample & manifest into the GitHub Repo so tests/demos run out-of-the-box
    sample_dir = Path(repo_sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    df_sample = pd.concat([
        df_train.head(800),
        df_val.head(350),
        df_test.head(350)
    ], axis=0, ignore_index=True)
    df_sample.to_parquet(sample_dir / "nfuq_sample_1500.parquet", index=False)
    with open(sample_dir / "nfuq_manifest_and_audit.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Full dataset saved to: {out_path}")
    print(f"[SUCCESS] Lightweight repo sample saved to: {sample_dir}")
    return manifest


if __name__ == "__main__":
    raw_csv = r"D:\Crypto\DoAn\9810e03bba4983da_MOHANAD_A4706\data\NF-UQ-NIDS-v2.csv"
    out_dir = r"D:\Crypto\DoAn\NF-UQ-NIDS-v2"
    repo_sample = str(REPO_ROOT / "data" / "public_loaders" / "sample_data")
    run_pipeline(raw_csv, out_dir, repo_sample, target_rows=650000)
