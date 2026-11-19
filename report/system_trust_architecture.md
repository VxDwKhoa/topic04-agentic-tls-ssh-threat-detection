# Phase 1 & Phase 5 — System, Trust Boundary & Security Architecture

Tài liệu này định nghĩa **Sơ đồ Kiến trúc Hệ thống (System Architecture)** và **Ranh giới Tin cậy (Trust Boundaries)** cho **Topic 04: Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response**, tuân thủ nghiêm ngặt quy định **No Raw Layer-7 Payload** (SR-01, CR-06).

---

## 1. System & Trust Boundary Diagram (Phase 1)

Sơ đồ dưới đây phân tách rõ 6 vùng tin cậy (Trust Zones): từ vùng mạng chưa tin cậy (Client/Server Subnet), qua điểm đo tin cậy (Sensor), kho đặc trưng đã lọc L7 (Feature Store), vùng mô hình xác suất (ML/DL & Agent) đến bộ điều khiển phản hồi tin cậy cao (Response Controller & Human Approval Gate).

```mermaid
flowchart TD
    subgraph Z1["ZONE 1: Monitored Network (Untrusted / Potentially Compromised)"]
        C["Client / Endpoint Subnet<br/>(Workstations, IoT, Lab Containers)"]
        S["Server / External Subnet<br/>(Web/TLS Services, SSH Servers, External C2)"]
        C <==>|"Encrypted TLS 1.3 / SSH Traffic"| S
    end

    subgraph Z2["ZONE 2: Network Sensor Boundary (Trusted Measurement Point)"]
        TAP["SPAN / Network TAP Mirror"]
        ZEEK["Metadata Sensors: Zeek / Suricata / NetFlow<br/>(Payload logging OFF • No TLS Decryption)"]
        Z1 -.->|"Passive Traffic Mirror"| TAP --> ZEEK
    end

    subgraph Z3["ZONE 3: Data Governance & Feature Store (Privacy Boundary)"]
        VAL["Scope & Privacy Normalizer<br/>(Enforce schema_allowlist.json • Drop/HMAC Raw SNI & IP)"]
        FS[("Metadata-Only Feature Store<br/>L3/L4 Flow + L5/L6 TLS/SSH/X.509<br/>Zero Raw L7 Payload")]
        ZEEK -->|"conn.log, ssl.log, x509.log, ssh.log"| VAL --> FS
    end

    subgraph Z4["ZONE 4: Detection & Agentic Hunting (Untrusted Probabilistic Component)"]
        RULE["Deterministic Rule Engine<br/>(RFC 9846 TLS / RFC 9142 SSH Policy)"]
        ML["Multi-Layer ML/DL Detectors<br/>(XGBoost Tabular • Isolation Forest • Temporal LSTM)"]
        GRAPH["Evidence Graph & Hunt Correlator<br/>(Session/Host/Fingerprint/Cert Correlation)"]
        AGENT["AI Threat-Hunting Agent (ReAct Core)<br/>(MITRE ATT&CK Mapping • Evidence Verifier)"]
        
        FS --> RULE & ML
        RULE & ML -->|"Alerts + Calibrated Scores"| GRAPH
        GRAPH <-->|"Deterministic Tool Calls"| AGENT
    end

    subgraph Z5["ZONE 5: Response Controller (High-Trust Enforcement Boundary)"]
        PLAN["Policy-Constrained Response Planner<br/>(Deny-by-default • Confidence × Severity × Asset Policy)"]
        LAB["Lab / Staging Environment<br/>(Auto Bounded Containment / Rate-Limit)"]
        GATE{{"Production Human / SOAR Approval Gate<br/>(Mandatory Sign-off)"}}
        PROD["Production Action + Auto-Rollback Timer + Audit Trail"]
        RETEST["Re-test & Verification Loop"]

        AGENT -->|"Verified Finding + Action Proposal"| PLAN
        PLAN -->|"Lab Target"| LAB --> RETEST
        PLAN -->|"Production Impact"| GATE
        GATE -->|"Approved"| PROD --> RETEST
    end
```

---

## 2. Bảng Phân tích Ranh giới Tin cậy (Trust Assumptions Matrix)

| Vùng / Thành phần (Zone / Component) | Giả định Tin cậy (Trust Assumption) | Dữ liệu Quan sát & Ràng buộc An toàn (Required Observation & Safety Rule) |
| :--- | :--- | :--- |
| **1. Client / Endpoint Subnet** | **Untrusted:** Có thể chứa máy bị nhiễm mã độc (C2 beaconing) hoặc cấu hình sai. | Định danh ẩn danh (`src_host_token`), thống kê luồng, hành vi TLS/SSH fingerprint (`ja4`, `hassh`). |
| **2. Server / Service Subnet** | **Untrusted / Semi-trusted:** Dịch vụ có thể bị tấn công brute-force hoặc chứng chỉ bị thay đổi. | Port/Service, tóm tắt chứng chỉ X.509 (`cert_spki_token`), phiên bản giao thức, Cipher/KEX policy. |
| **3. Network Sensor (Zeek/Suricata)** | **Trusted Measurement Point:** Điểm thu thập độc lập, không lưu payload. | Chỉ xuất metadata L3–L6; tắt hoàn toàn packet/payload dumping; giám sát sensor health. |
| **4. Privacy Filter & Feature Store** | **Privacy Enforcement Boundary:** Chứa telemetry an ninh, tuyệt đối không chứa nội dung ứng dụng. | Kiểm duyệt bằng [schema_allowlist.json](../data/schemas/schema_allowlist.json), băm/ẩn danh SNI và IP nội bộ. |
| **5. ML/DL Models & Hunting Agent** | **Untrusted Probabilistic Component:** Mô hình có thể dự đoán sai hoặc bị hallucination. | Mọi kết luận phải gắn với bằng chứng (Evidence-grounded); không được tự ý thực thi lệnh chặn trực tiếp. |
| **6. Response Controller & Human Gate** | **High-Trust Component:** Điểm thực thi chính sách an toàn cuối cùng. | Chỉ cho phép các hành động trong allowlist; giới hạn thời gian (TTL/Rollback); bắt buộc **Human Approval** đối với production. |

---

## 3. Luồng Xử lý Multi-Layer Feature Fusion & Agentic Response (Phase 5)

```mermaid
sequenceDiagram
    autonumber
    participant Net as Encrypted Traffic (TLS/SSH)
    participant Zeek as Zeek / Suricata Sensor
    participant Val as Privacy Validator (No-L7)
    participant ML as Rules + ML/DL Detectors
    participant Agent as Threat-Hunting AI Agent
    participant Gate as Response Controller / Human Gate

    Net->>Zeek: TLS 1.3 / SSH Handshake & Encrypted Packets
    Note over Zeek: Extract L3/L4 Flow + L5/L6 Handshake<br/>Raw L7 Payload Discarded
    Zeek->>Val: Stream JSON Logs (conn, ssl, x509, ssh)
    Val->>Val: Validate against schema_allowlist.json
    Val->>ML: Normalized Feature Vector x_t (f_L3, f_L4, f_TLS/SSH, f_context)
    ML->>ML: Compute Rule Score + Classifier Prob + Anomaly/Sequence Score
    ML-->>Agent: Trigger High-Confidence Alert + Initial Evidence
    Agent->>Agent: Query Evidence Graph, Cert History & MITRE ATT&CK Matrix
    Agent->>Gate: Propose Bounded Response (e.g., Temp Isolate Host + Rollback Plan)
    alt Lab / Staging Target
        Gate->>Gate: Execute Bounded Lab Containment & Re-test
    else Production Target
        Gate->>Gate: Request Human / SOAR Approval before Execution
    end
```
