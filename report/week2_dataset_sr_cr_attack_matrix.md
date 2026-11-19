# Week 2 Deliverable — Dataset Matrix, Assets, Threat/ATT&CK Mapping & SR/CR Specification

Tài liệu này tổng hợp toàn bộ sản phẩm bắt buộc của **Tuần 2 (Week 2 Required Artifacts)** theo Mục 5 (Phase 2), Mục 6 (Phase 3), Mục 7 (Phase 4), Mục 9 (Phase 6), Mục 12, Mục 14, Mục 16 và Mục 19 của hướng dẫn đề tài **Topic 04**.

---

## 1. Dataset Selection & Split Matrix (Mục 12, 14 & 19 — Undergraduate Scope)

Theo phạm vi Đại học (Mục 19: *1 flow dataset + 1 TLS/SSH/lab dataset*), nhóm chốt sử dụng tổ hợp dữ liệu sau:

| Vai trò trong đồ án | Tên Dataset | Định dạng & Tín hiệu sử dụng | Mục đích huấn luyện & Kiểm thử | Quy tắc bảo vệ quyền riêng tư (No-L7) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Public Flow Dataset (Primary L3/L4 Source)** | **NF-UQ-NIDS-v2** *(Stratified Subset)* | NetFlow-v2 chuẩn hóa (43 flow features): `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`, `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`, `TCP_FLAGS`, `FLOW_DURATION_MILLISECONDS`, kèm nhãn `Label` và `Attack` (Brute Force, Scanning, DoS, MITM, Botnet...). | Huấn luyện và đánh giá Baseline Supervised Tabular ML (Random Forest / XGBoost) và Unsupervised Anomaly Detection (Isolation Forest) ở tầng L3/L4. | Chỉ chứa thống kê luồng NetFlow L3/L4, hoàn toàn không chứa gói tin PCAP hay tải trọng ứng dụng Layer 7. |
| **2. Lab Cryptographic Metadata Dataset (L5/L6 + L3/L4 Fusion)** | **Lab-Generated Zeek TLS 1.3 & SSH Stream** *(Mục 13)* | Zeek JSON logs (`conn.log`, `ssl.log`, `x509.log`, `ssh.log`) ánh xạ sang 57 trường của Mục 14: TLS version, cipher/group/sigalg class, `ja4`, `hassh_or_ja4ssh`, X.509 SPKI/issuer/validity, `iat_mean`, `iat_std`, `pkt_size_seq_hash_or_vector`. | - Kiểm thử Deterministic Crypto Policy Rules.<br/>- Huấn luyện Sequence Model (C2 Beaconing) & Multi-layer Fusion (L3/L4 + L5/L6 Ablation Study).<br/>- Kiểm thử AI Agent Threat Hunting & Response. | Tắt hoàn toàn payload dumping trên Zeek; kiểm duyệt tự động qua [`data/schemas/schema_allowlist.json`](../data/schemas/schema_allowlist.json); băm HMAC hoặc loại bỏ `SNI/SAN` và IP thô trước khi tạo tập train. |

### Chiến lược Chia tập Dữ liệu (Data Splitting Strategy — Mục 14)

Tuyệt đối không chỉ chia ngẫu nhiên từng dòng (*no random row split only*) để tránh rò rỉ dữ liệu (data leakage):

| Quy tắc chia (Split Rule) | Tỷ lệ / Cách thức thực hiện | Mục tiêu đánh giá (Evaluation Purpose) |
| :--- | :--- | :--- |
| **1. Temporal Split (Chia theo thời gian)** | `70% Train` (cửa sổ thời gian đầu) / `15% Validation` (cửa sổ giữa - dùng để calibrate ngưỡng) / `15% Test` (cửa sổ thời gian sau). | Ngăn chặn rò rỉ từ các phiên lặp lại; đo lường độ bền vững theo thời gian (temporal drift). |
| **2. Host / Device Holdout** | Giữ lại ít nhất 1 máy Client hoặc Server trong tập Test chưa từng xuất hiện trong tập Train. | Đánh giá khả năng tổng quát hóa trên thiết bị/máy chủ mới (không học vẹt theo địa chỉ host). |
| **3. Attack-Family Holdout (Unseen Attack)** | Loại bỏ ít nhất 1 nhóm tấn công khỏi tập Train có giám sát và chỉ đưa vào tập Test. | Đo lường khả năng phát hiện tấn công chưa từng biết (Zero-day/Unseen attack) của mô hình Isolation Forest. |
| **4. ECH / Missing-Feature Split** | Ẩn/che (mask) các trường SNI và một số trường ClientHello trong kịch bản kiểm thử ECH (RFC 9849). | Đánh giá khả năng hoạt động của mô hình khi Encrypted Client Hello làm giảm khả năng quan sát metadata. |

---

## 2. Asset Identification Matrix (Phase 2 — Mục 5)

| Mã Tài sản | Tài sản cần bảo vệ (Asset) | Thuộc tính An ninh (Security Property) | Bằng chứng Metadata Quan sát được (Không cần L7 Payload) |
| :---: | :--- | :--- | :--- |
| **A-01** | **TLS/SSH Service Identity** | Authenticity, Integrity, Availability | Server endpoint token, chứng chỉ X.509 / `cert_spki_token`, SSH host-key fingerprint, kết quả thỏa thuận giao thức. |
| **A-02** | **Cryptographic Policy** | Confidentiality & Integrity Strength | Phiên bản TLS, `tls_cipher_class`, `tls_group_class`, `tls_sigalg_class`; các lớp SSH KEX, cipher, MAC, host-key. |
| **A-03** | **Certificate & Host-Key Trust** | Authenticity & Trust Continuity | `cert_issuer_class`, `cert_chain_len`, `cert_validity_days`, `cert_spki_token`, lịch sử thay đổi chứng chỉ. |
| **A-04** | **Session State** | Freshness, Anti-Downgrade, Continuity | Trạng thái bắt tay (`handshake_status`), `session_resumed`, tần suất phiên (`session_rate_5m`), chuyển đổi `conn_state`. |
| **A-05** | **Network Topology & Service Exposure** | Availability & Attack Surface Control | Pseudonymized IP/subnet (`src_host_token`, `dst_host_token`), port, hướng kết nối, lưu lượng bytes/packets, `fanout_5m`. |
| **A-06** | **Detection & Response Evidence** | Auditability & Accountability | `session_id`, phiên bản model/rule, đồ thị bằng chứng (Evidence Graph), ánh xạ MITRE ATT&CK, kết quả action/rollback. |

---

## 3. Threat Model & MITRE ATT&CK Mapping Matrix (Phase 3 — Mục 6)

| Mã Đe dọa | Kịch bản Đe dọa / Tấn công (Threat Scenario) | Tầng & Tín hiệu Quan sát (Layers / Signals) | Cơ chế Phát hiện & Ngữ cảnh MITRE ATT&CK |
| :---: | :--- | :--- | :--- |
| **T-01** | **TLS Downgrade / Weak Negotiation / Policy Regression** | L4 connection + TLS version, cipher, group, signature metadata. | Deterministic Policy Rule + Anomaly score; ánh xạ **`T1557` (Adversary-in-the-Middle)** khi có bằng chứng hạ cấp/chặn bắt đi kèm. |
| **T-02** | **Certificate Substitution / Unexpected Trust Path** | Tóm tắt chứng chỉ X.509 trong TLS handshake + lịch sử định danh máy chủ. | Kiểm tra tính liên tục của Chain/Issuer/SPKI + Graph anomaly; xác minh chéo trước khi kết luận **`T1557`** / **`T1587.003`**. |
| **T-03** | **Malware / C2 Beaconing over Encrypted TLS** | Thống kê L3/L4 (volume, `iat_mean`, `iat_std`) + `ja4` fingerprint + tính chu kỳ phiên. | Temporal/Sequence model + Tabular ML; ánh xạ **`T1573` (Encrypted Channel)** và **`T1071.001` (Web Protocols)** dựa trên bằng chứng hành vi chuỗi. |
| **T-04** | **SSH Brute-Force / Password Spraying** | L3/L4 connection churn (`session_rate_5m`, short duration) + trạng thái phiên SSH. | Rate/Temporal & Tabular model (huấn luyện từ nhãn Brute-Force của `NF-UQ-NIDS-v2` & Lab); ánh xạ **`T1110` (Brute Force)**. |
| **T-05** | **SSH Lateral Movement / New Admin Path** | Cạnh kết nối mới (`new_edge_score`: `src_host_token` $\rightarrow$ `dst_host_token`) + `hassh_or_ja4ssh`. | Graph/Behavioral context anomaly; ánh xạ **`T1021.004` (Remote Services: SSH)**. |
| **T-06** | **Fingerprint Impersonation / Evasion** | Mâu thuẫn giữa `ja4`/`hassh` được khai báo với phân phối kích thước gói và thời gian (`iat`). | Multi-view consistency (kết hợp L3/L4 + L5/L6) thay vì chỉ tin vào danh sách đen fingerprint; ánh xạ **`T1036` (Masquerading)**. |
| **T-07** | **L3/L4 Scanning / DoS Preceding TLS/SSH Abuse** | Quét cổng, `tcp_flag_vector` (SYN flood/REJ), `fanout_5m`, đột biến packets/bytes. | Flow classifier (`NF-UQ-NIDS-v2`) + tương quan vào cùng đồ thị sự cố; ánh xạ **`T1046` (Network Service Discovery)** / **`T1498` (Network DoS)**. |
| **T-08** | **ECH / Feature Suppression** | Giảm trường thông tin quan sát được trong TLS ClientHello (`ech_observed = true`). | Mô hình thích ứng thiếu đặc trưng (Missing-feature-aware); không suy đoán SNI ẩn; chỉ dựa trên flow, fingerprint và ngữ cảnh. |

---

## 4. Security Requirements (SR) & Cryptographic Requirements (CR) Traceability (Phase 4 & Phase 6)

### 4.1. Bảng Yêu cầu An ninh Hệ thống (Security Requirements — Mục 7)

| Mã SR | Yêu cầu An ninh (Security Requirement) | Bằng chứng Nghiệm thu (Acceptance Evidence) |
| :---: | :--- | :--- |
| **SR-01** | Phát hiện phải hoạt động hoàn toàn không có raw/decrypted L7 payload. | Kiểm toán `schema_allowlist.json`: 0 trường L7; tắt xuất payload trên Zeek/Suricata. |
| **SR-02** | Metadata L3/L4 và L5/L6 phải được đồng bộ tương quan theo thời gian và `session_id` ổn định. | Script join dữ liệu có khả năng tái lập; quy định rõ cửa sổ thời gian (`window_id`). |
| **SR-03** | Bất thường chính sách mật mã TLS/SSH phải được đánh giá theo chuẩn giao thức/tổ chức. | Bộ kiểm thử Deterministic Policy Rule + bằng chứng cấu hình. |
| **SR-04** | Điểm bất thường (Anomaly score) đơn lẻ không được gán nhãn "tấn công" nếu thiếu ngưỡng hiệu chuẩn và bằng chứng ngữ cảnh. | Tập Validation hiệu chuẩn ngưỡng + mã lý do (reason codes) + độ tự tin (confidence). |
| **SR-05** | Ánh xạ MITRE ATT&CK phải dựa trên bằng chứng quan sát được (Evidence-grounded). | Chuỗi truy vết: `Finding -> Events -> Analytic -> ATT&CK Rationale`. |
| **SR-06** | Phản hồi của Agent mặc định là từ chối (Deny-by-default) và bị giới hạn bởi chính sách tài sản/môi trường. | Kiểm thử Policy Engine (`response_policy_tiers.json`) và nhật ký kiểm toán. |
| **SR-07** | Các phản hồi có tác động đến Production bắt buộc phải có phê duyệt của con người (Human/SOAR Gate). | Bản ghi phê duyệt (Approval record) + kịch bản hoàn tác (Rollback path). |
| **SR-08** | Khả năng tổng quát hóa của mô hình phải được kiểm thử qua thời gian/thiết bị, không chỉ chia ngẫu nhiên. | Báo cáo kết quả trên tập Temporal split và Host/Attack holdout. |

### 4.2. Bảng Yêu cầu Mật mã học (Cryptographic Requirements — Mục 9)

| Mã CR | Yêu cầu Mật mã học (Cryptographic Requirement) | Metadata Quan sát & Tiêu chuẩn Tham chiếu (Standards Anchor) |
| :---: | :--- | :--- |
| **CR-01** | Cấu hình TLS phải tuân thủ hồ sơ an toàn; hành vi TLS 1.3 được đối chiếu theo đặc tả hiện hành. | `tls_version`, AEAD/cipher class, supported groups, signature algorithms, downgrade indicators; **RFC 9846** / **NIST SP 800-52 Rev.2**. |
| **CR-02** | Thay đổi định danh/tin cậy của chứng chỉ phải được giải thích rõ ràng và kiểm tra tính liên tục. | Thời hạn (`cert_validity_days`), `cert_issuer_class`, `cert_chain_len`, `cert_spki_token`, thuật toán/độ dài khóa; **RFC 5280** / **RFC 9162 (CT)**. |
| **CR-03** | Encrypted Client Hello (ECH) được xử lý như một tính năng bảo vệ quyền riêng tư hợp lệ, không coi là bất thường. | Cờ `ech_observed`; không khôi phục SNI/ALPN bị ẩn; **RFC 9849**. |
| **CR-04** | Thỏa thuận phiên SSH phải tránh các thuật toán bị cấm hoặc lỗi thời theo hồ sơ bảo mật. | `ssh_kex_class`, host-key, `ssh_cipher_class`, `ssh_mac_class`, `ssh_proto_version`; **RFC 4253** + **RFC 9142**. |
| **CR-05** | Fingerprint là một đặc trưng nhận diện, không phải bằng chứng mật mã chứng minh danh tính tuyệt đối. | `ja4`, `ja3_optional`, `hassh_or_ja4ssh` luôn phải kết hợp với ngữ cảnh endpoint, chứng chỉ và thống kê phiên. |
| **CR-06** | Tuyệt đối không đưa khóa giải mã, session secrets hoặc bản rõ ứng dụng vào mô hình ML hay ngữ cảnh LLM. | Kiểm toán ranh giới dữ liệu (Data boundary audit) và unit tests trong thư mục `tests/`. |
