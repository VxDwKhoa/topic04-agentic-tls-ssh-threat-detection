# NF-UQ-NIDS-v2 — Public L3/L4 Flow Dataset Loader & Multi-Model Split Documentation

Tài liệu này mô tả quy trình **Tiền xử lý (Preprocessing)**, **Ẩn danh & Loại bỏ Layer-7 (Privacy Normalizer & L7-Exclusion)**, và **Chiến lược Chia tập Dữ liệu Đa mô hình (Multi-Model Split Strategy)** cho bộ dữ liệu công khai **`NF-UQ-NIDS-v2`** trong đề tài **Topic 04 (NT219 / NT2205)**.

---

## 🔗 1. Đường dẫn Tải Bộ Dữ liệu Đã Tiền xử lý & Chia tập (Google Drive)

Do bộ dữ liệu sau khi chia có dung lượng **~93.43 MB** (`652,109` bản ghi ở định dạng nén `Parquet`), để giữ cho GitHub Repository gọn nhẹ, toàn bộ thư mục **`NF-UQ-NIDS-v2/`** được lưu trữ trên Google Drive của nhóm:

> 📁 **Google Drive Dataset Link (`NF-UQ-NIDS-v2/` — 93.43 MB):**  
> 👉 **[DÁN_LINK_GOOGLE_DRIVE_CỦA_BẠN_VÀO_ĐÂY](https://drive.google.com/)**

*(Trong repository này cũng đã đính kèm sẵn một bản mẫu **`sample_data/nfuq_sample_1500.parquet`** gồm 1,500 dòng và biên bản kiểm định **`sample_data/nfuq_manifest_and_audit.json`** để giảng viên hoặc CI/CD runner có thể chạy kiểm thử ngay lập tức mà không cần tải full dataset).*

---

## 🛡️ 2. Quy tắc Tiền xử lý & Tuân thủ "No Raw L7 Payload"

Quá trình tiền xử lý được thực thi bởi script [`preprocess_and_split_nfuq.py`](preprocess_and_split_nfuq.py) và được kiểm định tự động bởi [`../schemas/privacy_validator.py`](../schemas/privacy_validator.py), bám sát các mục trong hướng dẫn Topic 04:

1. **Loại bỏ hoàn toàn các trường mang ngữ nghĩa Layer-7 (`SR-01`, `CR-06`, Mục 13.4 Bước 7):**
   - Đã cắt bỏ (`drop`) 5 cột L7/application semantics khỏi bản gốc `NF-UQ-NIDS-v2`:
     `L7_PROTO`, `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`, `FTP_COMMAND_RET_CODE`.
   - Kết quả chạy `privacy_validator.py`: **`passed_no_l7_audit: true` (0 violations trên 652,109 dòng)**.

2. **Ẩn danh hóa định danh mạng (`Pseudonymization` — Mục 4, Mục 5, Mục 8, Mục 13.4 Bước 2):**
   - Cột `IPV4_SRC_ADDR` và `IPV4_DST_ADDR` được băm một chiều bằng thuật toán **`HMAC-SHA256`** (16 ký tự hex prefix) thành **`src_host_token`** và **`dst_host_token`** (ví dụ: `host_8f1c9a04e2b371d0`), sau đó xóa bỏ hoàn toàn IP thô.
   - Đảm bảo tính nhất quán (cùng 1 máy nguồn luôn có cùng `src_host_token`) để gom chuỗi thời gian cho mô hình **LSTM** và chia **Host Holdout**, nhưng không để lộ IP thật và ngăn mô hình học vẹt địa chỉ IP.

3. **Chuẩn hóa Cổng & Đặc trưng L3/L4 (Mục 14 — 40 Numerical Features):**
   - Chuyển `L4_SRC_PORT` thành **`src_port_class`** theo chuẩn RFC 6335 (`0`: well-known `0–1023`, `1`: registered `1024–49151`, `2`: ephemeral `49152–65535`).
   - Giữ **`dst_port`** và tạo thêm cờ ngữ cảnh cổng dịch vụ mã hóa: **`is_tls_port`** (`443, 8443, 993, 995`), **`is_ssh_port`** (`22, 2222`), cùng tỷ lệ bất đối xứng luồng **`bytes_ratio`** và **`pkts_ratio`**.

---

## 📊 3. Chiến lược Chia Tập Dữ liệu Đa Mô hình (Section 14 Split Rules)

Từ 30 triệu dòng đầu tiên của file gốc `NF-UQ-NIDS-v2.csv` (13.7 GB), hệ thống trích mẫu phân tầng (**Stratified Subset — Mục 12**) thu được **`652,109` dòng** trải đều qua 30 cửa sổ thời gian (`window_id`) và 21 lớp lưu lượng, ưu tiên giữ lại các luồng qua cổng TLS/SSH.

Bộ dữ liệu được chia theo **3 quy tắc bắt buộc của Mục 14**:

| Tập dữ liệu (Split) | Số lượng bản ghi | Tỷ lệ | Quy tắc áp dụng (Section 14 Rules) |
| :--- | :---: | :---: | :--- |
| **`train/`** | **425,771** | ~65.3% | **Early Temporal Windows (70% đầu của mỗi lớp đã biết):** Chỉ gồm các host thông thường và các lớp tấn công đã biết. Dùng để fit `StandardScaler`, huấn luyện **Random Forest / XGBoost**, **LSTM**, và lọc `Benign` để fit **Isolation Forest**. |
| **`val/`** | **91,230** | ~14.0% | **Middle Temporal Windows (15% giữa):** Dùng để tinh chỉnh siêu tham số và **Hiệu chuẩn ngưỡng báo động (Threshold Calibration — `SR-04`, Mục 10)**. |
| **`test/`** | **135,108** | ~20.7% | **Later Temporal Windows (15% cuối) + Host Holdout (`183` hosts chưa từng xuất hiện ở Train) + Attack-Family Holdout (`635` dòng Unseen Attacks: `Worms`, `Shellcode`, `Ransomware`):** Đánh giá khả năng tổng quát hóa theo thời gian, thiết bị mới và tấn công Zero-day. |
| **Tổng cộng** | **652,109** | **100%** | Đầy đủ 40 đặc trưng L3/L4 + 11 cột định danh/nhãn/ánh xạ MITRE ATT&CK. |

---

## 🗂️ 4. Cấu trúc Thư mục `NF-UQ-NIDS-v2/` (Phục vụ cả Random Forest & LSTM)

Một cấu trúc chia duy nhất phục vụ đồng thời **Mô hình Bảng (Random Forest / XGBoost / Isolation Forest)** và **Mô hình Chuỗi Thời gian (LSTM / 1D-CNN / GRU)**:

```text
NF-UQ-NIDS-v2/                          # Tổng dung lượng: ~93.43 MB
├── metadata_manifest.json              # Thống kê số dòng, scaler (mean/std), biên bản kiểm định No-L7
├── train/                              # 425,771 dòng
│   ├── flows_tabular.parquet           # [Dùng cho Random Forest / XGBoost / Isolation Forest]
│   ├── sequences_temporal/             # [Dùng cho LSTM / 1D-CNN] Chia thành 10 cửa sổ thời gian
│   │   ├── window_00/sequences.parquet # Đã sắp xếp theo (src_host_token, flow_order_id)
│   │   ├── ...
│   │   └── window_09/sequences.parquet
│   └── by_label/                       # Chia thành 19 thư mục con theo từng nhãn tấn công
│       ├── Benign/flows.parquet
│       ├── Brute_Force/flows.parquet
│       ├── Bot/flows.parquet
│       ├── mitm/flows.parquet
│       └── ...
├── val/                                # 91,230 dòng (cấu trúc y hệt train/)
│   ├── flows_tabular.parquet
│   ├── sequences_temporal/window_00..09/
│   └── by_label/<Attack_Label>/
└── test/                               # 135,108 dòng
    ├── flows_tabular.parquet
    ├── unseen_attack_holdout.parquet   # [Dùng riêng để test Anomaly Detection bắt Zero-day]
    ├── sequences_temporal/window_00..09/
    └── by_label/<Attack_Label>/
```

---

## 💻 5. Hướng dẫn Nạp Dữ liệu khi Huấn luyện (Quick-Start Code)

### A. Nạp dữ liệu dạng Bảng cho Random Forest / XGBoost / Isolation Forest (Tuần 5 & 6)
```python
import pandas as pd
from data.public_loaders.preprocess_and_split_nfuq import L3_L4_FEATURE_COLUMNS

train_df = pd.read_parquet(r"D:\Crypto\DoAn\NF-UQ-NIDS-v2\train\flows_tabular.parquet")
val_df   = pd.read_parquet(r"D:\Crypto\DoAn\NF-UQ-NIDS-v2\val\flows_tabular.parquet")
test_df  = pd.read_parquet(r"D:\Crypto\DoAn\NF-UQ-NIDS-v2\test\flows_tabular.parquet")

X_train, y_train = train_df[L3_L4_FEATURE_COLUMNS], train_df["label"]
X_val,   y_val   = val_df[L3_L4_FEATURE_COLUMNS],   val_df["label"]
X_test,  y_test  = test_df[L3_L4_FEATURE_COLUMNS],  test_df["label"]
```

### B. Nạp dữ liệu dạng Chuỗi 3D `(Batch, Time_Steps, Features)` cho LSTM / 1D-CNN (Tuần 7)
```python
import pandas as pd
from data.public_loaders.preprocess_and_split_nfuq import build_lstm_sequences

win_df = pd.read_parquet(r"D:\Crypto\DoAn\NF-UQ-NIDS-v2\train\sequences_temporal\window_00\sequences.parquet")
X_seq, y_seq = build_lstm_sequences(win_df, seq_len=8, stride=4)
# X_seq.shape -> (N_sequences, 8, 40)
```
