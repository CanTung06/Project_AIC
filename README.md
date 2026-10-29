# CIC-IDS2017 Password Brute-Force Detection

Pipeline phân loại có giám sát với `SGDClassifier`. Nhãn `is_attack=1` gồm
`FTP-Patator`, `SSH-Patator` và `Web Attack - Brute Force`; `BENIGN` là lớp 0.
Cột `attack_type` chỉ dùng để phân tích lỗi và bị loại khỏi feature khi train.

## Chạy pipeline

Từ thư mục gốc dự án:

```powershell
pip install -r requirements.txt
python run_all.py
```

Hoặc chạy từng bước theo thứ tự `s01` đến `s11`. Có thể chạy lại một phần, ví dụ:

```powershell
python run_all.py --only s05,s06
python run_all.py --from s07
```

## Xử lý mất cân bằng

- `s04`: tạo trọng số lớp cho toàn bộ train và tạo thêm tập train random undersampling
  với mục tiêu 10 mẫu bình thường cho mỗi mẫu tấn công.
- `s05`: huấn luyện hai biến thể `SGDClassifier`, một biến thể dùng `class_weight`,
  một biến thể dùng tập undersampling. Validation và test giữ nguyên phân phối tự nhiên.
- `s06` và `s07`: báo cáo PR-AUC, precision, recall, F1 lớp 1, TP/FP/FN/TN và
  confusion matrix. Ngưỡng được chọn trên validation; test chỉ đánh giá cuối.

## Đầu ra

- Dữ liệu chia tập: `data/processed/`
- Bảng CSV: `outputs/tables/`
- Metrics JSON: `outputs/metrics/`
- Biểu đồ: `outputs/figures/`
- Model joblib: `outputs/models/`

Các bước chưa được chạy; hãy chạy `run_all.py` sau khi đặt CSV trong `data/raw/`.
