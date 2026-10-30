# CIC-IDS2017 Password Brute-Force Detection

Pipeline phân loại có giám sát với `SGDClassifier`, `Logistic Regression` và
`XGBoost`. Nhãn `is_attack=1` gồm
`FTP-Patator`, `SSH-Patator` và `Web Attack - Brute Force`; `BENIGN` là lớp 0.
Cột `attack_type` chỉ dùng để phân tích lỗi và bị loại khỏi feature khi train.

## Chạy pipeline

Từ thư mục gốc dự án:

```powershell
pip install -r requirements.txt
python run_all.py
```

Các bước model: s05 là SGD, s06 là Logistic Regression, s07 là XGBoost.
Mỗi bước chọn threshold bằng validation rồi đánh giá trên test. Có thể chạy riêng:

```powershell
python run_all.py --only s05,s06,s07
python run_all.py --from s06
```

## Xử lý mất cân bằng

- `s04`: tạo trọng số lớp cho toàn bộ train và tạo thêm tập train random undersampling
  với đúng 3 mẫu bình thường cho mỗi mẫu tấn công.
- `s05`: train SGD, chọn threshold trên validation và đánh giá test.
- `s06`: train Logistic Regression, chọn threshold trên validation và đánh giá test.
- `s07`: train XGBoost trên train đầy đủ, early stopping và chọn threshold bằng validation,
  rồi đánh giá test.
- Ba bước model báo cáo PR-AUC, precision, recall, F1 lớp 1, TP/FP/FN/TN và confusion matrix.

## Đầu ra

- Dữ liệu chia tập: `data/processed/`
- Bảng CSV: `outputs/tables/`
- Metrics JSON: `outputs/metrics/`
- Biểu đồ: `outputs/figures/`
- Model joblib: `outputs/models/`

Các bước chưa được chạy; hãy chạy `run_all.py` sau khi đặt CSV trong `data/raw/`.
