# CIC-IDS2017 Password Brute-Force Detection

Pipeline phân loại có giám sát với `Logistic Regression` và `XGBoost`. Nhãn `is_attack=1` gồm
`FTP-Patator`, `SSH-Patator` và `Web Attack - Brute Force`; `BENIGN` là lớp 0.
Cột `attack_type` chỉ dùng để phân tích lỗi và bị loại khỏi feature khi train.

## Chạy pipeline

Từ thư mục gốc dự án:

```powershell
pip install -r requirements.txt
python run_all.py
```

Các bước model: s05 là Logistic Regression, s06 là XGBoost.
Mỗi bước train cả tập undersampling và full train, chọn threshold bằng validation,
rồi đánh giá trên test. Có thể chạy riêng:

```powershell
python run_all.py --only s05,s06
python run_all.py --from s07
```

## Xử lý mất cân bằng

- `s04`: tạo trọng số lớp cho toàn bộ train và tạo thêm tập train random undersampling
  với đúng 3 mẫu bình thường cho mỗi mẫu tấn công.
- `s05`: train Logistic Regression trên tập undersampling và full train, chọn threshold
  trên validation và đánh giá test.
- `s06`: train XGBoost trên tập undersampling và full train, dùng early stopping và chọn
  threshold bằng validation, rồi đánh giá test.
- `s07`: so sánh phân phối đặc trưng giữa train và test.
- Hai bước model báo cáo PR-AUC, precision, recall, F1 lớp 1, TP/FP/FN/TN và confusion matrix.
- `s08_test_external_data.py`: đánh giá các model đã train trên CSV ngoài đặt trong `data/test/`.
  Chạy riêng bằng `python -m steps.s08_test_external_data` sau khi train xong s05 và s06.

## Đầu ra

- Dữ liệu chia tập: `data/processed/`
- Bảng CSV: `outputs/tables/`
- Metrics JSON: `outputs/metrics/`
- Biểu đồ: `outputs/figures/`
- Model joblib: `outputs/models/`

Các bước chưa được chạy; hãy chạy `run_all.py` sau khi đặt CSV trong `data/raw/`.
