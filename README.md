# Phát hiện bất thường đăng nhập: Logistic Regression, Random Forest và SGDClassifier

Pipeline huấn luyện ba mô hình có giám sát để dự đoán nhãn `Is Attack IP`: Logistic Regression, Random Forest và SGDClassifier. Các mô hình dùng cùng feature, cùng các tập train/validation/test; mỗi mô hình chọn ngưỡng tối đa F1 lớp 1 riêng trên validation.

## Các bước

- `steps/s01_eda.py`: thống kê dữ liệu, nhãn Is Attack IP, thời gian và giá trị thiếu.
- `steps/s02_preprocess.py`: tạo feature lịch sử theo user, thêm số lần IP/ASN đã xuất hiện trước đó trên toàn dữ liệu, rồi chia train/validation/test.
- `steps/s03_train_logistic_regression.py`: huấn luyện Logistic Regression với chuẩn hóa feature và `class_weight="balanced"`.
- `steps/s04_evaluate_logistic_regression.py`: chọn ngưỡng tối đa F1 lớp 1 trên validation, đánh giá test.
- `steps/s05_train_random_forest.py`: huấn luyện Random Forest có giám sát dự đoán nhãn `Is Attack IP`, dùng `class_weight="balanced_subsample"`.
- `steps/s06_evaluate_random_forest.py`: chọn ngưỡng tối đa F1 lớp 1 trên validation, đánh giá test.
- `steps/s07_train_sgd_classifier.py`: huấn luyện SGDClassifier trên toàn bộ train bằng `partial_fit` theo chunk.
- `steps/s08_evaluate_sgd_classifier.py`: chọn ngưỡng tối đa F1 lớp 1 trên validation, đánh giá test.
- `steps/s09_compare_models.py`: so sánh cả ba mô hình trên cùng test set.
- Logistic Regression: `outputs/models/03_logistic_regression_is_attack_ip.joblib`.
- Random Forest: `outputs/models/05_random_forest_is_attack_ip.joblib`.
- SGDClassifier: `outputs/models/07_sgd_classifier_is_attack_ip.joblib`.
- Metrics: JSON trong `outputs/metrics/`; biểu đồ PR trong `outputs/figures/`.
- So sánh: `outputs/metrics/09_model_comparison.json` và `outputs/tables/09_model_comparison.csv`.

## Cài đặt và chạy

```powershell
pip install -r requirements.txt
python run_all.py
```

`s02` đọc toàn bộ raw CSV theo chunk 500.000 dòng và tạo lại các file trong `data/processed`. Đây là bước tốn thời gian nhất. Nếu processed files đã có và muốn chạy các mô hình cùng so sánh:

```powershell
python run_all.py --from-step s03
```

Để huấn luyện hai mô hình, đánh giá chúng và so sánh kết quả:

```powershell
python run_all.py --from-step s03
```

Nếu thay đổi feature ở s02, hãy chạy `python run_all.py --from-step s02` để dựng lại dữ liệu rồi chạy các bước tiếp theo.

## Nhãn và giới hạn

`is_anomaly` được tạo từ `Is Attack IP`: 0 là IP không bị gắn cờ, 1 là IP bị gắn cờ. Cả hai mô hình học nhãn này; nhãn không được đưa vào feature. `Is Account Takeover` không được dùng. User ID và source index cũng không nằm trong feature.

Các chỉ số cho biết mô hình dự đoán `Is Attack IP` tốt đến đâu; chúng không chứng minh nhãn này đồng nghĩa với đăng nhập bất thường hoặc ATO. Mỗi mô hình chọn ngưỡng trên validation; không điều chỉnh ngưỡng dựa trên test.
