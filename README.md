# Phát hiện bất thường đăng nhập

Pipeline học có giám sát dùng Logistic Regression, XGBoost và Random Forest trên log xác thực đặt trong `data/raw/`.
Dataset hiện tại dùng cột `Anomaly`: `0` là bình thường, `1` là bất thường.

Tiền xử lý tạo đặc trưng giờ/ngày, khoảng cách đến lần đăng nhập trước, số lần đăng nhập trước
và tần suất xuất hiện trước đó của IP, thiết bị, vị trí theo từng user. Các thuộc tính phân loại
được one-hot encode. User ID không được dùng làm định danh trực tiếp trong model.

## Chạy pipeline

Từ thư mục gốc dự án:

```powershell
pip install -r requirements.txt
python run_all.py
```

Các bước:
- `s01`: EDA và phân bố nhãn.
- `s02`: tiền xử lý, feature engineering và one-hot encoding.
- `s03`: chia train/validation/test ngẫu nhiên có phân tầng theo nhãn (70%/15%/15%) để mỗi tập phản ánh hỗn hợp các nguồn dữ liệu.
- `s04`: tạo tập train undersampling với tối đa 3 mẫu bình thường cho mỗi mẫu bất thường.
- `s05`: train và đánh giá Logistic Regression trên full train và undersampling.
- `s06`: train và đánh giá XGBoost trên full train và undersampling; threshold được chọn trên validation.
- `s07`: train và đánh giá Random Forest trên full train và undersampling; threshold được chọn trên validation.

## Dự đoán file ngoài

Sau khi chạy s02, s05, s06, và s07 chạy:

```powershell
python predict.py
```

Nhập model cần dùng và đường dẫn CSV log xác thực. File ngoài phải có cùng các trường đầu vào
như dataset, nhưng có thể không có cột nhãn. Kết quả được lưu ở `outputs/predictions/`.
Đặc trưng lịch sử được tính theo thứ tự Timestamp và chỉ dựa trên các sự kiện đứng trước mỗi dòng.

## Thư mục đầu ra

- Dữ liệu train, validation, test: `data/processed/`
- Bảng: `outputs/tables/`
- Metrics JSON: `outputs/metrics/`
- Biểu đồ: `outputs/figures/`
- Model joblib: `outputs/models/`

