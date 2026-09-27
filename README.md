# Plant AI - Nhận diện cây khỏe mạnh, héo hoặc bệnh

Đề tài môn Trí tuệ nhân tạo: xây dựng chương trình nhận diện tình trạng cây từ ảnh bằng mạng CNN.

## 1. Mô hình
Sử dụng MobileNetV2 (một kiến trúc CNN học sâu) theo phương pháp Transfer Learning.
Ba lớp:
- healthy: cây khỏe mạnh
- withered: cây héo
- diseased: cây bị bệnh

> Bạn có thể đổi số lớp bằng cách thay đổi các thư mục trong `dataset/`.

## 2. Yêu cầu
- Windows 10/11
- Python 3.10 hoặc 3.11
- RAM 8 GB vẫn có thể chạy; huấn luyện trên CPU sẽ chậm hơn GPU.

## 3. Cài đặt
Mở CMD/PowerShell tại thư mục project:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Chuẩn bị dữ liệu
Đặt ảnh vào:

```text
dataset/
  healthy/
  withered/
  diseased/
```

Mỗi thư mục nên có ít nhất 100 ảnh, tốt hơn là 300-1000 ảnh/lớp.
Ảnh JPG/JPEG/PNG.

Ví dụ:
```text
dataset/healthy/healthy001.jpg
dataset/withered/withered001.jpg
dataset/diseased/diseased001.jpg
```

## 5. Huấn luyện
```bash
python train.py
```

Sau khi xong, mô hình được lưu ở:
`models/plant_model.keras`

Biểu đồ accuracy/loss được lưu trong `results/`.

## 6. Chạy chương trình
```bash
python app.py
```

Chọn ảnh cây -> chương trình hiển thị:
- lớp dự đoán
- độ tin cậy
- ảnh đã chọn

## 7. Lưu ý
Nếu máy yếu, trong `train.py` giảm `EPOCHS = 5` và `BATCH_SIZE = 8`.
Kết quả phụ thuộc mạnh vào chất lượng và độ cân bằng của dataset.
