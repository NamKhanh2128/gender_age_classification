# Age & Gender Classification - Multi-Task Learning

Dự án phân loại tuổi và giới tính từ ảnh khuôn mặt sử dụng Deep Learning.

## Mục tiêu

- **Phân loại giới tính**: Nam/Nữ (2 classes)
- **Phân loại tuổi**: 8 nhóm tuổi từ (0,2) đến (60,100)
- **Multi-task learning**: Huấn luyện đồng thời cả 2 tasks với shared feature extraction

## Kiến trúc

### Model Architecture
- **Shared CNN layers**: 3 Convolutional layers + 2 Fully Connected layers
- **Gender branch**: Dense(256) → Dense(2) với softmax
- **Age branch**: Dense(256) → Dense(8) với softmax

### Age Groups
```
(0, 2), (4, 6), (8, 13), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)
```

## 📁 Cấu trúc dự án

```
IntroAI.20242.Project/
├── src/                          # Source code
│   ├── model.py                  # Multi-task CNN
│   ├── preprocessing.py          # Load và tiền xử lý dữ liệuliệu
│   └── constants.py              # Các hằng số
├── data/                         # Dataset
│   ├── raw/                      # Bộ dữ liệu Adience
├── outputs/                      # Training outputs
│   ├── dataset_compressed.npz    # File nén dữ liệu đã được tiền xử lý
│   └── trained_model_ver3.h5     # File model đã được huấn luyện
├── main.py                       # File chính để chạy chương trình
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Dependencies
├── config.json                   # Configuration file
└── README.md                     # This file
```

## Cài đặt

### 1. Clone repository
```
git clone https://github.com/20225683-vietddh/IntroAI.20242.git
cd IntroAI.20242.Project
```

### 2. Cài đặt dependencies
```
pip install -r requirements.txt
```

## Dataset

Dự án sử dụng dataset với cấu trúc:
- **Images**: Ảnh khuôn mặt đã được aligned (227x227x3 pixels)
- **Labels**: File `fold_0_data.txt` chứa thông tin user_id, image_name, age, gender
- **Format**: JPG images trong thư mục `data/raw/aligned/`

## Sử dụng

### 1. Khởi chạy Ứng dụng Web Streamlit (Khuyến nghị - Giao diện Mới)
Ứng dụng web Streamlit cung cấp giao diện hiện đại với biểu đồ Plotly tương tác, phân tích dữ liệu EDA, webcam capture, lịch sử dự đoán và nhiều tính năng khác.

```bash
# Kích hoạt môi trường ảo (Windows)
.venv\Scripts\activate

# Chạy ứng dụng Streamlit
streamlit run app.py
```
Sau khi chạy, ứng dụng sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

### 2. Khởi chạy Giao diện Tkinter cũ (Legacy GUI)
Nếu bạn vẫn muốn dùng giao diện Tkinter truyền thống:
```bash
.venv\Scripts\python.exe main.py
```

## Các Cải tiến nổi bật của Dự án

1. **Ứng dụng Web Premium (`app.py`)**:
   - **Glassmorphism Theme**: Thiết kế giao diện tối hiện đại, tinh tế.
   - **Nhiều nguồn ảnh**: Hỗ trợ Upload tệp, Chụp trực tiếp từ Webcam, và thử nghiệm nhanh bằng Ảnh mẫu (Sample Images).
   - **Biểu đồ phân phối Plotly**: Xem trực quan phân phối xác suất dự đoán giới tính và nhóm tuổi.
   - **Lịch sử hoạt động (History)**: Ghi lại các lượt quét trong phiên làm việc, hỗ trợ xuất báo cáo định dạng CSV.
   - **Phân tích dữ liệu EDA**: Tích hợp trang phân tích thống kê bộ dữ liệu Adience với biểu đồ Pie & Bar sinh động.

2. **Nâng cấp Mô hình Deep Learning (`src/model.py`)**:
   - **Transfer Learning**: Hỗ trợ sử dụng xương sống **MobileNetV2** pre-trained trên ImageNet làm backbone trích xuất đặc trưng giúp tăng vọt độ chính xác phân loại.
   - **Tích hợp Data Augmentation**: Thêm các lớp tăng cường ảnh (`RandomFlip`, `RandomRotation`, `RandomTranslation`) trực tiếp trong Keras model giúp chống overfitting hiệu quả mà không tốn dung lượng đĩa cứng.
   - **Callbacks thông minh**: Tự động cấu hình `EarlyStopping` và `ReduceLROnPlateau` giúp kiểm soát và tối ưu quá trình huấn luyện tự động.

3. **Tối ưu hóa Pipeline Dữ liệu (`src/preprocessing.py`)**:
   - **AdienceDatasetSequence**: Bộ sinh dữ liệu theo lô (Data Generator) kế thừa từ `keras.utils.Sequence` giúp tải ảnh động từ đĩa khi huấn luyện, ngăn chặn lỗi tràn bộ nhớ RAM (OOM) đối với tập dữ liệu lớn.

** 🎯 Happy coding! 🎯**