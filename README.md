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

** 🎯 Happy coding! 🎯**