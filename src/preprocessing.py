import os
import pandas as pd
import numpy as np
import cv2
from keras.utils import to_categorical # type: ignore
from src.constants import AGE_GROUPS, GENDER_LABELS

# Gán nhãn cho tuổi
def age_to_label(age_str):
    if age_str not in AGE_GROUPS:
        return None
    return AGE_GROUPS.index(age_str)

# Gán nhãn cho giới tính
def gender_to_label(gender_str):
    if gender_str in GENDER_LABELS:
        return GENDER_LABELS.index(gender_str)
    else:
        return None

# Tiền xử lý dữ liệu
def preprocess(data_dir='.', fold_files=None, image_size=227, max_samples=None, output_dir='../outputs'):
    if fold_files is None:
        fold_files = ['fold_0_data.txt']

    dfs = []
    for file in fold_files:
        path = os.path.join(data_dir, file)
        print(f"[!] Đang đọc file: {path}")
        df = pd.read_csv(path, sep='\t')
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[['user_id', 'original_image', 'face_id', 'age', 'gender']]
    df.dropna(inplace=True)

    X, y_age, y_gender = [], [], []

    for _, row in df.iterrows():
        # Xây dựng đường dẫn ảnh theo mẫu
        img_name = f"landmark_aligned_face.{row['face_id']}.{row['original_image'].split('.')[0]}.jpg"
        img_path = os.path.join(data_dir, 'raw', 'aligned', str(row['user_id']), img_name)

        if not os.path.isfile(img_path):
            print(f"[!] Không tìm thấy ảnh: {img_path}")
            continue

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None or img.shape[0] < image_size or img.shape[1] < image_size:
            continue

        # Resize về (256, 256), crop giữa (227, 227), chuẩn hóa
        img = cv2.resize(img, (256, 256))
        offset = (256 - image_size) // 2
        img = img[offset:offset+image_size, offset:offset+image_size]
        img = img.astype('float32') / 255.0

        # Xử lý age và gender thành label
        age_label = age_to_label(row['age'])
        gender_label = gender_to_label(row['gender'])

        if age_label is None or gender_label is None:
            continue

        X.append(img)
        y_age.append(age_label)
        y_gender.append(gender_label)

        # Dừng lại nếu đã đủ số lượng mẫu
        if max_samples and len(X) >= max_samples:
            print(f"Đã đạt ngưỡng {max_samples} mẫu!")
            break

        if len(X) % 500 == 0:
            print(f"Đã xử lý {len(X)} ảnh hợp lệ...")

    # Chuyển sang numpy và one-hot encoding
    X = np.array(X)
    y_age = to_categorical(y_age, num_classes=8)
    y_gender = to_categorical(y_gender, num_classes=2)

    os.makedirs(output_dir, exist_ok=True)

    # Lưu các file .npy vào thư mục outputs
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y_gender.npy'), y_gender)
    np.save(os.path.join(output_dir, 'y_age.npy'), y_age)
    print("Đã lưu dữ liệu thành công vào thư mục outputs!")

    return X, y_age, y_gender