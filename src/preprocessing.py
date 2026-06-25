import os
import pandas as pd
import numpy as np
import cv2
import math
from keras.utils import to_categorical # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
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

# Bộ tạo dữ liệu theo batch (Keras Sequence) chống tràn RAM
class AdienceDatasetSequence(Sequence):
    def __init__(self, data_dir='.', fold_files=None, batch_size=32, image_size=227, shuffle=True, max_samples=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        if fold_files is None:
            fold_files = ['fold_0_data.txt']

        dfs = []
        for file in fold_files:
            path = os.path.join(data_dir, file)
            if not os.path.exists(path):
                print(f"[!] Warning: File {path} không tồn tại.")
                continue
            print(f"[!] Đang đọc file: {path}")
            df = pd.read_csv(path, sep='\t')
            dfs.append(df)

        if not dfs:
            raise ValueError(f"Không tìm thấy file fold nào tại {data_dir}!")

        df = pd.concat(dfs, ignore_index=True)
        df = df[['user_id', 'original_image', 'face_id', 'age', 'gender']]
        df.dropna(inplace=True)

        self.samples = []
        for _, row in df.iterrows():
            # Xây dựng đường dẫn ảnh theo mẫu
            img_name = f"landmark_aligned_face.{row['face_id']}.{row['original_image'].split('.')[0]}.jpg"
            img_path = os.path.join(data_dir, 'raw', 'aligned', str(row['user_id']), img_name)

            if not os.path.isfile(img_path):
                continue

            # Xử lý age và gender thành label
            age_label = age_to_label(row['age'])
            gender_label = gender_to_label(row['gender'])

            if age_label is None or gender_label is None:
                continue

            self.samples.append({
                'img_path': img_path,
                'age_label': age_label,
                'gender_label': gender_label
            })

            if max_samples and len(self.samples) >= max_samples:
                print(f"Đã đạt giới hạn {max_samples} mẫu cho Generator!")
                break

        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

        print(f"[INFO] Khởi tạo Generator thành công với {len(self.samples)} mẫu.")

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = []
        batch_y_age = []
        batch_y_gender = []

        for i in batch_indices:
            sample = self.samples[i]
            img_path = sample['img_path']

            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                # Nếu ảnh lỗi, tạo ảnh đen
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                # Resize về (256, 256), crop giữa (image_size, image_size)
                img = cv2.resize(img, (256, 256))
                offset = (256 - self.image_size) // 2
                img = img[offset:offset+self.image_size, offset:offset+self.image_size]

            img = img.astype('float32') / 255.0

            batch_x.append(img)
            batch_y_age.append(sample['age_label'])
            batch_y_gender.append(sample['gender_label'])

        batch_x = np.array(batch_x)
        # Chuyển đổi labels thành category one-hot
        batch_y_age_cat = to_categorical(batch_y_age, num_classes=8)
        batch_y_gender_cat = to_categorical(batch_y_gender, num_classes=2)

        return batch_x, {
            'age_output': batch_y_age_cat,
            'gender_output': batch_y_gender_cat
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)