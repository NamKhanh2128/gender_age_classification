import numpy as np
import os

# Đường dẫn đến thư mục chứa các file .npy đã được tạo
OUTPUT_DIR = 'outputs'

# Đọc các file .npy
X = np.load(os.path.join(OUTPUT_DIR, 'X.npy'))
y_age = np.load(os.path.join(OUTPUT_DIR, 'y_age.npy'))
y_gender = np.load(os.path.join(OUTPUT_DIR, 'y_gender.npy'))

# Nén lại thành 1 file duy nhất
np.savez_compressed(os.path.join(OUTPUT_DIR, 'dataset_compressed.npz'),
                    X=X, y_age=y_age, y_gender=y_gender)

print("✅ Nén thành công! File lưu tại:", os.path.join(OUTPUT_DIR, 'dataset_compressed.npz'))
