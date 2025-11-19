import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model đã train sẵn (multi-task: 3 outputs)
model = load_model(r"C:\Users\win\Downloads\trained_model_caitien_3.h5")

# Map ngược index nhóm tuổi sang khoảng tuổi
age_group_map = {
    0: '0-24',
    1: '25-49',
    2: '50-74',
    3: '75-99',
    4: '100+'
}

# Map index giới tính
gender_map = {0: 'Male', 1: 'Female'}

def predict_single_image(model, img_path, input_size=(180, 180)):
    # Đọc và xử lý ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không đọc được ảnh: {img_path}")
        return
    img = cv2.resize(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    X = np.expand_dims(img, axis=0)  # shape (1,180,180,3)
    
    # Dự đoán
    preds = model.predict(X)
    # age_group: (1,5), gender: (1,2), age_estimation: (1,1)
    age_group_pred = np.argmax(preds[0][0])
    gender_probs = preds[1][0]      # <--- Lấy xác suất ở đây
    gender_pred = np.argmax(gender_probs)
    age_estimation_pred = preds[2][0][0]   # float

    print(f"Age group: {age_group_map[age_group_pred]} (class {age_group_pred})")
    print(f"Gender: {gender_map[gender_pred]} (class {gender_pred})")
    print(f"Estimated age: {age_estimation_pred:.1f} years")

    # Hiển thị phần trăm xác suất giới tính
    print(f"Gender probabilities:")
    for i, prob in enumerate(gender_probs):
        print(f"  {gender_map[i]}: {prob * 100:.1f}%")

    return {
        "age_group_class": int(age_group_pred),
        "age_group_range": age_group_map[age_group_pred],
        "gender_class": int(gender_pred),
        "gender": gender_map[gender_pred],
        "gender_probs": gender_probs.tolist(),
        "age_estimation": float(age_estimation_pred)
    }

#Ví dụ sử dụng:
image_path = r"C:\Users\win\Downloads\testanh11.jpg"
predict_single_image(model, image_path)
