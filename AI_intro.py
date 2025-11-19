import numpy as np
import cv2
import os
import pandas as pd
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

#download dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("jangedoo/utkface-new")

print("Path to dataset files:", path)





# Hàm chuyển đổi nhóm tuổi
def age_to_group(age):
    if age <= 24: return 0
    elif age <= 49: return 1
    elif age <= 74: return 2
    elif age <= 99: return 3
 
    else: return 4



# Class để đọc ảnh, resize, và trả về batch
class UTKFaceSequence(Sequence):
    def __init__(self, df, img_dir, batch_size=32, input_size=(180,180), num_age_classes=5, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_age_classes = num_age_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_df = self.df.iloc[indexes]
        X = []
        y_age = []
        y_gender = []
        y_age_estimation = []  # Lưu trữ age_estimation labels
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.img_dir, row['filename'])
            try:
                # Đọc ảnh
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Unable to read the image at {img_path}")
                    continue
                # Resize ảnh về kích thước chuẩn
                img = cv2.resize(img, self.input_size)
                # Chuyển ảnh từ BGR (OpenCV) sang RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Chuẩn hóa ảnh
                img = img / 255.0
                X.append(img)

                # One-hot encoding cho age group
                y_age.append(int(row['age_group']))
                y_gender.append(int(row['gender']))
                y_age_estimation.append(int(row['age']))  # Dự đoán tuổi thực tế

            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")
                continue

        X = np.array(X)
        y_age = np.eye(self.num_age_classes)[y_age]  # One-hot encode cho age group
        y_gender = np.eye(2)[y_gender]  # One-hot encode cho giới tính
        y_age_estimation = np.array(y_age_estimation)  # Lưu age_estimation

        return X, {'age_output': y_age, 'gender_output': y_gender, 'age_estimation_output': y_age_estimation}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Đọc dataset và tạo DataFrame
folder = "/kaggle/input/utkface-new/UTKFace"  # Đường dẫn đến folder ảnh UTKFace

filelist = [f for f in os.listdir(folder) if f.endswith('.jpg')]

data = []
for filename in filelist:
    try:
        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])
        age_group = age_to_group(age)
        data.append([filename, age, age_group, gender])
    except:
        continue

df = pd.DataFrame(data, columns=["filename", "age", "age_group", "gender"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

# Chia train/val/test (80/10/10)
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["age_group"])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp["age_group"])

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")




#coi phân bố dữ liệu trên age_group
print(df['age_group'].value_counts())



#build model
from tensorflow.keras.layers import Input, SeparableConv2D, BatchNormalization, SpatialDropout2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def build_utkface_cnn(input_shape=(180,180,3), age_groups=5, spatial_dropout_rate=0.2, fc_dropout_rate=0.3):
    inputs = Input(shape=input_shape)
    x = inputs

    # Model cho age_estimation
    # Block 1
    x_age_estimate = SeparableConv2D(64, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x)
    x_age_estimate = BatchNormalization()(x_age_estimate)
    x_age_estimate = SpatialDropout2D(spatial_dropout_rate)(x_age_estimate)
    x_age_estimate = MaxPooling2D(pool_size=(2,2))(x_age_estimate)      # 90x90x64

    # Block 2
    x_age_estimate = SeparableConv2D(128, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x_age_estimate)
    x_age_estimate = BatchNormalization()(x_age_estimate)
    x_age_estimate = SpatialDropout2D(spatial_dropout_rate)(x_age_estimate)
    x_age_estimate = MaxPooling2D(pool_size=(2,2))(x_age_estimate)      # 45x45x128

    # Block 3
    x_age_estimate = SeparableConv2D(128, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x_age_estimate)
    x_age_estimate = BatchNormalization()(x_age_estimate)
    x_age_estimate = SpatialDropout2D(spatial_dropout_rate)(x_age_estimate)
    x_age_estimate = MaxPooling2D(pool_size=(2,2))(x_age_estimate)      # 22x22x128

    # Block 4
    x_age_estimate = SeparableConv2D(256, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x_age_estimate)
    x_age_estimate = BatchNormalization()(x_age_estimate)
    x_age_estimate = SpatialDropout2D(spatial_dropout_rate)(x_age_estimate)
    x_age_estimate = MaxPooling2D(pool_size=(2,2))(x_age_estimate)      # 11x11x256

    # Block 5
    x_age_estimate = SeparableConv2D(256, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x_age_estimate)
    x_age_estimate = BatchNormalization()(x_age_estimate)
    x_age_estimate = SpatialDropout2D(spatial_dropout_rate)(x_age_estimate)
    x_age_estimate = MaxPooling2D(pool_size=(2,2))(x_age_estimate)      # 5x5x256

    x_age_estimate = Flatten()(x_age_estimate)

    # Fully connected layers cho Age Estimation
    x_age_estimate = Dense(128, activation='relu', kernel_initializer='he_uniform')(x_age_estimate)
    x_age_estimate = BatchNormalization()(x_age_estimate)
    x_age_estimate = Dropout(fc_dropout_rate)(x_age_estimate)

    x_age_estimate = Dense(64, activation='relu', kernel_initializer='he_uniform')(x_age_estimate)
    x_age_estimate = BatchNormalization()(x_age_estimate)
    x_age_estimate = Dropout(fc_dropout_rate)(x_age_estimate)

    x_age_estimate = Dense(32, activation='relu', kernel_initializer='he_uniform')(x_age_estimate)
    x_age_estimate = BatchNormalization()(x_age_estimate)
    x_age_estimate = Dropout(fc_dropout_rate)(x_age_estimate)

    # Output cho tuổi (age estimation) - sử dụng MSE cho regression
    age_estimation_out = Dense(1, activation='relu', name='age_estimation_output')(x_age_estimate)



    # Model Gender Classification (4 blocks)
    # Block 1
    x_gender = SeparableConv2D(256, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x)
    x_gender = BatchNormalization()(x_gender)
    x_gender = SpatialDropout2D(spatial_dropout_rate)(x_gender)
    x_gender = MaxPooling2D(pool_size=(2,2))(x_gender)  

    # Block 2
    x_gender = SeparableConv2D(256, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x_gender)
    x_gender = BatchNormalization()(x_gender)
    x_gender = SpatialDropout2D(spatial_dropout_rate)(x_gender)
    x_gender = MaxPooling2D(pool_size=(2,2))(x_gender)  

    x_gender = Flatten()(x_gender)

    # Fully connected layers cho Gender Classification
    x_gender = Dense(128, activation='relu', kernel_initializer='he_uniform')(x_gender)
    x_gender = BatchNormalization()(x_gender)
    x_gender = Dropout(fc_dropout_rate)(x_gender)

    x_gender = Dense(64, activation='relu', kernel_initializer='he_uniform')(x_gender)
    x_gender = BatchNormalization()(x_gender)
    x_gender = Dropout(fc_dropout_rate)(x_gender)

    x_gender = Dense(32, activation='relu', kernel_initializer='he_uniform')(x_gender)
    x_gender = BatchNormalization()(x_gender)
    x_gender = Dropout(fc_dropout_rate)(x_gender)

    # Output cho giới tính (gender) - sử dụng binary_crossentropy cho classification
    gender_out = Dense(2, activation='softmax', name='gender_output')(x_gender)



    # Model cho age classification (3 blocks)
    # Block 1
    x_age = SeparableConv2D(256, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x)
    x_age = BatchNormalization()(x_age)
    x_age = SpatialDropout2D(spatial_dropout_rate)(x_age)
    x_age = MaxPooling2D(pool_size=(2,2))(x_age)  

    # Block 2
    x_age = SeparableConv2D(256, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x_age)
    x_age = BatchNormalization()(x_age)
    x_age = SpatialDropout2D(spatial_dropout_rate)(x_age)
    x_age = MaxPooling2D(pool_size=(2,2))(x_age)  

    # Block 3
    x_age = SeparableConv2D(256, (3,3), padding='same', activation='relu', 
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(x_age)
    x_age = BatchNormalization()(x_age)
    x_age = SpatialDropout2D(spatial_dropout_rate)(x_age)
    x_age = MaxPooling2D(pool_size=(2,2))(x_age)  

    x_age = Flatten()(x_age)

    # Fully connected layers cho Age Classification
    x_age = Dense(128, activation='relu', kernel_initializer='he_uniform')(x_age)
    x_age = BatchNormalization()(x_age)
    x_age = Dropout(fc_dropout_rate)(x_age)

    x_age = Dense(64, activation='relu', kernel_initializer='he_uniform')(x_age)
    x_age = BatchNormalization()(x_age)
    x_age = Dropout(fc_dropout_rate)(x_age)

    x_age = Dense(32, activation='relu', kernel_initializer='he_uniform')(x_age)
    x_age = BatchNormalization()(x_age)
    x_age = Dropout(fc_dropout_rate)(x_age)

    # Output cho nhóm tuổi (age group) - sử dụng categorical_crossentropy cho classification
    age_out = Dense(age_groups, activation='softmax', name='age_output')(x_age)
    
    # Tạo mô hình
    model = Model(inputs, [age_out, gender_out, age_estimation_out])
    return model

# Build model với 5 nhóm tuổi
model = build_utkface_cnn(input_shape=(180,180,3), age_groups=5)
model.summary()




#generator cho các tập train, val và test
BATCH_SIZE = 32
IMG_SIZE = (180, 180)
N_AGE_GROUP = 5

train_gen = UTKFaceSequence(df_train, folder, batch_size=BATCH_SIZE, input_size=IMG_SIZE, num_age_classes=N_AGE_GROUP, shuffle=True)
val_gen = UTKFaceSequence(df_val, folder, batch_size=BATCH_SIZE, input_size=IMG_SIZE, num_age_classes=N_AGE_GROUP, shuffle=False)
test_gen = UTKFaceSequence(df_test, folder, batch_size=BATCH_SIZE, input_size=IMG_SIZE, num_age_classes=N_AGE_GROUP, shuffle=False)

X, y = train_gen[0]
print(f"X shape: {X.shape}")  # Kiểm tra kích thước ảnh trong batch
print(f"y['age_output'] shape: {y['age_output'].shape}")  # Kiểm tra nhãn age
print(f"y['gender_output'] shape: {y['gender_output'].shape}")  # Kiểm tra nhãn gender


#compile model
model.compile(
    optimizer='adam', 
    loss={
        'age_output': 'categorical_crossentropy',  # Phân loại nhóm tuổi (Age Classification)
        'gender_output': 'categorical_crossentropy',    # Phân loại giới tính (Gender Classification)
        'age_estimation_output': 'mean_squared_error'  # Dự đoán tuổi (Age Estimation)
    },
    metrics={
        'age_output': 'accuracy',  # Đo độ chính xác phân loại nhóm tuổi
        'gender_output': 'accuracy',  # Đo độ chính xác phân loại giới tính
        'age_estimation_output': 'mae'  # MAE cho dự đoán tuổi (age estimation)
    }
)






#train
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch, lr):                      #thiet lap tu dong giam learning rate sau 9 epoch
    if epoch in [9, 18, 27, 36, 45]:
        return lr * 0.6
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[lr_scheduler]
)








#Vẽ đồ thị kết quả
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 4))

# (a) Age Estimation Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['age_estimation_output_loss'], color='blue', label='Train Loss')
plt.plot(history.history['val_age_estimation_output_loss'], color='red', label='Val Loss')
plt.title('(a) Age Estimation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# (b) Age Classification Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['age_output_loss'], color='blue', label='Train Loss')
plt.plot(history.history['val_age_output_loss'], color='red', label='Val Loss')
plt.title('(b) Age Classification')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# (c) Gender Classification Loss
plt.subplot(1, 3, 3)
plt.plot(history.history['gender_output_loss'], color='blue', label='Train Loss')
plt.plot(history.history['val_gender_output_loss'], color='red', label='Val Loss')
plt.title('(c) Gender Classification')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()




# Đánh giá mô hình trên tập test và kiểm tra các giá trị trả về
results = model.evaluate(test_gen)

# In ra các giá trị trả về
print(f"Results from model.evaluate(): {results}")


test_loss, test_age_loss, test_gender_loss, test_age_estimation_loss, test_age_estimation_mae, test_age_accuracy, test_gender_accuracy = results

# In kết quả
print(f"Test Loss: {test_loss}")
print(f"Test Age Classification Loss: {test_age_loss}")
print(f"Test Gender Classification Loss: {test_gender_loss}")
print(f"Test Age Estimation Loss: {test_age_estimation_loss}")
print(f"Test Age Estimation MAE: {test_age_estimation_mae}")
print(f"Test Age Classification Accuracy: {test_age_accuracy}")
print(f"Test Gender Classification Accuracy: {test_gender_accuracy}")




#lưu model
model_save_path = '/kaggle/working/trained_model_caitien_3.h5'
model.save(model_save_path)
print(f"Đã lưu model tại: {model_save_path}")









#kiem tra đầu ra model để test
print(model.output_names)