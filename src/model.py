import tensorflow as tf
import time
import numpy as np
import os
from tensorflow.keras import layers, Model # type: ignore

def build_model(input_shape=(227, 227, 3), backbone='custom_cnn', augment=True):
    inputs = layers.Input(shape=input_shape)
    
    # Lớp tăng cường dữ liệu (Data Augmentation)
    if augment:
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
    else:
        x = inputs

    if backbone == 'mobilenet_v2':
        # Load MobileNetV2 backbone (không bao gồm top classification head)
        from tensorflow.keras.applications import MobileNetV2
        
        # Rescale inputs từ [0, 1] về [-1, 1] theo chuẩn MobileNetV2
        x_scaled = layers.Rescaling(scale=2.0, offset=-1.0)(x)
        
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        # Mặc định để True để fine-tune toàn bộ mạng nếu cần
        base_model.trainable = True
        
        features = base_model(x_scaled)
        x = layers.GlobalAveragePooling2D()(features)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        shared_features = layers.Dropout(0.5)(x)
    else:
        # custom_cnn
        x = layers.Conv2D(96, (7, 7), strides=2, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(256, (5, 5), strides=2, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(384, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        shared_features = layers.Dropout(0.5)(x)

    gender_branch = layers.Dense(256, activation='relu')(shared_features)
    gender_branch = layers.Dropout(0.3)(gender_branch)
    gender_output = layers.Dense(2, activation='softmax', name='gender_output')(gender_branch)

    age_branch = layers.Dense(256, activation='relu')(shared_features)
    age_branch = layers.Dropout(0.3)(age_branch)
    age_output = layers.Dense(8, activation='softmax', name='age_output')(age_branch)

    model = Model(inputs=inputs, outputs=[gender_output, age_output])
    return model

def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        loss={
            'gender_output': 'categorical_crossentropy',
            'age_output': 'categorical_crossentropy'
        },
        loss_weights={
            'gender_output': 1.0,
            'age_output': 1.0
        },
        metrics={
            'gender_output': ['accuracy'],
            'age_output': ['accuracy']
        }
    )
    return model

def train_model(
    model=None,
    X=None,
    y_age=None,
    y_gender=None,
    train_generator=None,
    val_generator=None,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    model_save_path='outputs/trained_model.h5',
):
    """
    Huấn luyện mô hình sử dụng dữ liệu đã load sẵn vào RAM hoặc qua Generator.
    """
    # Tạo thư mục chứa mô hình nếu chưa tồn tại
    dir_name = os.path.dirname(model_save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Callbacks để tự động dừng và giảm LR
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    if train_generator is not None:
        print("[INFO] Bắt đầu huấn luyện bằng Generator...")
        start_time = time.time()
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        duration = time.time() - start_time
        print(f"[INFO] Huấn luyện hoàn tất sau {duration / 60:.2f} phút.")
        print(f"[INFO] Lưu mô hình cuối vào {model_save_path}...")
        model.save(model_save_path)
        return history, model

    # Kiểm tra đầu vào khi dùng RAM
    total_samples = X.shape[0]
    assert y_age.shape[0] == total_samples and y_gender.shape[0] == total_samples, "Kích thước không khớp!"

    # Tạo chỉ số train/val
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    val_samples = int(total_samples * validation_split)
    val_indices = indices[:val_samples]
    train_indices = indices[val_samples:]

    print(f"[INFO] Tổng mẫu: {total_samples}")
    print(f"[INFO] Mẫu train: {len(train_indices)}, Mẫu val: {len(val_indices)}")

    # Tách dữ liệu theo chỉ số
    X_train, y_age_train, y_gender_train = X[train_indices], y_age[train_indices], y_gender[train_indices]
    X_val, y_age_val, y_gender_val = X[val_indices], y_age[val_indices], y_gender[val_indices]

    y_train = {
        'age_output': y_age_train,
        'gender_output': y_gender_train
    }
    y_val = {
        'age_output': y_age_val,
        'gender_output': y_gender_val
    }

    print("[INFO] Bắt đầu huấn luyện...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    duration = time.time() - start_time
    print(f"[INFO] Huấn luyện hoàn tất sau {duration / 60:.2f} phút.")

    print(f"[INFO] Lưu mô hình vào {model_save_path}...")
    model.save(model_save_path)

    return history, model