import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =========================================================
# CẤU HÌNH
# =========================================================
PROJECT_DIR = r"D:\plant_ai_project"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_300")

MODEL_PATH = os.path.join(PROJECT_DIR, "plant_model.keras")
CLASS_NAMES_PATH = os.path.join(PROJECT_DIR, "class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 20

CLASS_NAMES = ["diseased", "healthy", "wilted"]

print("=" * 70)
print("TRAINING MODEL NHẬN DIỆN CÂY")
print("=" * 70)

# Kiểm tra dataset
for class_name in CLASS_NAMES:
    folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục lớp: {folder}\n"
            f"Hãy kiểm tra dataset_300."
        )

# ---------------------------------------------------------
# Tạo train/validation dataset
# ---------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nCLASS MAPPING CỐ ĐỊNH:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} -> {name}")

# ---------------------------------------------------------
# Đếm ảnh trong train để tạo class weight
# ---------------------------------------------------------
counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

for _, labels in train_ds:
    labels_np = labels.numpy()
    for label in labels_np:
        counts[int(label)] += 1

print("\nSỐ ẢNH TRAIN:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:10s}: {counts[i]}")

total = counts.sum()
class_weight = {
    i: float(total / (len(CLASS_NAMES) * counts[i]))
    for i in range(len(CLASS_NAMES))
    if counts[i] > 0
}

print("\nCLASS WEIGHT:")
for i, weight in class_weight.items():
    print(f"  {CLASS_NAMES[i]}: {weight:.4f}")

# ---------------------------------------------------------
# Tối ưu pipeline
# ---------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------------------------------------------------------
# Data augmentation
# Chỉ hoạt động trong lúc train
# ---------------------------------------------------------
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# ---------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3), name="image")

x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input, name="mobilenet_preprocess")(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="prediction"
)(x)

model = models.Model(inputs, outputs)

# ---------------------------------------------------------
# Loss
# ---------------------------------------------------------
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=0.05
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------
callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# =========================================================
# PHASE 1: TRAIN HEAD
# =========================================================
print("\n" + "=" * 70)
print("PHASE 1 - HUẤN LUYỆN LỚP PHÂN LOẠI")
print("=" * 70)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

# =========================================================
# PHASE 2: FINE-TUNING
# =========================================================
print("\n" + "=" * 70)
print("PHASE 2 - FINE TUNING MOBILENETV2")
print("=" * 70)

base_model.trainable = True

# Chỉ mở khoảng 40 layer cuối.
# BatchNormalization vẫn giữ frozen để training ổn định.
for layer in base_model.layers[:-40]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

# =========================================================
# LOAD MODEL TỐT NHẤT
# =========================================================
print("\nĐang tải model tốt nhất...")

model = tf.keras.models.load_model(MODEL_PATH)

# Lưu class mapping chính xác
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=4)

# =========================================================
# ĐÁNH GIÁ
# =========================================================
print("\n" + "=" * 70)
print("ĐÁNH GIÁ MODEL")
print("=" * 70)

loss, accuracy = model.evaluate(val_ds, verbose=1)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss    : {loss:.4f}")

print("\nModel đã lưu:")
print(MODEL_PATH)

print("\nClass mapping đã lưu:")
print(CLASS_NAMES_PATH)

print("\n" + "=" * 70)
print("TRAIN HOÀN THÀNH")
print("=" * 70)
