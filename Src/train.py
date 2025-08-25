# src/train.py
# Train CNN & MobileNetV2 on GTSRB dataset (with train.csv/test.csv)

import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 10
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helper: Load dataset
# -----------------------------
def load_data(csv_file, base_dir, img_size):
    df = pd.read_csv(csv_file)
    images, labels = [], []
    for i, row in df.iterrows():
        img_path = base_dir / row['Path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(row['ClassId'])
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)
    return X, y


print("📂 Loading training data...")
X, y = load_data(DATA_DIR / "train.csv", DATA_DIR, IMG_SIZE)
print("📂 Loading test data...")
X_test, y_test = load_data(DATA_DIR / "test.csv", DATA_DIR, IMG_SIZE)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

num_classes = len(np.unique(y))

# -----------------------------
# Data Augmentation
# -----------------------------
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1)
datagen.fit(X_train)

# -----------------------------
# Model 1: Custom CNN
# -----------------------------
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

cnn = build_cnn((IMG_SIZE, IMG_SIZE, 3), num_classes)

print("🧠 Training CNN...")
cnn.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS)

cnn.save(MODEL_DIR / "cnn_model.h5")

# -----------------------------
# Model 2: MobileNetV2 Transfer Learning
# -----------------------------
def build_mobilenet(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

mobilenet = build_mobilenet((IMG_SIZE, IMG_SIZE, 3), num_classes)

print("🧠 Training MobileNetV2...")
mobilenet.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(X_val, y_val),
              epochs=EPOCHS)

mobilenet.save(MODEL_DIR / "mobilenet_model.h5")

# -----------------------------
# Evaluation
# -----------------------------
print("📊 Evaluating CNN on test set...")
y_pred_cnn = np.argmax(cnn.predict(X_test), axis=1)
print(classification_report(y_test, y_pred_cnn))

print("📊 Evaluating MobileNetV2 on test set...")
y_pred_mobilenet = np.argmax(mobilenet.predict(X_test), axis=1)
print(classification_report(y_test, y_pred_mobilenet))

# Confusion matrix for CNN
cm = confusion_matrix(y_test, y_pred_cnn)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(MODEL_DIR / "cnn_confusion_matrix.png")
plt.close()

# Confusion matrix for MobileNetV2
cm2 = confusion_matrix(y_test, y_pred_mobilenet)
plt.figure(figsize=(12, 10))
sns.heatmap(cm2, annot=False, cmap="Greens")
plt.title("Confusion Matrix - MobileNetV2")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(MODEL_DIR / "mobilenet_confusion_matrix.png")
plt.close()

print("✅ Training complete. Models saved in /models")
