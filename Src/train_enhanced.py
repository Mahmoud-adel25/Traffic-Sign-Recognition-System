# src/train_enhanced.py
# Enhanced Traffic Sign Recognition Training Script
# Comprehensive training with advanced features and detailed evaluation

import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from data_preprocessing import DataPreprocessor
from model_evaluation import ModelEvaluator
from utils import print_model_summary, save_model_info, plot_training_progress

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 15
DATA_DIR = Path("Data/Dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Training configuration
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# Callbacks configuration
PATIENCE = 5
MIN_DELTA = 0.001
FACTOR = 0.5
MIN_LR = 1e-7

print("🚦 Traffic Sign Recognition - Enhanced Training Script")
print("=" * 60)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Validation Split: {VALIDATION_SPLIT}")
print("=" * 60)

# -----------------------------
# Data Preprocessing
# -----------------------------
def prepare_dataset():
    """Prepare and preprocess the dataset"""
    print("\n📂 Preparing dataset...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(str(DATA_DIR), IMG_SIZE)
    
    # Prepare dataset
    dataset = preprocessor.prepare_dataset(
        "Data/Dataset/Train.csv", 
        "Data/Dataset/Test.csv", 
        VALIDATION_SPLIT
    )
    
    return dataset, preprocessor

# -----------------------------
# Model Architectures
# -----------------------------
def build_enhanced_cnn(input_shape, num_classes):
    """
    Build an enhanced CNN model with better architecture
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_enhanced_mobilenet(input_shape, num_classes):
    """
    Build an enhanced MobileNetV2 model with better fine-tuning
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# -----------------------------
# Training Callbacks
# -----------------------------
def create_callbacks(model_name):
    """Create training callbacks for better training"""
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=FACTOR,
            patience=PATIENCE//2,
            min_lr=MIN_LR,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=MODEL_DIR / f"{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

# -----------------------------
# Training Functions
# -----------------------------
def train_model(model, model_name, train_generator, val_data, callbacks):
    """Train a model with comprehensive logging"""
    print(f"\n🧠 Training {model_name}...")
    print("-" * 40)
    
    # Print model summary
    print_model_summary(model, model_name)
    
    # Training start time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Training end time
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
    
    # Save model
    model.save(MODEL_DIR / f"{model_name}.h5")
    print(f"💾 Model saved as {model_name}.h5")
    
    # Save model info
    save_model_info(model, model_name, str(MODEL_DIR))
    
    # Plot training progress
    plot_training_progress(
        history, 
        model_name, 
        str(MODEL_DIR / f"{model_name}_training_progress.png")
    )
    
    return history

# -----------------------------
# Main Training Pipeline
# -----------------------------
def main():
    """Main training pipeline"""
    print("\n🚀 Starting Enhanced Training Pipeline")
    print("=" * 60)
    
    # Prepare dataset
    dataset, preprocessor = prepare_dataset()
    
    # Extract data
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    train_datagen = dataset['train_datagen']
    class_names = dataset['class_names']
    num_classes = dataset['num_classes']
    
    print(f"\n📊 Dataset Summary:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names)
    
    # Train Custom CNN
    print("\n" + "="*60)
    print("TRAINING CUSTOM CNN")
    print("="*60)
    
    cnn_model = build_enhanced_cnn((IMG_SIZE, IMG_SIZE, 3), num_classes)
    cnn_callbacks = create_callbacks("cnn")
    
    cnn_history = train_model(
        cnn_model,
        "cnn",
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        (X_val, y_val),
        cnn_callbacks
    )
    
    # Train MobileNetV2
    print("\n" + "="*60)
    print("TRAINING MOBILENETV2")
    print("="*60)
    
    mobilenet_model = build_enhanced_mobilenet((IMG_SIZE, IMG_SIZE, 3), num_classes)
    mobilenet_callbacks = create_callbacks("mobilenet")
    
    mobilenet_history = train_model(
        mobilenet_model,
        "mobilenet",
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        (X_val, y_val),
        mobilenet_callbacks
    )
    
    # -----------------------------
    # Comprehensive Evaluation
    # -----------------------------
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Evaluate CNN
    print("\n📊 Evaluating Custom CNN...")
    cnn_results = evaluator.evaluate_model(cnn_model, X_test, y_test, "Custom CNN")
    
    # Evaluate MobileNetV2
    print("\n📊 Evaluating MobileNetV2...")
    mobilenet_results = evaluator.evaluate_model(mobilenet_model, X_test, y_test, "MobileNetV2")
    
    # Generate visualizations
    print("\n📈 Generating evaluation visualizations...")
    
    # Confusion matrices
    evaluator.plot_confusion_matrix(
        cnn_results['confusion_matrix'], 
        "Custom CNN",
        str(MODEL_DIR / "cnn_confusion_matrix_enhanced.png")
    )
    
    evaluator.plot_confusion_matrix(
        mobilenet_results['confusion_matrix'], 
        "MobileNetV2",
        str(MODEL_DIR / "mobilenet_confusion_matrix_enhanced.png")
    )
    
    # Class performance
    evaluator.plot_class_performance(
        cnn_results,
        str(MODEL_DIR / "cnn_class_performance.png")
    )
    
    evaluator.plot_class_performance(
        mobilenet_results,
        str(MODEL_DIR / "mobilenet_class_performance.png")
    )
    
    # Model comparison
    evaluator.compare_models(
        ["Custom CNN", "MobileNetV2"],
        str(MODEL_DIR / "model_comparison.png")
    )
    
    # Generate detailed report
    print("\n📋 Generating detailed evaluation report...")
    report = evaluator.generate_detailed_report(
        str(MODEL_DIR / "evaluation_report.txt")
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✅ Custom CNN Accuracy: {cnn_results['accuracy']:.4f}")
    print(f"✅ MobileNetV2 Accuracy: {mobilenet_results['accuracy']:.4f}")
    print(f"📁 Models saved in: {MODEL_DIR}")
    print(f"📊 Evaluation report: {MODEL_DIR}/evaluation_report.txt")
    print("="*60)
    
    return {
        'cnn_model': cnn_model,
        'mobilenet_model': mobilenet_model,
        'cnn_results': cnn_results,
        'mobilenet_results': mobilenet_results,
        'evaluator': evaluator
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    # Run training pipeline
    results = main()
