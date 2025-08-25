"""
Utility Functions for Traffic Sign Recognition
Helper functions for data handling, visualization, and model operations
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import warnings
warnings.filterwarnings('ignore')

def load_class_names(meta_path: str) -> List[str]:
    """
    Load class names from metadata file
    
    Args:
        meta_path: Path to Meta.csv file
        
    Returns:
        List of class names
    """
    try:
        meta_df = pd.read_csv(meta_path)
        return meta_df['SignName'].values.tolist()
    except FileNotFoundError:
        print(f"Warning: {meta_path} not found. Using numeric class labels.")
        return None
    except KeyError:
        print("Warning: 'SignName' column not found in metadata. Using numeric class labels.")
        return None

def preprocess_image_for_prediction(img: np.ndarray, target_size: int = 48) -> np.ndarray:
    """
    Preprocess a single image for model prediction
    
    Args:
        img: Input image (BGR format from OpenCV)
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image ready for model input
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    
    # Normalize pixel values
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def create_model_summary_table(model: tf.keras.Model) -> pd.DataFrame:
    """
    Create a summary table of model parameters
    
    Args:
        model: Keras model
        
    Returns:
        DataFrame with layer information
    """
    summary_list = []
    
    for layer in model.layers:
        layer_info = {
            'Layer Name': layer.name,
            'Layer Type': layer.__class__.__name__,
            'Output Shape': str(layer.output_shape),
            'Parameters': layer.count_params()
        }
        summary_list.append(layer_info)
    
    return pd.DataFrame(summary_list)

def plot_model_architecture(model: tf.keras.Model, save_path: Optional[str] = None):
    """
    Plot model architecture summary
    
    Args:
        model: Keras model
        save_path: Optional path to save the plot
    """
    # Get model summary
    summary_df = create_model_summary_table(model)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, len(summary_df) * 0.4))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title(f'Model Architecture Summary - {model.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model architecture saved to {save_path}")
    
    plt.show()

def calculate_model_size(model: tf.keras.Model) -> Dict[str, float]:
    """
    Calculate model size in different units
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model sizes in different units
    """
    total_params = model.count_params()
    
    # Assuming float32 (4 bytes per parameter)
    size_bytes = total_params * 4
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    
    return {
        'parameters': total_params,
        'size_bytes': size_bytes,
        'size_kb': size_kb,
        'size_mb': size_mb
    }

def save_model_info(model: tf.keras.Model, model_name: str, save_dir: str = "models"):
    """
    Save comprehensive model information
    
    Args:
        model: Keras model
        model_name: Name of the model
        save_dir: Directory to save information
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Calculate model size
    size_info = calculate_model_size(model)
    
    # Create model info dictionary
    model_info = {
        'model_name': model_name,
        'total_parameters': size_info['parameters'],
        'model_size_mb': size_info['size_mb'],
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'number_of_layers': len(model.layers)
    }
    
    # Save as JSON
    import json
    info_path = save_path / f"{model_name}_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model information saved to {info_path}")
    return model_info

def create_data_visualization(X: np.ndarray, y: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            num_samples: int = 16, save_path: Optional[str] = None):
    """
    Create comprehensive data visualization
    
    Args:
        X: Image data
        y: Labels
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.ravel()
    
    # Randomly select samples
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img = X[idx]
        label = y[idx]
        
        # Convert back to 0-255 range for display
        img_display = (img * 255).astype(np.uint8)
        
        axes[i].imshow(img_display)
        
        if class_names and label < len(class_names):
            title = f'{class_names[label]} (Class {label})'
        else:
            title = f'Class {label}'
        
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Sample Traffic Sign Images from Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data visualization saved to {save_path}")
    
    plt.show()

def plot_training_progress(history: tf.keras.callbacks.History, 
                          model_name: str, save_path: Optional[str] = None):
    """
    Plot comprehensive training progress
    
    Args:
        history: Training history
        model_name: Name of the model
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training loss
    axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning curves (log scale for loss)
    axes[1, 0].semilogy(history.history['loss'], label='Training', linewidth=2)
    axes[1, 0].semilogy(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Model Loss (Log Scale)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy improvement
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)
    
    axes[1, 1].plot(epochs, [acc * 100 for acc in train_acc], label='Training', linewidth=2)
    axes[1, 1].plot(epochs, [acc * 100 for acc in val_acc], label='Validation', linewidth=2)
    axes[1, 1].set_title('Model Accuracy (%)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Progress - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training progress saved to {save_path}")
    
    plt.show()

def get_prediction_with_confidence(model: tf.keras.Model, 
                                 img: np.ndarray, 
                                 class_names: Optional[List[str]] = None,
                                 top_k: int = 5) -> Dict[str, Any]:
    """
    Get model prediction with confidence scores and top-k predictions
    
    Args:
        model: Trained model
        img: Preprocessed image
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    # Make prediction
    predictions = model.predict(img, verbose=0)
    
    # Get top-k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_probabilities = predictions[0][top_indices]
    
    # Prepare results
    results = {
        'predicted_class': int(top_indices[0]),
        'confidence': float(top_probabilities[0]),
        'top_predictions': []
    }
    
    # Add class names if available
    for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
        prediction = {
            'rank': i + 1,
            'class_id': int(idx),
            'probability': float(prob),
            'confidence_percentage': float(prob * 100)
        }
        
        if class_names and idx < len(class_names):
            prediction['class_name'] = class_names[idx]
        else:
            prediction['class_name'] = f'Class {idx}'
        
        results['top_predictions'].append(prediction)
    
    return results

def create_prediction_visualization(img: np.ndarray, 
                                  prediction_results: Dict[str, Any],
                                  original_img: Optional[np.ndarray] = None,
                                  save_path: Optional[str] = None):
    """
    Create visualization of prediction results
    
    Args:
        img: Preprocessed image used for prediction
        prediction_results: Results from get_prediction_with_confidence
        original_img: Original image (for display)
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    if original_img is not None:
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
    else:
        # Convert back to 0-255 range for display
        img_display = (img[0] * 255).astype(np.uint8)
        axes[0].imshow(img_display)
        axes[0].set_title('Input Image')
    
    axes[0].axis('off')
    
    # Display top predictions
    top_preds = prediction_results['top_predictions']
    classes = [pred['class_name'] for pred in top_preds]
    probabilities = [pred['confidence_percentage'] for pred in top_preds]
    
    bars = axes[1].barh(range(len(classes)), probabilities, color='skyblue', alpha=0.7)
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Top Predictions')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1f}%', va='center', fontweight='bold')
    
    plt.suptitle(f'Prediction Results\nPredicted: {top_preds[0]["class_name"]} ({top_preds[0]["confidence_percentage"]:.1f}%)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    
    plt.show()

def print_model_summary(model: tf.keras.Model, model_name: str):
    """
    Print comprehensive model summary
    
    Args:
        model: Keras model
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"MODEL SUMMARY: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Basic model info
    size_info = calculate_model_size(model)
    print(f"Model Name: {model_name}")
    print(f"Total Parameters: {size_info['parameters']:,}")
    print(f"Model Size: {size_info['size_mb']:.2f} MB")
    print(f"Input Shape: {model.input_shape}")
    print(f"Output Shape: {model.output_shape}")
    print(f"Number of Layers: {len(model.layers)}")
    
    # Layer summary
    print(f"\n{'='*40}")
    print("LAYER SUMMARY")
    print(f"{'='*40}")
    
    summary_df = create_model_summary_table(model)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")

def main():
    """Example usage of utility functions"""
    print("Utility functions module loaded successfully!")
    print("Available functions:")
    print("- load_class_names()")
    print("- preprocess_image_for_prediction()")
    print("- create_model_summary_table()")
    print("- plot_model_architecture()")
    print("- calculate_model_size()")
    print("- save_model_info()")
    print("- create_data_visualization()")
    print("- plot_training_progress()")
    print("- get_prediction_with_confidence()")
    print("- create_prediction_visualization()")
    print("- print_model_summary()")

if __name__ == "__main__":
    main()
