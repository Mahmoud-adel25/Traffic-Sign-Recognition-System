"""
Data Preprocessing Module for Traffic Sign Recognition
Handles image loading, preprocessing, augmentation, and dataset preparation
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    """Comprehensive data preprocessing class for traffic sign recognition"""
    
    def __init__(self, data_dir: str, img_size: int = 48):
        """
        Initialize the data preprocessor
        
        Args:
            data_dir: Path to the dataset directory
            img_size: Target image size for resizing
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.class_names = None
        self.class_distribution = None
        
    def load_class_metadata(self) -> pd.DataFrame:
        """Load class metadata and sign names"""
        meta_path = self.data_dir / "Meta.csv"
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
            self.class_names = meta_df['SignName'].values
            return meta_df
        else:
            print("Warning: Meta.csv not found. Using numeric class labels.")
            return None
    
    def load_data(self, csv_file: str, base_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess images from CSV file
        
        Args:
            csv_file: Path to CSV file with image paths and labels
            base_dir: Base directory for images
            
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        df = pd.read_csv(csv_file)
        images, labels = [], []
        
        print(f"Loading {len(df)} images...")
        
        for i, row in df.iterrows():
            if i % 1000 == 0:
                print(f"Processed {i}/{len(df)} images...")
                
            img_path = base_dir / row['Path']
            if img_path.exists():
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = self.preprocess_image(img)
                    images.append(img)
                    labels.append(row['ClassId'])
        
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images with shape {X.shape}")
        return X, y
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def analyze_class_distribution(self, labels: np.ndarray) -> dict:
        """
        Analyze class distribution in the dataset
        
        Args:
            labels: Array of class labels
            
        Returns:
            Dictionary with class distribution statistics
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        self.class_distribution = distribution
        
        # Calculate statistics
        total_samples = len(labels)
        min_samples = min(distribution.values())
        max_samples = max(distribution.values())
        mean_samples = np.mean(list(distribution.values()))
        
        print(f"Class Distribution Analysis:")
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {len(distribution)}")
        print(f"Min samples per class: {min_samples}")
        print(f"Max samples per class: {max_samples}")
        print(f"Mean samples per class: {mean_samples:.1f}")
        
        return {
            'distribution': distribution,
            'total_samples': total_samples,
            'num_classes': len(distribution),
            'min_samples': min_samples,
            'max_samples': max_samples,
            'mean_samples': mean_samples
        }
    
    def create_data_generators(self, X_train: np.ndarray, y_train: np.ndarray,
                             validation_split: float = 0.1) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Create data generators with augmentation for training
        
        Args:
            X_train: Training images
            y_train: Training labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (training_generator, validation_generator)
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=15,           # Random rotation up to 15 degrees
            width_shift_range=0.1,       # Random horizontal shift
            height_shift_range=0.1,      # Random vertical shift
            zoom_range=0.1,              # Random zoom
            shear_range=0.1,             # Random shear
            horizontal_flip=False,       # No horizontal flip for traffic signs
            fill_mode='nearest',         # Fill strategy for transformed pixels
            validation_split=validation_split
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            validation_split=validation_split
        )
        
        # Fit the generators
        train_datagen.fit(X_train)
        val_datagen.fit(X_train)
        
        return train_datagen, val_datagen
    
    def visualize_class_distribution(self, labels: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize class distribution
        
        Args:
            labels: Array of class labels
            save_path: Optional path to save the plot
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(15, 8))
        plt.bar(unique, counts, alpha=0.7, color='skyblue')
        plt.xlabel('Class ID')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution in Dataset')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def visualize_sample_images(self, images: np.ndarray, labels: np.ndarray, 
                               num_samples: int = 16, save_path: Optional[str] = None):
        """
        Visualize sample images from the dataset
        
        Args:
            images: Array of images
            labels: Array of labels
            num_samples: Number of samples to display
            save_path: Optional path to save the plot
        """
        # Randomly select samples
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            img = images[idx]
            label = labels[idx]
            
            # Convert back to 0-255 range for display
            img_display = (img * 255).astype(np.uint8)
            
            axes[i].imshow(img_display)
            axes[i].set_title(f'Class {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Sample Traffic Sign Images', y=1.02, fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample images plot saved to {save_path}")
        
        plt.show()
    
    def prepare_dataset(self, train_csv: str, test_csv: str, 
                       validation_split: float = 0.1) -> dict:
        """
        Complete dataset preparation pipeline
        
        Args:
            train_csv: Path to training CSV file
            test_csv: Path to test CSV file
            validation_split: Fraction of training data for validation
            
        Returns:
            Dictionary containing all prepared data
        """
        print("🚀 Starting dataset preparation...")
        
        # Load class metadata
        meta_df = self.load_class_metadata()
        
        # Load training data
        print("📂 Loading training data...")
        X_train_full, y_train_full = self.load_data(train_csv, self.data_dir)
        
        # Load test data
        print("📂 Loading test data...")
        X_test, y_test = self.load_data(test_csv, self.data_dir)
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=validation_split, 
            stratify=y_train_full, 
            random_state=42
        )
        
        # Analyze class distribution
        print("📊 Analyzing class distribution...")
        train_stats = self.analyze_class_distribution(y_train)
        val_stats = self.analyze_class_distribution(y_val)
        test_stats = self.analyze_class_distribution(y_test)
        
        # Create data generators
        print("🔄 Creating data generators...")
        train_datagen, val_datagen = self.create_data_generators(X_train, y_train, validation_split)
        
        # Prepare dataset dictionary
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_datagen': train_datagen,
            'val_datagen': val_datagen,
            'class_names': self.class_names,
            'num_classes': len(np.unique(y_train)),
            'img_size': self.img_size,
            'train_stats': train_stats,
            'val_stats': val_stats,
            'test_stats': test_stats
        }
        
        print("✅ Dataset preparation completed!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {dataset['num_classes']}")
        
        return dataset

def main():
    """Example usage of the DataPreprocessor"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor("Data/Dataset", img_size=48)
    
    # Prepare dataset
    dataset = preprocessor.prepare_dataset("Data/Dataset/Train.csv", "Data/Dataset/Test.csv")
    
    # Visualize class distribution
    preprocessor.visualize_class_distribution(dataset['y_train'], "class_distribution.png")
    
    # Visualize sample images
    preprocessor.visualize_sample_images(dataset['X_train'], dataset['y_train'], 
                                       save_path="sample_images.png")
    
    return dataset

if __name__ == "__main__":
    main()
