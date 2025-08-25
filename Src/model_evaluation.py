"""
Model Evaluation Module for Traffic Sign Recognition
Provides comprehensive evaluation metrics, visualizations, and analysis tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation class for traffic sign recognition"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the model evaluator
        
        Args:
            class_names: List of class names for better visualization
        """
        self.class_names = class_names
        self.results = {}
        
    def evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray, 
                      y_test: np.ndarray, model_name: str) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained Keras model
            X_test: Test images
            y_test: Test labels
            model_name: Name of the model for results storage
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"🔍 Evaluating {model_name}...")
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Detailed classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.class_names if self.class_names else None,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        self.results[model_name] = results
        
        # Print summary
        print(f"📊 {model_name} Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, 
                            save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)):
        """
        Plot confusion matrix with detailed annotations
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names if self.class_names else range(cm.shape[1]),
                   yticklabels=self.class_names if self.class_names else range(cm.shape[0]))
        
        plt.title(f'Confusion Matrix - {model_name}\n(Percentage of True Labels)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_performance(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot per-class performance metrics
        
        Args:
            results: Model evaluation results
            save_path: Optional path to save the plot
        """
        class_report = results['classification_report']
        
        # Extract per-class metrics
        classes = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                classes.append(class_name)
                precision_scores.append(metrics['precision'])
                recall_scores.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Precision
        axes[0].bar(range(len(classes)), precision_scores, alpha=0.7, color='skyblue')
        axes[0].set_title('Per-Class Precision')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Precision')
        axes[0].set_xticks(range(len(classes)))
        axes[0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # Recall
        axes[1].bar(range(len(classes)), recall_scores, alpha=0.7, color='lightgreen')
        axes[1].set_title('Per-Class Recall')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Recall')
        axes[1].set_xticks(range(len(classes)))
        axes[1].set_xticklabels(classes, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # F1-Score
        axes[2].bar(range(len(classes)), f1_scores, alpha=0.7, color='salmon')
        axes[2].set_title('Per-Class F1-Score')
        axes[2].set_xlabel('Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_xticks(range(len(classes)))
        axes[2].set_xticklabels(classes, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Per-Class Performance - {results["model_name"]}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history: tf.keras.callbacks.History, 
                            model_name: str, save_path: Optional[str] = None):
        """
        Plot training history (accuracy and loss)
        
        Args:
            history: Training history from model.fit()
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'Model Accuracy - {model_name}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'Model Loss - {model_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, model_names: List[str], save_path: Optional[str] = None):
        """
        Compare multiple models side by side
        
        Args:
            model_names: List of model names to compare
            save_path: Optional path to save the plot
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Extract metrics for each model
        model_metrics = {}
        for name in model_names:
            if name in self.results:
                model_metrics[name] = [self.results[name][metric] for metric in metrics]
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(model_metrics)
        
        for i, (model_name, values) in enumerate(model_metrics.items()):
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * (len(model_metrics) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (model_name, values) in enumerate(model_metrics.items()):
            for j, value in enumerate(values):
                ax.text(j + i * width, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a detailed evaluation report
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TRAFFIC SIGN RECOGNITION - MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        for model_name, results in self.results.items():
            report_lines.append(f"📊 {model_name.upper()} RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Overall Accuracy: {results['accuracy']:.4f}")
            report_lines.append(f"Precision: {results['precision']:.4f}")
            report_lines.append(f"Recall: {results['recall']:.4f}")
            report_lines.append(f"F1-Score: {results['f1_score']:.4f}")
            report_lines.append("")
            
            # Add classification report
            report_lines.append("Detailed Classification Report:")
            report_lines.append(str(classification_report(
                results['y_true'], results['y_pred'],
                target_names=self.class_names if self.class_names else None
            )))
            report_lines.append("")
        
        # Model comparison summary
        if len(self.results) > 1:
            report_lines.append("🏆 MODEL COMPARISON SUMMARY")
            report_lines.append("-" * 40)
            
            comparison_data = []
            for model_name, results in self.results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy']:.4f}",
                    'Precision': f"{results['precision']:.4f}",
                    'Recall': f"{results['recall']:.4f}",
                    'F1-Score': f"{results['f1_score']:.4f}"
                })
            
            df = pd.DataFrame(comparison_data)
            report_lines.append(df.to_string(index=False))
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Detailed report saved to {save_path}")
        
        return report_text
    
    def analyze_misclassifications(self, results: Dict, X_test: np.ndarray, 
                                 num_examples: int = 10, save_path: Optional[str] = None):
        """
        Analyze and visualize misclassified examples
        
        Args:
            results: Model evaluation results
            X_test: Test images
            num_examples: Number of misclassified examples to show
            save_path: Optional path to save the plot
        """
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Find misclassified examples
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return
        
        # Select random misclassified examples
        if len(misclassified_indices) > num_examples:
            selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
        else:
            selected_indices = misclassified_indices
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(selected_indices):
            img = X_test[idx]
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            
            # Convert back to 0-255 range for display
            img_display = (img * 255).astype(np.uint8)
            
            axes[i].imshow(img_display)
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                            color='red' if true_label != pred_label else 'green')
            axes[i].axis('off')
        
        plt.suptitle(f'Misclassified Examples - {results["model_name"]}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Misclassification analysis saved to {save_path}")
        
        plt.show()

def main():
    """Example usage of the ModelEvaluator"""
    # This would be used after training models
    print("Model evaluation module loaded successfully!")
    print("Use this module to evaluate trained models and generate comprehensive reports.")

if __name__ == "__main__":
    main()
