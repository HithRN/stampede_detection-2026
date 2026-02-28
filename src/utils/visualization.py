"""
Visualization utilities for stampede detection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

from config.config import Config


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history object
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=Config.DPI)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.savefig(os.path.join(Config.FIGURES_DIR, 'training_history.png'), dpi=Config.DPI)
    
    plt.close()


def visualize_feature_importance(flow_sequences, scalar_features, labels, save_path=None):
    """
    Visualize the additional features to understand their contribution.
    
    Args:
        flow_sequences: Flow sequences array
        scalar_features: Scalar features array
        labels: Label array
        save_path: Path to save the plot (optional)
    """
    feature_names = ["Flow Acceleration", "Flow Divergence", "Scene Changes", "Motion Entropy"]
    
    plt.figure(figsize=(15, 10))
    
    # Process each sequence
    num_sequences = min(5, len(flow_sequences))
    for seq_idx in range(num_sequences):
        category = Config.CLASS_NAMES[labels[seq_idx]]
        
        # Plot the 4 scalar features over time for this sequence
        plt.subplot(num_sequences, 1, seq_idx + 1)
        
        for feature_idx in range(4):
            feature_values = scalar_features[seq_idx, :, feature_idx]
            plt.plot(feature_values, label=feature_names[feature_idx], marker='o')
        
        plt.title(f"Sequence {seq_idx + 1} (Category: {category})")
        plt.xlabel("Frame")
        plt.ylabel("Feature Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=Config.DPI)
    else:
        plt.savefig(os.path.join(Config.FIGURES_DIR, 'feature_visualization.png'), dpi=Config.DPI)
    
    print(f"Feature importance visualization saved")
    plt.close()


def visualize_sample_predictions(model, X_flow, X_scalar, y_true, num_samples=5, save_path=None):
    """
    Visualize sample predictions with their optical flow.
    
    Args:
        model: Trained model
        X_flow: Flow sequences
        X_scalar: Scalar features
        y_true: True labels
        num_samples: Number of samples to visualize
        save_path: Path to save the plot (optional)
    """
    # Get predictions
    predictions = model.predict([X_flow[:num_samples], X_scalar[:num_samples]], verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 3))
    
    for i in range(num_samples):
        # Show first, middle, and last frame optical flow
        frames_to_show = [0, Config.SEQUENCE_LENGTH // 2, Config.SEQUENCE_LENGTH - 1]
        
        for j, frame_idx in enumerate(frames_to_show):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            
            # Visualize optical flow as HSV
            flow = X_flow[i, frame_idx]
            mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
            
            hsv = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            ax.imshow(rgb)
            ax.set_title(f"Frame {frame_idx}")
            ax.axis('off')
        
        # Show prediction result
        ax = axes[i, 3] if num_samples > 1 else axes[3]
        
        true_label = Config.CLASS_NAMES[y_true[i]]
        pred_label = Config.CLASS_NAMES[y_pred[i]]
        confidence = predictions[i, y_pred[i]] * 100
        
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        
        ax.text(0.5, 0.6, f"True: {true_label}", ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.4, f"Pred: {pred_label}", ha='center', va='center', 
                fontsize=12, color=color, weight='bold')
        ax.text(0.5, 0.2, f"Conf: {confidence:.1f}%", ha='center', va='center', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=Config.DPI)
    else:
        plt.savefig(os.path.join(Config.FIGURES_DIR, 'sample_predictions.png'), dpi=Config.DPI)
    
    print(f"Sample predictions visualization saved")
    plt.close()


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = Config.CLASS_NAMES
    
    plt.figure(figsize=Config.FIGURE_SIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=Config.DPI)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curves(fpr, tpr, roc_auc, class_names=None, title='ROC Curves', save_path=None):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        fpr: False positive rates dictionary
        tpr: True positive rates dictionary
        roc_auc: ROC AUC scores dictionary
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = Config.CLASS_NAMES
    
    plt.figure(figsize=Config.FIGURE_SIZE)
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    
    n_classes = len(class_names)
    for i, color in enumerate(colors[:n_classes]):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=Config.DPI)
        print(f"ROC curves saved to {save_path}")
    
    plt.close()


def visualize_optical_flow_sequence(flow_sequence, save_path=None):
    """
    Visualize a sequence of optical flow frames.
    
    Args:
        flow_sequence: Optical flow sequence (seq_len, H, W, 2)
        save_path: Path to save the visualization
    """
    seq_len = len(flow_sequence)
    cols = min(8, seq_len)
    rows = (seq_len + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if seq_len > 1 else [axes]
    
    for i, flow in enumerate(flow_sequence):
        # Convert optical flow to HSV for visualization
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(rgb)
        axes[i].set_title(f"Frame {i}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(seq_len, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=Config.DPI)
    
    plt.close()
