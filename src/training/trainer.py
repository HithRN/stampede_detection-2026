"""
Training module for stampede detection models
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from config.config import Config
from src.models.model_builder import create_enhanced_cnn_lstm_model, get_model_info
from src.utils.helpers import print_section_header, save_training_metadata
from src.utils.visualization import plot_training_history


def log_hyperparameters_to_json(model, batch_size, epochs, learning_rate,
                                  optimizer_name, scheduler_info=None,
                                  weight_init=None, log_file='hyperparameters.json'):
    """
    Log all training hyperparameters to a JSON file at training start.
    
    Args:
        model: Keras model instance
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate value
        optimizer_name: Name of optimizer
        scheduler_info: Dictionary with scheduler details (optional)
        weight_init: Weight initialization method (optional)
        log_file: Path to save JSON log file
    """
    def convert_to_serializable(obj):
        """Convert numpy/tensorflow types to JSON serializable types"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Get model info
    model_info = get_model_info(model)
    
    # Extract optimizer details from model
    optimizer_config = model.optimizer.get_config()
    optimizer_config = convert_to_serializable(optimizer_config)
    
    # Prepare hyperparameters dictionary
    hyperparameters = {
        "training_configuration": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "epochs": int(epochs)
        },
        "optimizer": {
            "name": optimizer_name,
            "type": model.optimizer.__class__.__name__,
            "config": optimizer_config
        },
        "model_architecture": {
            "total_params": model_info['total_params'],
            "trainable_params": model_info['trainable_params'],
            "non_trainable_params": model_info['non_trainable_params'],
            "input_shapes": {
                "flow_input": str(model.input[0].shape),
                "scalar_input": str(model.input[1].shape)
            },
            "output_shape": str(model.output.shape)
        },
        "loss_function": str(model.loss),
        "metrics": model_info['metrics']
    }
    
    # Add scheduler information if provided
    if scheduler_info:
        hyperparameters["learning_rate_scheduler"] = convert_to_serializable(scheduler_info)
    else:
        hyperparameters["learning_rate_scheduler"] = "None"
    
    # Add weight initialization if provided
    if weight_init:
        hyperparameters["weight_initialization"] = str(weight_init)
    else:
        # Try to extract from model layers
        init_methods = []
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                init_methods.append({
                    "layer": layer.name,
                    "initializer": layer.kernel_initializer.__class__.__name__
                })
        hyperparameters["weight_initialization"] = init_methods if init_methods else "default (GlorotUniform)"
    
    # Save to JSON file
    with open(log_file, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    
    print_section_header("HYPERPARAMETERS LOGGED")
    print(f"Saved to: {log_file}")
    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Total Parameters: {model_info['total_params']:,}")
    print("=" * 70 + "\n")
    
    return hyperparameters


def create_output_folders(base_dir='paper_figures'):
    """
    Create structured folder hierarchy for saving visualizations.
    
    Returns:
        dict: Dictionary with paths to all output folders
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    folders = {
        'base': base_dir,
        'optical_flow': os.path.join(base_dir, 'optical_flow_visualizations'),
        'predictions': os.path.join(base_dir, 'prediction_overlays'),
        'correct': os.path.join(base_dir, 'correct_predictions'),
        'incorrect': os.path.join(base_dir, 'incorrect_predictions'),
        'comparison': os.path.join(base_dir, 'correct_vs_incorrect'),
        'timestamp': timestamp
    }
    
    # Create all directories
    for folder_name, folder_path in folders.items():
        if folder_name != 'timestamp':
            os.makedirs(folder_path, exist_ok=True)
    
    print_section_header("OUTPUT FOLDERS CREATED")
    print(f"Base directory: {base_dir}")
    for name, path in folders.items():
        if name not in ['base', 'timestamp']:
            print(f"  • {name}: {path}")
    print("=" * 70 + "\n")
    
    return folders


def train_enhanced_model_with_visualizations(X_flow, X_scalar, y, original_frames=None,
                                             model_path=None, continue_training=False):
    """
    Enhanced training function that generates visualizations after training.
    
    Args:
        X_flow: Optical flow sequences
        X_scalar: Scalar features
        y: Labels
        original_frames: Original video frames (optional)
        model_path: Path to save/load model
        continue_training: Whether to continue from checkpoint
        
    Returns:
        model: Trained model
        history: Training history
        last_epoch: Last completed epoch
    """
    print_section_header("ENHANCED MODEL TRAINING WITH VISUALIZATIONS")
    
    # Sanity check for data
    print("\nData Statistics:")
    print(f"X_flow - min: {np.min(X_flow):.4f}, max: {np.max(X_flow):.4f}, "
          f"nan: {np.isnan(X_flow).any()}, inf: {np.isinf(X_flow).any()}")
    print(f"X_scalar - min: {np.min(X_scalar):.4f}, max: {np.max(X_scalar):.4f}, "
          f"nan: {np.isnan(X_scalar).any()}, inf: {np.isinf(X_scalar).any()}")
    
    # Convert labels to one-hot encoding
    y_onehot = to_categorical(y, num_classes=Config.NUM_CLASSES)
    
    # Split data into training and validation sets
    split_result = train_test_split(
        X_flow, X_scalar, y_onehot, y,
        test_size=Config.VALIDATION_SPLIT,
        random_state=42,
        stratify=y
    )
    
    X_flow_train, X_flow_val = split_result[0], split_result[1]
    X_scalar_train, X_scalar_val = split_result[2], split_result[3]
    y_train, y_val_onehot = split_result[4], split_result[5]
    y_train_labels, y_val_labels = split_result[6], split_result[7]
    
    # Also split original frames if provided
    if original_frames is not None and len(original_frames) > 0:
        _, original_frames_val = train_test_split(
            original_frames, test_size=Config.VALIDATION_SPLIT,
            random_state=42, stratify=y
        )
    else:
        original_frames_val = None
    
    # Create enhanced model
    model = create_enhanced_cnn_lstm_model()
    model.summary()
    
    # Log hyperparameters
    hyperparams = log_hyperparameters_to_json(
        model=model,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        optimizer_name="Adam",
        log_file='hyperparameters.json'
    )
    
    # Setup callbacks
    checkpoint_path = model_path if model_path else Config.MODEL_CHECKPOINT_PATH
    
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Training logs setup
    training_logs = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Create CSV log file
    with open('training_logs.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
        )
        writer.writeheader()
    
    # Train model
    print_section_header("TRAINING STARTED")
    history = model.fit(
        [X_flow_train, X_scalar_train],
        y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=([X_flow_val, X_scalar_val], y_val_onehot),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Extract and save logs
    for epoch in range(len(history.history['loss'])):
        training_logs['epoch'].append(epoch + 1)
        training_logs['train_loss'].append(float(history.history['loss'][epoch]))
        training_logs['train_accuracy'].append(float(history.history['accuracy'][epoch]))
        training_logs['val_loss'].append(float(history.history['val_loss'][epoch]))
        training_logs['val_accuracy'].append(float(history.history['val_accuracy'][epoch]))
    
    # Append to CSV
    with open('training_logs.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
        )
        for i in range(len(training_logs['epoch'])):
            writer.writerow({
                'epoch': training_logs['epoch'][i],
                'train_loss': training_logs['train_loss'][i],
                'train_accuracy': training_logs['train_accuracy'][i],
                'val_loss': training_logs['val_loss'][i],
                'val_accuracy': training_logs['val_accuracy'][i]
            })
    
    # Save training plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_logs['epoch'], training_logs['train_loss'], 'b-',
             label='Train Loss', linewidth=2)
    plt.plot(training_logs['epoch'], training_logs['val_loss'], 'r-',
             label='Validation Loss', linewidth=2)
    plt.title('Loss vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_logs['epoch'], training_logs['train_accuracy'], 'b-',
             label='Train Accuracy', linewidth=2)
    plt.plot(training_logs['epoch'], training_logs['val_accuracy'], 'r-',
             label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'training_metrics_plots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate visualizations
    print_section_header("GENERATING VISUALIZATIONS")
    folders = create_output_folders('paper_figures')
    
    if original_frames_val is not None:
        from src.training.visualization_generator import save_prediction_visualizations
        save_prediction_visualizations(
            model, X_flow_val, X_scalar_val, y_val_labels,
            original_frames_val, folders, num_samples=20
        )
    
    # Save final model
    final_model_path = 'enhanced_stampede_detection_final.h5'
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save metadata
    save_training_metadata(Config.EPOCHS)
    
    print_section_header("TRAINING COMPLETE")
    
    return model, history, Config.EPOCHS
