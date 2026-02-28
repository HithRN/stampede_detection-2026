"""
Helper utility functions for stampede detection
"""

import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

from config.config import Config


def configure_gpu():
    """Configure GPU settings for TensorFlow"""
    if Config.USE_GPU:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    if Config.GPU_MEMORY_GROWTH:
                        tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s)")
                return True
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                return False
    return False


def save_training_metadata(last_epoch, filepath=None):
    """
    Save training metadata to track progress.
    
    Args:
        last_epoch: Last completed epoch
        filepath: Path to metadata file
    """
    if filepath is None:
        filepath = Config.TRAINING_METADATA_FILE
    
    metadata = {
        'last_epoch': last_epoch,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Training metadata saved to {filepath}")


def load_training_metadata(filepath=None):
    """
    Load training metadata.
    
    Args:
        filepath: Path to metadata file
        
    Returns:
        Dictionary with metadata or None
    """
    if filepath is None:
        filepath = Config.TRAINING_METADATA_FILE
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded training metadata from {filepath}")
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def ensure_directories():
    """Ensure all necessary directories exist"""
    Config.ensure_directories()


def safe_divide(numerator, denominator, default=0.0):
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division fails
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def sanitize_predictions(predictions):
    """
    Sanitize prediction probabilities to handle NaN/Inf values.
    
    Args:
        predictions: Model prediction probabilities
        
    Returns:
        Sanitized predictions
    """
    # Replace NaN/Inf with safe values
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure each sample sums to 1
    prob_sum = np.sum(predictions, axis=1, keepdims=True)
    prob_sum = np.clip(prob_sum, 1e-8, None)  # Avoid division by zero
    predictions = predictions / prob_sum
    
    return predictions


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_log_filename(prefix="log"):
    """Create a timestamped log filename"""
    timestamp = get_timestamp()
    return f"{prefix}_{timestamp}.txt"


def print_section_header(title, width=70):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def print_subsection_header(title, width=70):
    """Print a formatted subsection header"""
    print("\n" + "-" * width)
    print(f"{title:^{width}}")
    print("-" * width)


def format_time(seconds):
    """
    Format seconds into a readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Labels array
        
    Returns:
        Dictionary of class weights
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    class_weights = {}
    for cls, count in zip(unique, counts):
        weight = total / (len(unique) * count)
        class_weights[int(cls)] = weight
    
    print("\nClass weights calculated:")
    for cls, weight in class_weights.items():
        print(f"  Class {cls} ({Config.CLASS_NAMES[cls]}): {weight:.4f}")
    
    return class_weights
