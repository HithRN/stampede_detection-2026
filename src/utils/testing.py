"""
Testing utilities for stampede detection system
"""

import os
import numpy as np
from tensorflow.keras.models import load_model

from config.config import Config
from src.inference.predictor import predict_with_enhanced_model, load_model_for_inference
from src.utils.helpers import print_section_header


def test_video_only(model_path, video_path):
    """
    Test a video with a pre-trained model (test-only mode).
    
    Args:
        model_path: Path to trained model
        video_path: Path to test video
    """
    print_section_header("TEST-ONLY MODE")
    
    # Load model
    model = load_model_for_inference(model_path)
    
    # Make prediction
    category, confidence, perf_metrics = predict_with_enhanced_model(model, video_path)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Video Classification Result: {category.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 50)
    
    return category, confidence, perf_metrics


def test_enhanced_model(model_path, video_path):
    """
    Test enhanced model with additional features.
    
    Args:
        model_path: Path to trained model
        video_path: Path to test video
    """
    return test_video_only(model_path, video_path)


def validate_model_outputs(model, X_flow_sample, X_scalar_sample):
    """
    Validate that model produces valid outputs.
    
    Args:
        model: Trained model
        X_flow_sample: Sample optical flow data
        X_scalar_sample: Sample scalar features
        
    Returns:
        bool: True if outputs are valid
    """
    try:
        # Make prediction
        predictions = model.predict([X_flow_sample, X_scalar_sample], verbose=0)
        
        # Check shape
        expected_shape = (X_flow_sample.shape[0], Config.NUM_CLASSES)
        if predictions.shape != expected_shape:
            print(f"Error: Unexpected prediction shape {predictions.shape}, expected {expected_shape}")
            return False
        
        # Check for NaN/Inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            print("Warning: Model produced NaN or Inf values")
            return False
        
        # Check probability range
        if np.any(predictions < 0) or np.any(predictions > 1):
            print("Warning: Predictions outside [0, 1] range")
            return False
        
        # Check sum to 1
        prob_sums = np.sum(predictions, axis=1)
        if not np.allclose(prob_sums, 1.0, atol=1e-5):
            print("Warning: Predictions do not sum to 1")
            return False
        
        print("Model outputs are valid")
        return True
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return False


def run_sanity_checks(model, X_flow, X_scalar, y):
    """
    Run sanity checks on data and model.
    
    Args:
        model: Trained model
        X_flow: Optical flow data
        X_scalar: Scalar features
        y: Labels
    """
    print_section_header("SANITY CHECKS")
    
    # Data checks
    print("\n1. Data Statistics:")
    print(f"   X_flow - Shape: {X_flow.shape}")
    print(f"            Min: {np.min(X_flow):.4f}, Max: {np.max(X_flow):.4f}")
    print(f"            NaN: {np.isnan(X_flow).any()}, Inf: {np.isinf(X_flow).any()}")
    
    print(f"   X_scalar - Shape: {X_scalar.shape}")
    print(f"              Min: {np.min(X_scalar):.4f}, Max: {np.max(X_scalar):.4f}")
    print(f"              NaN: {np.isnan(X_scalar).any()}, Inf: {np.isinf(X_scalar).any()}")
    
    print(f"   Labels - Shape: {y.shape}")
    print(f"            Unique values: {np.unique(y)}")
    print(f"            Distribution: {np.bincount(y)}")
    
    # Model checks
    print("\n2. Model Validation:")
    sample_size = min(5, len(X_flow))
    X_flow_sample = X_flow[:sample_size]
    X_scalar_sample = X_scalar[:sample_size]
    
    is_valid = validate_model_outputs(model, X_flow_sample, X_scalar_sample)
    
    if is_valid:
        print("   ✓ All checks passed")
    else:
        print("   ✗ Some checks failed")
    
    print("\n" + "=" * 70)


def test_data_pipeline(data_path=None):
    """
    Test the data loading pipeline.
    
    Args:
        data_path: Path to dataset
    """
    if data_path is None:
        data_path = Config.DATA_PATH
    
    print_section_header("TESTING DATA PIPELINE")
    
    from src.data.data_loader import load_optical_flow_data
    
    try:
        print(f"Loading data from: {data_path}")
        X_flow, X_scalar, y, original_frames, sequence_ids = load_optical_flow_data(data_path)
        
        print("\nData loading successful!")
        print(f"Loaded {len(X_flow)} sequences")
        print(f"Flow shape: {X_flow.shape}")
        print(f"Scalar shape: {X_scalar.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Original frames shape: {original_frames.shape}")
        print(f"Number of sequence IDs: {len(sequence_ids)}")
        
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False


def benchmark_inference_speed(model, num_sequences=100):
    """
    Benchmark model inference speed.
    
    Args:
        model: Trained model
        num_sequences: Number of sequences to test
    """
    import time
    
    print_section_header("INFERENCE SPEED BENCHMARK")
    
    # Create dummy data
    X_flow = np.random.randn(
        num_sequences,
        Config.SEQUENCE_LENGTH,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
        Config.FLOW_CHANNELS
    ).astype(np.float32)
    
    X_scalar = np.random.randn(
        num_sequences,
        Config.SEQUENCE_LENGTH,
        Config.SCALAR_FEATURES
    ).astype(np.float32)
    
    # Warm-up
    _ = model.predict([X_flow[:1], X_scalar[:1]], verbose=0)
    
    # Benchmark
    start = time.time()
    predictions = model.predict([X_flow, X_scalar], verbose=0)
    end = time.time()
    
    total_time = end - start
    time_per_sequence = total_time / num_sequences
    fps = (num_sequences * Config.SEQUENCE_LENGTH) / total_time
    
    print(f"\nBenchmark Results:")
    print(f"  Total sequences: {num_sequences}")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Time per sequence: {time_per_sequence:.4f} seconds")
    print(f"  FPS: {fps:.2f}")
    print("\n" + "=" * 70)
    
    return {
        'total_time': total_time,
        'time_per_sequence': time_per_sequence,
        'fps': fps
    }
