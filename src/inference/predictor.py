"""
Inference and prediction module for stampede detection
"""

import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

from config.config import Config
from src.data.data_loader import preprocess_video_for_inference, calculate_scalar_features
from src.utils.helpers import print_section_header, print_subsection_header, sanitize_predictions


def predict_with_enhanced_model(model, video_path, temp_dir=None, timeout_seconds=300):
    """
    Predict crowd density for a video using the enhanced model with performance monitoring.
    
    Args:
        model: Trained model
        video_path: Path to input video
        temp_dir: Temporary directory for processing
        timeout_seconds: Maximum processing time
        
    Returns:
        category: Predicted category
        confidence: Prediction confidence
        perf_metrics: Performance metrics dictionary
    """
    if temp_dir is None:
        temp_dir = Config.TEMP_FLOW_DIR
    
    # Initialize performance metrics
    perf_metrics = {
        'device': 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU',
        'batch_size': Config.INFERENCE_BATCH_SIZE,
        'total_frames_processed': 0,
        'total_sequences': 0,
        'total_inference_time': 0.0,
        'avg_time_per_sequence': 0.0,
        'avg_time_per_frame': 0.0,
        'fps': 0.0,
        'frame_processing_times': [],
        'sequence_inference_times': []
    }
    
    print_section_header("INFERENCE PERFORMANCE MONITORING")
    print(f"Device: {perf_metrics['device']}")
    print(f"Batch Size: {perf_metrics['batch_size']}")
    print("=" * 70 + "\n")
    
    # Test model inference speed
    print("Testing model inference speed...")
    dummy_flow_input = np.zeros(
        (1, Config.SEQUENCE_LENGTH, Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.FLOW_CHANNELS),
        dtype=np.float32
    )
    dummy_scalar_input = np.zeros((1, Config.SEQUENCE_LENGTH, Config.SCALAR_FEATURES), dtype=np.float32)
    
    start = time.time()
    dummy_pred = model.predict([dummy_flow_input, dummy_scalar_input], verbose=0)
    end = time.time()
    
    print(f"Model inference test: {end - start:.3f} seconds for a single sequence")
    print(f"Prediction shape: {dummy_pred.shape}, values: min={dummy_pred.min():.3f}, max={dummy_pred.max():.3f}\n")
    
    start_time = time.time()
    
    def check_timeout():
        if time.time() - start_time > timeout_seconds:
            print(f"Processing timed out after {timeout_seconds} seconds!")
            return True
        return False
    
    # Process video
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Processing video: {video_path}")
    
    frame_start = time.time()
    try:
        flow_sequences, frame_sequences = preprocess_video_for_inference(video_path, temp_dir)
    except Exception as e:
        print(f"Error processing video: {e}")
        return "Error", 0.0, perf_metrics
    frame_end = time.time()
    
    perf_metrics['total_frames_processed'] = len(flow_sequences) * Config.SEQUENCE_LENGTH if flow_sequences else 0
    perf_metrics['frame_processing_times'].append(frame_end - frame_start)
    
    if check_timeout():
        return "Timeout", 0.0, perf_metrics
    
    if len(flow_sequences) == 0:
        print("Warning: No sequences could be created from video")
        return "Unknown", 0.0, perf_metrics
    
    perf_metrics['total_sequences'] = len(flow_sequences)
    
    print(f"Created {len(flow_sequences)} sequences from video")
    
    # Calculate scalar features
    print("Calculating additional features for prediction...")
    feature_start = time.time()
    
    scalar_features_list = []
    for flow_seq, frame_seq in zip(flow_sequences, frame_sequences):
        scalar_feat = calculate_scalar_features(flow_seq, frame_seq)
        scalar_features_list.append(scalar_feat)
    
    scalar_features = np.array(scalar_features_list)
    feature_end = time.time()
    
    print(f"Feature calculation took {feature_end - feature_start:.3f} seconds")
    
    # Prepare data for prediction
    X_flow_pred = np.array(flow_sequences)
    X_scalar_pred = scalar_features
    
    # Make predictions in batches
    batch_size = perf_metrics['batch_size']
    all_predictions = []
    
    print(f"Making predictions in batches of {batch_size}...")
    
    total_inference_start = time.time()
    
    num_batches = (len(X_flow_pred) + batch_size - 1) // batch_size
    for i in range(0, len(X_flow_pred), batch_size):
        batch_flow = X_flow_pred[i:i + batch_size]
        batch_scalar = X_scalar_pred[i:i + batch_size]
        
        batch_start = time.time()
        batch_preds = model.predict([batch_flow, batch_scalar], verbose=0)
        batch_end = time.time()
        
        perf_metrics['sequence_inference_times'].append(batch_end - batch_start)
        all_predictions.append(batch_preds)
        
        print(f"Processed batch {i // batch_size + 1}/{num_batches} "
              f"(Time: {batch_end - batch_start:.3f}s)")
        
        if check_timeout():
            return "Timeout", 0.0, perf_metrics
    
    total_inference_end = time.time()
    perf_metrics['total_inference_time'] = total_inference_end - total_inference_start
    
    # Aggregate predictions
    predictions = np.vstack(all_predictions) if len(all_predictions) > 1 else all_predictions[0]
    
    # Sanitize predictions
    predictions = sanitize_predictions(predictions)
    
    avg_prediction = np.mean(predictions, axis=0)
    class_idx = np.argmax(avg_prediction)
    
    category = Config.CLASS_NAMES[class_idx]
    confidence = avg_prediction[class_idx]
    
    # Calculate performance metrics
    if perf_metrics['total_sequences'] > 0:
        perf_metrics['avg_time_per_sequence'] = perf_metrics['total_inference_time'] / perf_metrics['total_sequences']
    
    if perf_metrics['total_frames_processed'] > 0:
        total_time = time.time() - start_time
        perf_metrics['avg_time_per_frame'] = total_time / perf_metrics['total_frames_processed']
        perf_metrics['fps'] = perf_metrics['total_frames_processed'] / total_time
    
    # Print performance summary
    print_section_header("INFERENCE PERFORMANCE SUMMARY")
    print(f"Device Used: {perf_metrics['device']}")
    print(f"Batch Size: {perf_metrics['batch_size']}")
    print(f"Total Frames Processed: {perf_metrics['total_frames_processed']}")
    print(f"Total Sequences Created: {perf_metrics['total_sequences']}")
    print(f"\nTiming Metrics:")
    print(f"  Total Inference Time: {perf_metrics['total_inference_time']:.4f} seconds")
    print(f"  Average Time per Sequence ({Config.SEQUENCE_LENGTH} frames): {perf_metrics['avg_time_per_sequence']:.4f} seconds")
    print(f"  Average Time per Frame: {perf_metrics['avg_time_per_frame']:.4f} seconds")
    print(f"  FPS (Frames Per Second): {perf_metrics['fps']:.2f}")
    print("=" * 70 + "\n")
    
    # Save performance log
    log_filename = f"inference_performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w') as log_file:
        log_file.write("=" * 70 + "\n")
        log_file.write("INFERENCE PERFORMANCE LOG\n")
        log_file.write("=" * 70 + "\n")
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"\nPrediction Results:\n")
        log_file.write(f"  Category: {category}\n")
        log_file.write(f"  Confidence: {confidence * 100:.2f}%\n")
        log_file.write(f"\nDevice Configuration:\n")
        log_file.write(f"  Device: {perf_metrics['device']}\n")
        log_file.write(f"  Batch Size: {perf_metrics['batch_size']}\n")
        log_file.write(f"\nProcessing Statistics:\n")
        log_file.write(f"  Total Frames Processed: {perf_metrics['total_frames_processed']}\n")
        log_file.write(f"  Total Sequences Created: {perf_metrics['total_sequences']}\n")
        log_file.write(f"\nTiming Metrics:\n")
        log_file.write(f"  Total Inference Time: {perf_metrics['total_inference_time']:.4f} seconds\n")
        log_file.write(f"  Average Time per Sequence ({Config.SEQUENCE_LENGTH} frames): {perf_metrics['avg_time_per_sequence']:.4f} seconds\n")
        log_file.write(f"  Average Time per Frame: {perf_metrics['avg_time_per_frame']:.4f} seconds\n")
        log_file.write(f"  FPS (Frames Per Second): {perf_metrics['fps']:.2f}\n")
        log_file.write("=" * 70 + "\n")
    
    print(f"Performance metrics saved to: {log_filename}")
    
    # Print detailed prediction results
    print("\nDetailed prediction results:")
    for i, cat in enumerate(Config.CLASS_NAMES):
        print(f"{cat}: {avg_prediction[i] * 100:.2f}%")
    
    return category, confidence, perf_metrics


def predict_single_sequence(model, flow_sequence, frame_sequence):
    """
    Predict crowd density for a single sequence.
    
    Args:
        model: Trained model
        flow_sequence: Optical flow sequence
        frame_sequence: Original frame sequence
        
    Returns:
        prediction: Prediction probabilities
        category: Predicted category
        confidence: Prediction confidence
    """
    # Calculate scalar features
    scalar_features = calculate_scalar_features(flow_sequence, frame_sequence)
    
    # Prepare data
    X_flow = np.expand_dims(flow_sequence, axis=0)
    X_scalar = np.expand_dims(scalar_features, axis=0)
    
    # Make prediction
    prediction = model.predict([X_flow, X_scalar], verbose=0)
    prediction = sanitize_predictions(prediction)
    
    class_idx = np.argmax(prediction[0])
    category = Config.CLASS_NAMES[class_idx]
    confidence = prediction[0][class_idx]
    
    return prediction[0], category, confidence


def batch_predict_videos(model, video_paths, output_dir=None):
    """
    Predict crowd density for multiple videos.
    
    Args:
        model: Trained model
        video_paths: List of video paths
        output_dir: Directory to save results
        
    Returns:
        results: List of prediction results
    """
    if output_dir is None:
        output_dir = Config.RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print_section_header(f"BATCH PREDICTION - {len(video_paths)} VIDEOS")
    
    for idx, video_path in enumerate(video_paths, 1):
        print(f"\nProcessing video {idx}/{len(video_paths)}: {video_path}")
        
        try:
            category, confidence, perf_metrics = predict_with_enhanced_model(model, video_path)
            
            result = {
                'video_path': video_path,
                'category': category,
                'confidence': float(confidence),
                'performance': perf_metrics
            }
            
            results.append(result)
            
            print(f"Result: {category} ({confidence * 100:.2f}% confidence)")
            
        except Exception as e:
            print(f"Error processing video: {e}")
            results.append({
                'video_path': video_path,
                'category': 'Error',
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Save results
    import json
    results_path = os.path.join(output_dir, f'batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print_section_header("BATCH PREDICTION COMPLETE")
    print(f"Results saved to: {results_path}")
    
    return results


def load_model_for_inference(model_path):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    from tensorflow.keras.models import load_model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully")
    
    return model
