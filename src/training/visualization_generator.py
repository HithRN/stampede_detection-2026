"""
Visualization generator for training predictions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from config.config import Config


def visualize_optical_flow(flow, save_path=None, title="Optical Flow"):
    """
    Create a colorful HSV visualization of optical flow.
    
    Args:
        flow: Optical flow array (H, W, 2)
        save_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        hsv_rgb: RGB image of the flow visualization
    """
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude
    
    # Convert to RGB for display/saving
    hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    if save_path:
        cv2.imwrite(save_path, hsv_rgb)
    
    return hsv_rgb


def create_flow_grid_visualization(flow_sequence, save_path, max_frames=8):
    """
    Create a grid visualization of optical flow sequence.
    
    Args:
        flow_sequence: Sequence of flow frames (T, H, W, 2)
        save_path: Path to save the visualization
        max_frames: Maximum number of frames to display
    """
    num_frames = min(len(flow_sequence), max_frames)
    cols = 4
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for idx in range(num_frames):
        flow = flow_sequence[idx]
        flow_vis = visualize_optical_flow(flow)
        flow_vis_rgb = cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(flow_vis_rgb)
        axes[idx].set_title(f'Frame {idx+1}', fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved optical flow grid: {save_path}")


def create_prediction_overlay(original_frame, prediction_probs, true_label=None,
                              save_path=None, class_names=None):
    """
    Create an overlay showing prediction probabilities on the original frame.
    
    Args:
        original_frame: Original video frame
        prediction_probs: Prediction probabilities
        true_label: True label (optional)
        save_path: Path to save the visualization
        class_names: List of class names
        
    Returns:
        overlay_frame: Frame with prediction overlay
    """
    if class_names is None:
        class_names = Config.CLASS_NAMES
    
    # Create a copy of the frame
    overlay_frame = original_frame.copy()
    h, w = overlay_frame.shape[:2]
    
    # Handle NaN values in predictions
    if np.any(np.isnan(prediction_probs)):
        print(f"Warning: NaN detected in predictions: {prediction_probs}")
        prediction_probs = np.nan_to_num(prediction_probs, nan=0.0)
        prob_sum = np.sum(prediction_probs)
        if prob_sum > 0:
            prediction_probs = prediction_probs / prob_sum
        else:
            prediction_probs = np.ones(len(class_names)) / len(class_names)
    
    # Get predicted class
    pred_class = np.argmax(prediction_probs)
    pred_confidence = prediction_probs[pred_class]
    
    # Define colors for each class (BGR format)
    class_colors = {
        0: (0, 255, 0),      # Normal - Green
        1: (0, 255, 255),    # Moderate - Yellow
        2: (0, 165, 255),    # Dense - Orange
        3: (0, 0, 255)       # Risky - Red
    }
    
    # Draw prediction box
    box_height = 150
    box_y_start = h - box_height - 10
    cv2.rectangle(overlay_frame, (10, box_y_start), (w - 10, h - 10), (0, 0, 0), -1)
    overlay = overlay_frame.copy()
    cv2.rectangle(overlay, (10, box_y_start), (w - 10, h - 10), (0, 0, 0), -1)
    overlay_frame = cv2.addWeighted(overlay, 0.6, overlay_frame, 0.4, 0)
    
    # Add prediction text
    y_offset = box_y_start + 30
    pred_text = f"Prediction: {class_names[pred_class].upper()}"
    conf_text = f"Confidence: {pred_confidence*100:.1f}%"
    
    cv2.putText(overlay_frame, pred_text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, class_colors[pred_class], 2)
    cv2.putText(overlay_frame, conf_text, (20, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add probability bars
    bar_y = y_offset + 60
    bar_width = w - 100
    bar_height = 15
    
    for i, (class_name, prob) in enumerate(zip(class_names, prediction_probs)):
        # Handle NaN in individual probabilities
        if np.isnan(prob) or np.isinf(prob):
            prob = 0.0
        prob = np.clip(prob, 0.0, 1.0)
        
        # Class label
        cv2.putText(overlay_frame, class_name[:4].upper(), (20, bar_y + i*25 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Probability bar background
        cv2.rectangle(overlay_frame, (100, bar_y + i*25),
                     (100 + bar_width, bar_y + i*25 + bar_height), (50, 50, 50), -1)
        
        # Probability bar fill
        fill_width = int(bar_width * prob)
        cv2.rectangle(overlay_frame, (100, bar_y + i*25),
                     (100 + fill_width, bar_y + i*25 + bar_height),
                     class_colors[i], -1)
        
        # Percentage text
        cv2.putText(overlay_frame, f"{prob*100:.1f}%",
                   (105 + bar_width, bar_y + i*25 + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add true label if provided
    if true_label is not None:
        true_text = f"True: {class_names[true_label].upper()}"
        color = (0, 255, 0) if true_label == pred_class else (0, 0, 255)
        cv2.putText(overlay_frame, true_text, (w - 250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if save_path:
        cv2.imwrite(save_path, overlay_frame)
    
    return overlay_frame


def create_comparison_figure(correct_indices, incorrect_indices, y_val, y_pred, y_pred_probs,
                               original_frames_val, folders, class_names, num_pairs=4):
    """
    Create side-by-side comparison of correct vs incorrect predictions.
    
    Args:
        correct_indices: Indices of correct predictions
        incorrect_indices: Indices of incorrect predictions
        y_val: True labels
        y_pred: Predicted labels
        y_pred_probs: Prediction probabilities
        original_frames_val: Original frames
        folders: Output folders dictionary
        class_names: List of class names
        num_pairs: Number of pairs to visualize
    """
    fig, axes = plt.subplots(num_pairs, 2, figsize=(12, 4*num_pairs))
    
    for i in range(num_pairs):
        # Correct prediction
        if i < len(correct_indices):
            correct_idx = correct_indices[i]
            if len(original_frames_val[correct_idx]) > 0:
                frame_correct = original_frames_val[correct_idx][len(original_frames_val[correct_idx])//2]
                frame_with_overlay = create_prediction_overlay(
                    frame_correct, y_pred_probs[correct_idx], y_val[correct_idx],
                    None, class_names
                )
                axes[i, 0].imshow(cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB))
                axes[i, 0].set_title(f'CORRECT: {class_names[y_val[correct_idx]]}',
                                     color='green', fontweight='bold')
                axes[i, 0].axis('off')
        
        # Incorrect prediction
        if i < len(incorrect_indices):
            incorrect_idx = incorrect_indices[i]
            if len(original_frames_val[incorrect_idx]) > 0:
                frame_incorrect = original_frames_val[incorrect_idx][len(original_frames_val[incorrect_idx])//2]
                frame_with_overlay = create_prediction_overlay(
                    frame_incorrect, y_pred_probs[incorrect_idx], y_val[incorrect_idx],
                    None, class_names
                )
                axes[i, 1].imshow(cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB))
                axes[i, 1].set_title(
                    f'INCORRECT: True={class_names[y_val[incorrect_idx]]}, Pred={class_names[y_pred[incorrect_idx]]}',
                    color='red', fontweight='bold'
                )
                axes[i, 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(folders['comparison'], 'correct_vs_incorrect_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison figure: {save_path}")


def save_prediction_visualizations(model, X_flow_val, X_scalar_val, y_val,
                                     original_frames_val, folders,
                                     num_samples=10, class_names=None):
    """
    Generate and save visualizations for correct and incorrect predictions.
    
    Args:
        model: Trained model
        X_flow_val: Validation optical flow data
        X_scalar_val: Validation scalar features
        y_val: Validation labels (integer format)
        original_frames_val: Original frames
        folders: Dictionary of output folder paths
        num_samples: Number of samples to visualize
        class_names: List of class names
    """
    if class_names is None:
        class_names = Config.CLASS_NAMES
    
    print("\n" + "=" * 70)
    print("GENERATING PREDICTION VISUALIZATIONS".center(70))
    print("=" * 70)
    
    # Get predictions
    y_pred_probs = model.predict([X_flow_val, X_scalar_val], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Find correct and incorrect predictions
    correct_mask = (y_pred == y_val)
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    print(f"\nCorrect predictions: {len(correct_indices)}/{len(y_val)}")
    print(f"Incorrect predictions: {len(incorrect_indices)}/{len(y_val)}")
    
    # Save optical flow visualizations
    print(f"\nSaving optical flow visualizations...")
    for idx in range(min(num_samples, len(X_flow_val))):
        flow_seq = X_flow_val[idx]
        save_path = os.path.join(folders['optical_flow'],
                                 f'flow_sequence_{idx:03d}_class_{class_names[y_val[idx]]}.png')
        create_flow_grid_visualization(flow_seq, save_path)
    
    # Save correct predictions
    print(f"\nSaving correct prediction overlays...")
    samples_per_class = num_samples // len(class_names)
    for class_id in range(len(class_names)):
        class_correct = correct_indices[y_val[correct_indices] == class_id]
        for i, idx in enumerate(class_correct[:samples_per_class]):
            if len(original_frames_val[idx]) > 0:
                mid_frame = original_frames_val[idx][len(original_frames_val[idx])//2]
                save_path = os.path.join(folders['correct'],
                                         f'correct_{class_names[class_id]}_{i:02d}.png')
                create_prediction_overlay(mid_frame, y_pred_probs[idx], y_val[idx],
                                          save_path, class_names)
    
    # Save incorrect predictions
    print(f"\nSaving incorrect prediction overlays...")
    for i, idx in enumerate(incorrect_indices[:num_samples]):
        if len(original_frames_val[idx]) > 0:
            mid_frame = original_frames_val[idx][len(original_frames_val[idx])//2]
            save_path = os.path.join(folders['incorrect'],
                                     f'incorrect_{i:02d}_true_{class_names[y_val[idx]]}_pred_{class_names[y_pred[idx]]}.png')
            create_prediction_overlay(mid_frame, y_pred_probs[idx], y_val[idx],
                                      save_path, class_names)
    
    # Create comparison figure
    print(f"\nCreating correct vs incorrect comparison...")
    create_comparison_figure(correct_indices, incorrect_indices, y_val, y_pred, y_pred_probs,
                              original_frames_val, folders, class_names)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION GENERATION COMPLETE".center(70))
    print("=" * 70 + "\n")
