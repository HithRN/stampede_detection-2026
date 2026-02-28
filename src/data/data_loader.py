"""
Data loading and preprocessing module for stampede detection
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

from config.config import Config


def load_optical_flow_data(data_path):
    """
    Load optical flow data from directory structure with JPG images.
    The images contain optical flow encoded in RGB channels:
    - R channel = flow X direction (normalized to -1 to 1)
    - G channel = flow Y direction (normalized to -1 to 1)
    
    Args:
        data_path: Root path to dataset
        
    Returns:
        X_flow: Optical flow sequences (N, seq_len, H, W, 2)
        X_scalar: Scalar features (N, seq_len, 4)
        y: Labels (N,)
        original_frames: Original frames (N, seq_len, H, W, 3)
        sequence_video_ids: Video IDs for each sequence
    """
    X = []  # Will contain sequences of optical flow
    original_frames = []  # Will contain original frames for SSIM calculation
    y = []  # Will contain labels
    sequence_video_ids = []

    print(f"Loading data from: {data_path}")
    
    categories = Config.CLASS_NAMES

    for category_idx, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        print(f"Processing category: {category}")
        
        if not os.path.exists(category_path):
            print(f"Warning: Path {category_path} does not exist")
            continue

        # Check if there are image files directly in this folder
        frames = sorted([f for f in os.listdir(category_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))])

        if frames:
            # If we have image files directly in the category folder
            print(f"  Found {len(frames)} frames directly in {category} folder")

            # Create sequences of SEQUENCE_LENGTH frames with 50% overlap
            for i in range(0, max(1, len(frames) - Config.SEQUENCE_LENGTH + 1), Config.SEQUENCE_LENGTH // 2):
                sequence = []
                orig_sequence = []
                
                for j in range(i, i + Config.SEQUENCE_LENGTH):
                    if j < len(frames):
                        frame_path = os.path.join(category_path, frames[j])

                        # Read the image
                        img = cv2.imread(frame_path)
                        if img is None:
                            print(f"    Warning: Could not read image {frame_path}")
                            continue

                        # Store original resized frame for SSIM calculation
                        orig_img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
                        orig_sequence.append(orig_img)

                        # Extract optical flow from BGR image channels
                        # R channel = flow X, G channel = flow Y
                        flow = np.zeros((orig_img.shape[0], orig_img.shape[1], 2), dtype=np.float32)
                        flow[..., 0] = orig_img[..., 2] / 255.0 * 2 - 1  # X direction from R channel
                        flow[..., 1] = orig_img[..., 1] / 255.0 * 2 - 1  # Y direction from G channel

                        sequence.append(flow)

                if len(sequence) == Config.SEQUENCE_LENGTH:
                    X.append(np.array(sequence))
                    original_frames.append(np.array(orig_sequence))
                    y.append(category_idx)
                    sequence_video_ids.append(f"{category}_direct_{i}")
        else:
            # Check for subdirectories (like "test 1", "test 2")
            subfolders = [f for f in os.listdir(category_path) 
                         if os.path.isdir(os.path.join(category_path, f))]
            print(f"  Found {len(subfolders)} subfolders in {category} category")

            for subfolder in subfolders:
                subfolder_path = os.path.join(category_path, subfolder)
                frames = sorted([f for f in os.listdir(subfolder_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))])
                print(f"    {subfolder}: Found {len(frames)} frames")

                # Create sequences with frames from this subfolder with 50% overlap
                for i in range(0, max(1, len(frames) - Config.SEQUENCE_LENGTH + 1), Config.SEQUENCE_LENGTH // 2):
                    sequence = []
                    orig_sequence = []
                    
                    for j in range(i, i + Config.SEQUENCE_LENGTH):
                        if j < len(frames):
                            frame_path = os.path.join(subfolder_path, frames[j])

                            img = cv2.imread(frame_path)
                            if img is None:
                                print(f"      Warning: Could not read image {frame_path}")
                                continue

                            # Store original resized frame for SSIM calculation
                            orig_img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
                            orig_sequence.append(orig_img)

                            # Extract optical flow from BGR image channels
                            flow = np.zeros((orig_img.shape[0], orig_img.shape[1], 2), dtype=np.float32)
                            flow[..., 0] = orig_img[..., 2] / 255.0 * 2 - 1  # X from R channel
                            flow[..., 1] = orig_img[..., 1] / 255.0 * 2 - 1  # Y from G channel

                            sequence.append(flow)

                    if len(sequence) == Config.SEQUENCE_LENGTH:
                        X.append(np.array(sequence))
                        original_frames.append(np.array(orig_sequence))
                        y.append(category_idx)
                        sequence_video_ids.append(f"{category}_{subfolder}_{i}")

    X = np.array(X) if X else np.array([])
    original_frames = np.array(original_frames) if original_frames else np.array([])
    y = np.array(y) if y else np.array([])

    print(f"\nLoaded {len(X)} sequences with shape {X[0].shape if len(X) > 0 else 'N/A'}")
    print(f"Loaded {len(y)} labels")

    # Calculate additional features
    scalar_features = None
    if len(X) > 0:
        print("Calculating additional features...")

        # Flow acceleration features
        flow_acceleration = calculate_flow_acceleration(X)
        print(f"  Flow acceleration shape: {flow_acceleration.shape}")

        # Flow divergence features
        flow_divergence = calculate_flow_divergence(X)
        print(f"  Flow divergence shape: {flow_divergence.shape}")

        # Scene change detection using SSIM
        scene_changes = calculate_scene_changes(original_frames)
        print(f"  Scene changes shape: {scene_changes.shape}")

        # Motion entropy
        motion_entropy = calculate_motion_entropy(X)
        print(f"  Motion entropy shape: {motion_entropy.shape}")

        # Combine all scalar features
        scalar_features = np.stack([
            flow_acceleration,
            flow_divergence,
            scene_changes,
            motion_entropy
        ], axis=2)  # Shape: [num_sequences, sequence_length, 4]

        # Clean up NaN/Inf values
        scalar_features = np.nan_to_num(scalar_features, nan=0.0, posinf=1e3, neginf=-1e3)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        print(f"\nDataset loaded successfully!")
        print(f"Total sequences: {len(X)}")
        print(f"Optical flow shape: {X.shape}")
        print(f"Scalar features shape: {scalar_features.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Original frames shape: {original_frames.shape}")
    else:
        scalar_features = np.array([])
        print(f"Optical flow shape: {X.shape}")
        print(f"Scalar features shape: {scalar_features.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Original frames shape: {original_frames.shape}")

    return X, scalar_features, y, original_frames, sequence_video_ids



def calculate_scalar_features(flow_sequence, frame_sequence):
    """
    Calculate additional scalar features from optical flow and frames for a SINGLE sequence.
    This is a wrapper function used during inference.
    
    Features:
    1. Flow Acceleration: Change in magnitude over time
    2. Flow Divergence: Spatial spread of motion vectors
    3. Scene Changes: Frame-to-frame difference (SSIM)
    4. Motion Entropy: Distribution of motion directions
    
    Args:
        flow_sequence: Single optical flow sequence (seq_len, H, W, 2)
        frame_sequence: Single frame sequence (seq_len, H, W, 3)
        
    Returns:
        features: Array of shape (sequence_length, 4)
    """
    # Convert single sequence to batch format (add batch dimension)
    flow_batch = np.expand_dims(flow_sequence, axis=0)
    frame_batch = np.expand_dims(frame_sequence, axis=0)
    
    # Calculate each feature using the batch functions
    acceleration = calculate_flow_acceleration(flow_batch)[0]  # [0] to remove batch dim
    divergence = calculate_flow_divergence(flow_batch)[0]
    scene_change = calculate_scene_changes(frame_batch)[0]
    motion_ent = calculate_motion_entropy(flow_batch)[0]
    
    # Stack into feature array: shape (sequence_length, 4)
    features = np.stack([acceleration, divergence, scene_change, motion_ent], axis=1)
    
    return features



def calculate_flow_acceleration(flow_sequences):
    """
    Calculate frame-wise differences in optical flow (acceleration).
    """
    accelerations = []

    for sequence in flow_sequences:
        # Calculate frame-by-frame differences in optical flow
        seq_acceleration = []
        for i in range(1, len(sequence)):
            # Calculate difference between consecutive frames
            accel = sequence[i] - sequence[i - 1]
            # Compute magnitude of acceleration
            accel_magnitude = np.sqrt(np.sum(accel ** 2, axis=2))
            # Average magnitude across the frame
            mean_accel = np.mean(accel_magnitude)
            seq_acceleration.append(mean_accel)

        # Pad the sequence with a zero at the beginning to maintain sequence length
        seq_acceleration.insert(0, 0)
        accelerations.append(np.array(seq_acceleration))

    return np.array(accelerations)


def calculate_flow_divergence(flow_sequences):
    """
    Calculate spatial divergence of optical flow.
    """
    divergences = []

    for sequence in flow_sequences:
        seq_divergence = []
        for flow in sequence:
            # Calculate spatial derivatives
            dx = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3)

            # Divergence is the sum of partial derivatives
            divergence = dx + dy

            # Get mean divergence as a scalar feature
            mean_divergence = np.mean(np.abs(divergence))
            seq_divergence.append(mean_divergence)

        divergences.append(np.array(seq_divergence))

    return np.array(divergences)


def calculate_scene_changes(original_frames):
    """
    Detect scene changes using structural similarity (SSIM).
    Returns a sequence of SSIM scores between consecutive frames.
    """
    ssim_scores = []

    for sequence in original_frames:
        seq_ssim = []
        for i in range(1, len(sequence)):
            # Convert to grayscale if not already
            if len(sequence[i].shape) == 3:
                frame1 = cv2.cvtColor(sequence[i - 1], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(sequence[i], cv2.COLOR_BGR2GRAY)
            else:
                frame1 = sequence[i - 1]
                frame2 = sequence[i]

            # Calculate SSIM between consecutive frames
            score = ssim(frame1, frame2, data_range=frame2.max() - frame2.min())

            # 1-score to get change metric (higher value means more change)
            change_score = 1.0 - score
            seq_ssim.append(change_score)

        # Pad with zero for first frame
        seq_ssim.insert(0, 0)
        ssim_scores.append(np.array(seq_ssim))

    return np.array(ssim_scores)


def calculate_motion_entropy(flow_sequences):
    """
    Calculate entropy of motion to measure chaos/randomness in the optical flow field.
    """
    entropies = []

    for sequence in flow_sequences:
        seq_entropy = []
        for flow in sequence:
            # Calculate magnitude and angle of optical flow
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Calculate entropy of magnitudes
            # First, create histogram
            hist, _ = np.histogram(mag, bins=32, range=(0, np.max(mag) if np.max(mag) > 0 else 1))
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            else:
                hist = np.ones_like(hist) / len(hist)
            flow_entropy = entropy(hist, base=2)
            if not np.isfinite(flow_entropy):
                flow_entropy = 0.0

            seq_entropy.append(flow_entropy)

        entropies.append(np.array(seq_entropy))

    return np.array(entropies)


def split_data(X_flow, X_scalar, y, original_frames=None, sequence_video_ids=None):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X_flow: Optical flow sequences
        X_scalar: Scalar features
        y: Labels
        original_frames: Original video frames (optional)
        sequence_video_ids: Sequence IDs (optional)
        
    Returns:
        Tuple of train, validation, and test sets
    """
    # First split: separate test set
    split_data_result = train_test_split(
        X_flow, X_scalar, y,
        test_size=Config.TEST_SPLIT,
        random_state=42,
        stratify=y
    )
    
    X_flow_temp, X_flow_test = split_data_result[0], split_data_result[1]
    X_scalar_temp, X_scalar_test = split_data_result[2], split_data_result[3]
    y_temp, y_test = split_data_result[4], split_data_result[5]
    
    # Second split: separate validation from training
    val_size = Config.VALIDATION_SPLIT / (1 - Config.TEST_SPLIT)
    
    split_data_result = train_test_split(
        X_flow_temp, X_scalar_temp, y_temp,
        test_size=val_size,
        random_state=42,
        stratify=y_temp
    )
    
    X_flow_train, X_flow_val = split_data_result[0], split_data_result[1]
    X_scalar_train, X_scalar_val = split_data_result[2], split_data_result[3]
    y_train, y_val = split_data_result[4], split_data_result[5]
    
    print(f"\nData split completed:")
    print(f"Training set: {len(X_flow_train)} sequences")
    print(f"Validation set: {len(X_flow_val)} sequences")
    print(f"Test set: {len(X_flow_test)} sequences")
    
    return (X_flow_train, X_scalar_train, y_train,
            X_flow_val, X_scalar_val, y_val,
            X_flow_test, X_scalar_test, y_test)


def preprocess_video_for_inference(video_path, output_dir, max_frames=200, frame_skip=5, max_sequences=20):
    """
    Process a video file and extract optical flow sequences.
    Restores the original temporal scaling, double-resize trick, and sequence batching.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save optical flow debug frames (optional)
        max_frames: Max frames to sample from the video
        frame_skip: How many frames to skip to capture macroscopic motion
        max_sequences: Maximum number of chunks to generate to prevent OOM
        
    Returns:
        flow_sequences: List of optical flow sequences
        original_frames: List of original frame sequences
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0  # Fallback
        
    # Calculate time intervals for frame sampling - aim for ~4 seconds of content
    target_duration = 4  # seconds
    target_frames = min(max_frames, int(target_duration * fps))

    # Calculate frame indices to capture macroscopic motion
    if total_frames <= target_frames:
        frame_indices = list(range(0, total_frames, frame_skip))
    else:
        # Pick frames evenly distributed across the video
        frame_indices = [int(i * total_frames / target_frames) for i in range(target_frames)]

    print(f"Video: {total_frames} frames at {fps} FPS. Using {len(frame_indices)} frames for analysis.")

    # Aggressive downsampling to smooth micro-movements (arms/legs) and capture macro crowd flow
    downsample_factor = 0.2  
    
    flow_frames = []
    original_frames = []
    prev_gray = None
    processed_count = 0
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # 1. Save original frame for SSIM scene change calculation
        resized_frame = cv2.resize(frame, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        original_frames.append(resized_frame)
        
        # 2. The Double-Resize Trick: Downsample heavily, then scale back up
        small_frame = cv2.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        
        if prev_gray is None:
            prev_gray = gray
            continue
            
        # 3. Custom Farneback tuned for the blurred inputs
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, 
            levels=2,      # Reduced from 3
            winsize=11,    # Reduced from 15
            iterations=2,  # Reduced from 3
            poly_n=5, 
            poly_sigma=1.2, 
            flags=0
        )
        
        flow_frames.append(flow)
        prev_gray = gray
        processed_count += 1
        
    cap.release()
    
    if len(flow_frames) < Config.SEQUENCE_LENGTH:
        print(f"Warning: Only extracted {len(flow_frames)} flow frames. Less than required {Config.SEQUENCE_LENGTH}.")
        return [], []
        
    # 4. Sequence Chunking (Prevent GPU memory crash and get an even average)
    flow_sequences = []
    frame_sequences = []
    
    # Calculate step size to grab a maximum of 20 evenly spaced sequences
    step_size = max(1, (len(flow_frames) - Config.SEQUENCE_LENGTH) // max_sequences)
    start_indices = list(range(0, len(flow_frames) - Config.SEQUENCE_LENGTH + 1, step_size))[:max_sequences]
    
    for i in start_indices:
        flow_seq = flow_frames[i:i + Config.SEQUENCE_LENGTH]
        # Align original frames to flow frames
        frame_seq = original_frames[i:i + Config.SEQUENCE_LENGTH]
        
        flow_sequences.append(np.array(flow_seq))
        frame_sequences.append(np.array(frame_seq))
        
    print(f"Extracted {len(flow_sequences)} highly-tuned sequences for prediction.")
    return flow_sequences, frame_sequences
