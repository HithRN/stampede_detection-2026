"""
Main entry point for Stampede Detection System

This script orchestrates the complete workflow:
- Data loading and preprocessing
- Model training with visualizations
- Model evaluation
- Video prediction/inference
"""

import os
import sys
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from src.data.data_loader import load_optical_flow_data
from src.models.model_builder import create_enhanced_cnn_lstm_model
from src.training.trainer import train_enhanced_model_with_visualizations
from src.evaluation.evaluator import evaluate_model_comprehensive, generate_evaluation_summary_report
from src.inference.predictor import predict_with_enhanced_model, load_model_for_inference
from src.utils.helpers import (
    configure_gpu, ensure_directories, load_training_metadata,
    save_training_metadata, print_section_header, format_time
)
from src.utils.visualization import visualize_feature_importance
from src.utils.testing import test_video_only, test_enhanced_model, run_sanity_checks


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Stampede Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model and test on video
  python main.py
  
  # Test only with pre-trained model
  python main.py --test-only --model-path model.h5 --video-path video.mp4
  
  # Train with custom data path
  python main.py --data-path /path/to/dataset
  
  # Continue training from checkpoint
  python main.py --continue-training --model-path checkpoint.h5
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only test a video with a pre-trained model (skip training)'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate model on validation data (skip training)'
    )
    
    # Paths
    parser.add_argument(
        '--data-path',
        type=str,
        default=Config.DATA_PATH,
        help=f'Path to training dataset (default: {Config.DATA_PATH})'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=Config.MODEL_CHECKPOINT_PATH,
        help=f'Path to model checkpoint (default: {Config.MODEL_CHECKPOINT_PATH})'
    )
    
    parser.add_argument(
        '--video-path',
        type=str,
        default=Config.TEST_VIDEO_PATH,
        help=f'Path to test video (default: {Config.TEST_VIDEO_PATH})'
    )
    
    # Training options
    parser.add_argument(
        '--continue-training',
        action='store_true',
        default=False,
        help='Continue training from checkpoint if model exists'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.EPOCHS,
        help=f'Number of training epochs (default: {Config.EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=Config.BATCH_SIZE,
        help=f'Batch size for training (default: {Config.BATCH_SIZE})'
    )
    
    # Feature options
    parser.add_argument(
        '--use-enhanced',
        action='store_true',
        default=True,
        help='Use enhanced model with additional features (default: True)'
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip generating visualizations during training'
    )
    
    parser.add_argument(
        '--skip-feature-viz',
        action='store_true',
        help='Skip feature importance visualization'
    )
    
    # Evaluation options
    parser.add_argument(
        '--run-sanity-checks',
        action='store_true',
        help='Run sanity checks on data and model'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    print_section_header("STAMPEDE DETECTION SYSTEM")
    print("Enhanced CNN-LSTM Model with Optical Flow Analysis")
    print("=" * 70 + "\n")
    
    # Configure GPU
    gpu_available = configure_gpu()
    
    # Ensure output directories exist
    ensure_directories()
    
    # Override config if arguments provided
    if args.epochs != Config.EPOCHS:
        Config.EPOCHS = args.epochs
    if args.batch_size != Config.BATCH_SIZE:
        Config.BATCH_SIZE = args.batch_size
    
    # Start timer
    start_time = time.time()
    
    # ==================== TEST-ONLY MODE ====================
    if args.test_only:
        print_section_header("TEST-ONLY MODE")
        print(f"Model: {args.model_path}")
        print(f"Video: {args.video_path}")
        print("=" * 70 + "\n")
        
        try:
            if args.use_enhanced:
                category, confidence, perf_metrics = test_enhanced_model(
                    args.model_path, args.video_path
                )
            else:
                category, confidence, perf_metrics = test_video_only(
                    args.model_path, args.video_path
                )
            
            # Print final results
            print_section_header("FINAL RESULTS")
            print(f"Video Classification: {category.upper()}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print("=" * 70)
            
        except Exception as e:
            print(f"Error during testing: {e}")
            sys.exit(1)
        
        return
    
    # ==================== EVALUATION-ONLY MODE ====================
    if args.eval_only:
        print_section_header("EVALUATION-ONLY MODE")
        
        try:
            # Load model
            model = load_model_for_inference(args.model_path)
            
            # Load data
            print("Loading validation data...")
            X_flow, X_scalar, y, original_frames, _ = load_optical_flow_data(args.data_path)
            
            if len(X_flow) == 0:
                print("Error: No data loaded. Please check the data path.")
                sys.exit(1)
            
            # Run evaluation
            evaluation_results = evaluate_model_comprehensive(
                model, X_flow, X_scalar, y, original_frames
            )
            
            # Generate summary report
            generate_evaluation_summary_report(evaluation_results)
            
            print_section_header("EVALUATION COMPLETE")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            sys.exit(1)
        
        return
    
    # ==================== FULL TRAINING MODE ====================
    print_section_header("FULL TRAINING AND EVALUATION MODE")
    
    # Load dataset
    print("\n1. Loading Dataset")
    print("-" * 70)
    print(f"Data path: {args.data_path}")
    print("Loading optical flow data and calculating additional features...")
    
    try:
        X_flow, X_scalar, y, original_frames, sequence_video_ids = load_optical_flow_data(
            args.data_path
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    if len(X_flow) == 0:
        print("Error: No data loaded. Please check the data path and directory structure.")
        print(f"Expected structure: {args.data_path}/[normal|moderate|dense|risky]/video_folders/")
        sys.exit(1)
    
    print(f"✓ Data loaded successfully: {len(X_flow)} sequences")
    
    # Run sanity checks if requested
    if args.run_sanity_checks:
        # Create a temporary model for checks
        temp_model = create_enhanced_cnn_lstm_model()
        run_sanity_checks(temp_model, X_flow, X_scalar, y)
    
    # Check for existing training metadata
    initial_epoch = 0
    if args.continue_training:
        metadata = load_training_metadata()
        if metadata:
            initial_epoch = metadata.get('last_epoch', 0)
            print(f"\n✓ Resuming training from epoch {initial_epoch}")
    
    # Train model
    print("\n2. Training Model")
    print("-" * 70)
    
    try:
        model, history, last_epoch = train_enhanced_model_with_visualizations(
            X_flow, X_scalar, y,
            original_frames=original_frames,
            model_path=args.model_path,
            continue_training=args.continue_training
        )
        
        # Save training metadata
        save_training_metadata(last_epoch)
        
        print(f"\n✓ Training completed: {last_epoch} epochs")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate model
    print("\n3. Evaluating Model")
    print("-" * 70)
    
    try:
        # Split data for evaluation (use same split as training)
        from sklearn.model_selection import train_test_split
        
        _, X_flow_val, _, X_scalar_val, _, y_val = train_test_split(
            X_flow, X_scalar, y,
            test_size=Config.VALIDATION_SPLIT,
            random_state=42,
            stratify=y
        )
        
        # Run comprehensive evaluation
        evaluation_results = evaluate_model_comprehensive(
            model, X_flow_val, X_scalar_val, y_val,
            original_frames_val=None,
            config=Config.get_config_dict()
        )
        
        # Generate summary report
        generate_evaluation_summary_report(evaluation_results)
        
        print("\n✓ Evaluation completed")
        
    except Exception as e:
        print(f"Warning: Could not complete evaluation: {e}")
    
    # Test on video
    print("\n4. Testing on Video")
    print("-" * 70)
    print(f"Video path: {args.video_path}")
    
    if os.path.exists(args.video_path):
        try:
            category, confidence, perf_metrics = predict_with_enhanced_model(
                model, args.video_path, Config.TEMP_FLOW_DIR
            )
            
            # Print results
            print_section_header("VIDEO PREDICTION RESULTS")
            print(f"Classification: {category.upper()}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print("=" * 70)
            
        except Exception as e:
            print(f"Warning: Could not test video: {e}")
    else:
        print(f"Warning: Test video not found at {args.video_path}")
        print("Skipping video prediction.")
    
    # Feature importance visualization
    if not args.skip_feature_viz and args.use_enhanced:
        print("\n5. Feature Importance Visualization")
        print("-" * 70)
        
        try:
            num_samples = min(5, len(X_flow))
            visualize_feature_importance(
                X_flow[:num_samples],
                X_scalar[:num_samples],
                y[:num_samples],
                save_path=os.path.join(Config.FIGURES_DIR, 'feature_importance.png')
            )
            print("✓ Feature importance visualization saved")
            
        except Exception as e:
            print(f"Warning: Could not create feature visualization: {e}")
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print_section_header("EXECUTION SUMMARY")
    print(f"Total execution time: {format_time(total_time)}")
    print(f"Model saved to: {args.model_path}")
    print(f"Outputs saved to: {Config.OUTPUT_DIR}")
    print(f"Figures saved to: {Config.FIGURES_DIR}")
    print(f"Results saved to: {Config.RESULTS_DIR}")
    print("=" * 70)
    
    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
