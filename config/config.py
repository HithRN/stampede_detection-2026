"""
Configuration file for Stampede Detection System
"""

import os

class Config:
    """Configuration class for stampede detection"""
    
    # ============= DATA CONFIGURATION =============
    DATA_PATH = r"C:\MMM\stamp_detect\Stampede_detection_dataset\Stampede_detection_dataset"
    TEST_VIDEO_PATH = r"C:\MMM\stamp_detect\Untitled video - Made with Clipchamp (2).mp4"
    
    # Class names for crowd density
    CLASS_NAMES = ["normal", "moderate", "dense", "risky"]
    NUM_CLASSES = len(CLASS_NAMES)
    
    # ============= MODEL CONFIGURATION =============
    # Sequence parameters
    SEQUENCE_LENGTH = 16
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    FLOW_CHANNELS = 2  # Optical flow has 2 channels (x, y)
    SCALAR_FEATURES = 4  # Number of additional features
    
    # Model architecture
    LSTM_UNITS = 256
    DENSE_UNITS = 128
    DROPOUT_RATE = 0.3
    
    # ============= TRAINING CONFIGURATION =============
    BATCH_SIZE = 8
    EPOCHS = 1
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 10
    MODEL_CHECKPOINT_PATH = 'enhanced_stampede_detection_checkpoint.h5'
    TRAINING_METADATA_FILE = 'training_metadata.json'
    
    # ============= INFERENCE CONFIGURATION =============
    TEMP_FLOW_DIR = "temp_optical_flow_test"
    INFERENCE_BATCH_SIZE = 8
    
    # ============= OUTPUT CONFIGURATION =============
    OUTPUT_DIR = 'outputs'
    FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    
    # ============= GPU CONFIGURATION =============
    USE_GPU = True
    GPU_MEMORY_GROWTH = True
    
    # ============= VISUALIZATION =============
    DPI = 100
    FIGURE_SIZE = (10, 8)
    
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.FIGURES_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.TEMP_FLOW_DIR, exist_ok=True)
    
    @staticmethod
    def get_config_dict():
        """Return configuration as a dictionary"""
        return {
            'data': {
                'class_names': Config.CLASS_NAMES,
                'num_classes': Config.NUM_CLASSES,
                'sequence_length': Config.SEQUENCE_LENGTH,
                'img_height': Config.IMG_HEIGHT,
                'img_width': Config.IMG_WIDTH,
            },
            'training': {
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'learning_rate': Config.LEARNING_RATE,
            }
        }
