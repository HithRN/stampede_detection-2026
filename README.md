# Stampede Detection System

A deep learning-based system for detecting crowd density and stampede risk using optical flow and CNN-LSTM architecture with enhanced temporal features.

## Features

- **Dual-Input CNN-LSTM Architecture**: Combines optical flow and scalar features
- **Enhanced Feature Extraction**: Flow acceleration, divergence, scene changes, motion entropy
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, classification reports
- **Performance Monitoring**: Real-time FPS tracking and inference metrics
- **Modular Design**: Clean separation of concerns for easy maintenance
- **Visualization Pipeline**: Automatic generation of training and evaluation plots
- **GPU Support**: Optimized for GPU acceleration with memory management

## 📁 Project Structure

```
stampede_detection/
├── config/
│   └── config.py                           # Configuration settings
├── src/
│   ├── data/
│   │   ├── data_loader.py                  # Data loading & preprocessing
│   │   └── __init__.py
│   ├── models/
│   │   ├── model_builder.py                # CNN-LSTM architectures
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py                      # Training logic
│   │   ├── visualization_generator.py      # Training visualizations
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── evaluator.py                    # Comprehensive evaluation
│   │   └── __init__.py
│   ├── inference/
│   │   ├── predictor.py                    # Video prediction
│   │   └── __init__.py
│   └── utils/
│       ├── helpers.py                      # Utility functions
│       ├── visualization.py                # Plotting utilities
│       ├── testing.py                      # Testing utilities
│       └── __init__.py
├── outputs/
│   ├── figures/                            # Generated plots
│   └── results/                            # Results and logs
├── main.py                                 # Main entry point
├── requirements.txt                        # Dependencies
└── README.md                               # This file
```

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/HithRN/stampede_detection-2026
cd stampede_detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure paths**
Edit `config/config.py` and set:
- `DATA_PATH`: Path to your training dataset
- `TEST_VIDEO_PATH`: Path to test video

### Basic Usage

**Train and evaluate model:**
```bash
python main.py
```

**Test with pre-trained model:**
```bash
python main.py --test-only --model-path model.h5 --video-path video.mp4
```

**Evaluate existing model:**
```bash
python main.py --eval-only --model-path model.h5
```

##  Detailed Usage

### Command Line Arguments

#### Mode Selection
- `--test-only`: Test video with pre-trained model (skip training)
- `--eval-only`: Evaluate model on validation data (skip training)

#### Paths
- `--data-path PATH`: Path to training dataset
- `--model-path PATH`: Path to model checkpoint
- `--video-path PATH`: Path to test video

#### Training Options
- `--continue-training`: Continue from existing checkpoint
- `--epochs N`: Number of training epochs (default: 50)
- `--batch-size N`: Batch size (default: 16)

#### Feature Options
- `--use-enhanced`: Use enhanced model with scalar features (default: True)
- `--skip-visualization`: Skip generating visualizations
- `--skip-feature-viz`: Skip feature importance plots

#### Utilities
- `--run-sanity-checks`: Run data and model validation

### Usage Examples

**1. Full training pipeline with custom epochs:**
```bash
python main.py --epochs 100 --batch-size 32
```

**2. Continue training from checkpoint:**
```bash
python main.py --continue-training --model-path checkpoint.h5 --epochs 50
```

**3. Test multiple videos:**
```bash
python main.py --test-only --model-path model.h5 --video-path video1.mp4
python main.py --test-only --model-path model.h5 --video-path video2.mp4
```

**4. Train with custom dataset:**
```bash
python main.py --data-path /path/to/dataset --epochs 75
```

**5. Quick sanity check:**
```bash
python main.py --run-sanity-checks --epochs 1
```

##  Dataset Structure

Expected directory structure:
```
Stampede_detection_dataset/
├── normal/
│   ├── video1/
│   │   ├── frame_0000.npy    # Optical flow (H, W, 2)
│   │   ├── frame_0000.jpg    # Original frame
│   │   ├── frame_0001.npy
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── video2/
│       └── ...
├── moderate/
│   └── ...
├── dense/
│   └── ...
└── risky/
    └── ...
```

**File Formats:**
- `.npy`: Optical flow numpy arrays (height, width, 2) - x and y components
- `.jpg`: Original video frames for SSIM calculation

##  Model Architecture

### Enhanced CNN-LSTM Model

```
Input Branch 1: Optical Flow (16, 64, 64, 2)
    ↓
TimeDistributed CNN (3 Conv blocks)
    ├── Conv2D(32) → BatchNorm → MaxPool
    ├── Conv2D(64) → BatchNorm → MaxPool
    └── Conv2D(128) → BatchNorm → MaxPool
    ↓
TimeDistributed Flatten
    ↓
Input Branch 2: Scalar Features (16, 4)
    ↓
Concatenate
    ↓
LSTM(256) → Dropout(0.3)
    ↓
LSTM(128) → Dropout(0.3)
    ↓
Dense(64) → BatchNorm
    ↓
Dense(4, softmax) → Output
```

### Additional Features (Scalar Branch)

1. **Flow Acceleration**: Temporal change in motion magnitude
2. **Flow Divergence**: Spatial spread of motion vectors
3. **Scene Changes**: SSIM-based frame differences
4. **Motion Entropy**: Distribution of motion directions

##  Output Files

### Training Outputs
```
outputs/figures/
├── training_metrics_plots.png       # Loss and accuracy curves
├── confusion_matrix_validation.png  # Validation confusion matrix
├── roc_auc_multiclass.png          # ROC curves for all classes
├── classification_report.txt        # Detailed metrics
├── feature_importance.png           # Feature visualization
└── paper_figures/                   # Additional visualizations
    ├── optical_flow_visualizations/
    ├── correct_predictions/
    ├── incorrect_predictions/
    └── correct_vs_incorrect/

outputs/results/
├── y_val_pred_proba.npy            # Validation probabilities
├── y_val_pred.npy                  # Validation predictions
├── y_val_true.npy                  # True labels
└── evaluation_summary.txt          # Summary report

Root directory:
├── enhanced_stampede_detection_checkpoint.h5  # Best model
├── enhanced_stampede_detection_final.h5       # Final model
├── hyperparameters.json                       # Training config
├── training_logs.csv                          # Epoch-by-epoch logs
└── inference_performance_log_*.txt            # Inference metrics
```

##  Classification Categories

| Category | Description | Risk Level |
|----------|-------------|------------|
| **Normal** | Low crowd density, safe conditions | ✅ Safe |
| **Moderate** | Medium density, manageable | ⚠️ Monitor |
| **Dense** | High density, requires attention | ⚠️ Caution |
| **Risky** | Very high density, stampede risk | 🚨 Danger |

##  Performance Metrics

The system provides:
- **Training**: Loss, accuracy, validation metrics
- **Evaluation**: Precision, recall, F1-score, ROC-AUC
- **Inference**: FPS, time per frame, time per sequence
- **GPU**: Device utilization, memory usage

##  Configuration

Key configuration in `config/config.py`:

```python
# Data Configuration
SEQUENCE_LENGTH = 16        # Frames per sequence
IMG_HEIGHT = 64            # Image height
IMG_WIDTH = 64             # Image width

# Training Configuration
BATCH_SIZE = 16            # Training batch size
EPOCHS = 50                # Training epochs
LEARNING_RATE = 0.0001     # Learning rate
VALIDATION_SPLIT = 0.2     # Validation split ratio

# Model Configuration
LSTM_UNITS = 256           # LSTM hidden units
DROPOUT_RATE = 0.5         # Dropout rate
```

##  Troubleshooting

**Out of Memory (OOM) Error:**
- Reduce `BATCH_SIZE` in `config/config.py`
- Enable GPU memory growth (already configured)
- Reduce `IMG_HEIGHT` and `IMG_WIDTH`

**No data loaded:**
- Check dataset path in `config/config.py`
- Verify directory structure matches expected format
- Ensure `.npy` and `.jpg` files exist

**Slow inference:**
- Use `--batch-size` to increase batch size for inference
- Ensure GPU is being used (check console output)
- Reduce video resolution or frame rate

##  Requirements

- Python 3.7+
- TensorFlow 2.8+
- OpenCV 4.5+
- NumPy 1.21+
- Scikit-learn 0.24+
- Matplotlib 3.4+
- Seaborn 0.11+
- Pandas 1.3+

See `requirements.txt` for complete list.

##  Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

##  License

[Add your license here]

##  Authors

Geetanjali Bhola, Dr. Sumit Srivastava, Hith Rahil Nidhan,
Sanyam Kathed

##  Acknowledgments

- Original optical flow implementation
- TensorFlow and Keras teams
- Open-source community



---

**Note**: This is a modular implementation designed for easy maintenance, extension, and research.
