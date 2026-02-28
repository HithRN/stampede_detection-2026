"""
Model architecture definitions for stampede detection
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense,
    Dropout, LSTM, TimeDistributed, Input, Concatenate
)
from tensorflow.keras.optimizers import Adam
import numpy as np

from config.config import Config


def create_enhanced_cnn_lstm_model():
    """
    Create an enhanced CNN-LSTM model that incorporates additional motion features.
    
    Architecture:
    - Optical Flow Branch: CNN for spatial feature extraction (TimeDistributed)
    - Scalar Features Branch: Direct input of calculated features
    - Combined: Concatenated features fed into LSTM
    - Classification: Dense layers with dropout
    
    Returns:
        Compiled Keras Model
    """
    # Input shapes
    flow_input_shape = (Config.SEQUENCE_LENGTH, Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.FLOW_CHANNELS)
    scalar_features_shape = (Config.SEQUENCE_LENGTH, Config.SCALAR_FEATURES)
    
    # Optical flow input branch
    flow_input = Input(shape=flow_input_shape, name='optical_flow_input')
    
    # CNN feature extraction with TimeDistributed to process each frame
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(flow_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    # Flatten CNN output for each time step
    x = TimeDistributed(Flatten())(x)
    
    # Scalar features input branch
    scalar_input = Input(shape=scalar_features_shape, name='scalar_features_input')
    
    # Concatenate CNN features with scalar features
    combined = Concatenate(axis=2)([x, scalar_input])
    
    # LSTM to capture temporal patterns
    lstm1 = LSTM(Config.LSTM_UNITS, return_sequences=True)(combined)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(Config.DENSE_UNITS)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)
    
    # Final classification layers
    dense1 = Dense(64, activation='relu')(dropout2)
    bn = BatchNormalization()(dense1)
    output = Dense(Config.NUM_CLASSES, activation='softmax')(bn)
    
    # Create model with multiple inputs
    model = Model(inputs=[flow_input, scalar_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_basic_cnn_lstm_model():
    """
    Create a basic CNN-LSTM model without additional features.
    (Kept for backward compatibility)
    
    Returns:
        Compiled Keras Model
    """
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'),
                       input_shape=(Config.SEQUENCE_LENGTH, Config.IMG_HEIGHT, 
                                  Config.IMG_WIDTH, Config.FLOW_CHANNELS)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        
        TimeDistributed(Flatten()),
        
        LSTM(256, return_sequences=True),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def log_model_architecture(model, filepath='model_architecture.txt'):
    """
    Save model architecture summary to a file.
    
    Args:
        model: Keras model
        filepath: Path to save the architecture summary
    """
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Model architecture saved to {filepath}")


def get_model_info(model):
    """
    Get detailed model information.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model information
    """
    total_params = int(model.count_params())
    trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    non_trainable_params = int(sum([tf.size(w).numpy() for w in model.non_trainable_weights]))
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'optimizer': model.optimizer.__class__.__name__,
        'loss': str(model.loss),
        'metrics': model.metrics_names if hasattr(model, 'metrics_names') else ['accuracy']
    }
    
    return info
