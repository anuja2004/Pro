"""
Neural network model for fraud detection - No TensorFlow Federated dependency
"""
import tensorflow as tf

def create_fraud_model(feature_count, config):
    """
    Create Keras model for fraud detection
    
    Args:
        feature_count: Number of input features
        config: Configuration dictionary with model parameters
        
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(feature_count,)),
        
        # First hidden layer - larger for learning complex patterns
        tf.keras.layers.Dense(
            config['model']['layer1_units'], 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config['model']['l2_lambda']),
            name='dense_1'
        ),
        tf.keras.layers.BatchNormalization(name='batch_norm_1'),
        tf.keras.layers.Dropout(config['model']['dropout1'], name='dropout_1'),
        
        # Second hidden layer
        tf.keras.layers.Dense(
            config['model']['layer2_units'], 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config['model']['l2_lambda']),
            name='dense_2'
        ),
        tf.keras.layers.BatchNormalization(name='batch_norm_2'),
        tf.keras.layers.Dropout(config['model']['dropout2'], name='dropout_2'),
        
        # Third hidden layer
        tf.keras.layers.Dense(
            config['model']['layer3_units'], 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config['model']['l2_lambda']),
            name='dense_3'
        ),
        tf.keras.layers.Dropout(config['model']['dropout3'], name='dropout_3'),
        
        # Output layer for binary classification
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile model with class weights for imbalanced data
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['training']['client_learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn')
        ]
    )
    
    return model


def create_lightweight_model(feature_count, config):
    """
    Create a smaller, faster model for quick testing
    
    Args:
        feature_count: Number of input features
        config: Configuration dictionary
        
    Returns:
        Compiled lightweight Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(feature_count,)),
        tf.keras.layers.Dense(32, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.2, name='dropout_1'),
        tf.keras.layers.Dense(16, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.1, name='dropout_2'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def create_ensemble_model(feature_count, config):
    """
    Create an ensemble of models for more robust predictions
    
    Args:
        feature_count: Number of input features
        config: Configuration dictionary
        
    Returns:
        Ensemble model
    """
    # This creates 3 different models that will be averaged
    input_layer = tf.keras.layers.Input(shape=(feature_count,))
    
    # Model 1: Wider architecture
    x1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    x1 = tf.keras.layers.Dropout(0.3)(x1)
    x1 = tf.keras.layers.Dense(32, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.2)(x1)
    x1 = tf.keras.layers.Dense(1, activation='sigmoid')(x1)
    
    # Model 2: Deeper architecture
    x2 = tf.keras.layers.Dense(48, activation='relu')(input_layer)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dropout(0.25)(x2)
    x2 = tf.keras.layers.Dense(24, activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dropout(0.15)(x2)
    x2 = tf.keras.layers.Dense(12, activation='relu')(x2)
    x2 = tf.keras.layers.Dense(1, activation='sigmoid')(x2)
    
    # Model 3: Simple architecture
    x3 = tf.keras.layers.Dense(32, activation='relu')(input_layer)
    x3 = tf.keras.layers.Dropout(0.2)(x3)
    x3 = tf.keras.layers.Dense(16, activation='relu')(x3)
    x3 = tf.keras.layers.Dense(1, activation='sigmoid')(x3)
    
    # Average the outputs
    averaged = tf.keras.layers.Average()([x1, x2, x3])
    
    model = tf.keras.Model(inputs=input_layer, outputs=averaged)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model


def get_model_summary(model):
    """
    Get a string summary of the model architecture
    
    Args:
        model: Keras model
        
    Returns:
        String summary
    """
    import io
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
    return string_buffer.getvalue()


def save_model_architecture(model, filepath):
    """
    Save model architecture visualization
    
    Args:
        model: Keras model
        filepath: Path to save the plot
    """
    try:
        tf.keras.utils.plot_model(
            model, 
            to_file=filepath, 
            show_shapes=True, 
            show_dtype=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=96
        )
        print(f"Model architecture saved to {filepath}")
    except ImportError:
        print("Graphviz not installed. Skipping model visualization.")
    except Exception as e:
        print(f"Could not save model architecture: {e}")


# Default model configuration
DEFAULT_MODEL_CONFIG = {
    'layer1_units': 64,
    'layer2_units': 32,
    'layer3_units': 16,
    'dropout1': 0.3,
    'dropout2': 0.2,
    'dropout3': 0.1,
    'l2_lambda': 0.001
}


def create_model_from_config(feature_count, model_config=None):
    """
    Create model from configuration dictionary
    
    Args:
        feature_count: Number of input features
        model_config: Dictionary with model parameters
        
    Returns:
        Compiled Keras model
    """
    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG
    
    # Wrap in the expected structure
    config = {
        'model': model_config,
        'training': {
            'client_learning_rate': 0.001
        }
    }
    
    return create_fraud_model(feature_count, config)


# For backward compatibility
create_tff_model = None  # Explicitly set to None to avoid TFF import errors

print("✅ Model module loaded successfully (No TFF dependency)")