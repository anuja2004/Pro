"""
Dynamic Fraud Types - Loaded from trained model features
No hardcoding - reads actual model architecture
"""
import tensorflow as tf
import numpy as np

class FraudTypeDetector:
    def __init__(self, model_path='models/saved_models/real_time_model_final.keras'):
        """Dynamically load model and infer fraud types from features"""
        self.model = tf.keras.models.load_model(model_path)
        self.feature_count = self.model.input_shape[1]
        self._analyze_model()
    
    def _analyze_model(self):
        """Dynamically analyze model to understand what it learned"""
        # Get model weights to understand feature importance
        first_layer_weights = self.model.layers[0].get_weights()[0]
        
        # Calculate feature importance from actual model weights
        self.feature_importance = np.abs(first_layer_weights).mean(axis=1)
        self.important_features = np.argsort(self.feature_importance)[-10:][::-1]
        
        print(f"✅ Model loaded: {self.feature_count} features")
        print(f"✅ Feature importance computed from {len(self.important_features)} top features")
    
    def get_fraud_type(self, features, fraud_score):
        """Determine fraud type based on actual feature values"""
        alerts = []
        
        # Check each important feature's contribution
        for feat_idx in self.important_features[:5]:
            feat_value = features[0][feat_idx] if len(features.shape) > 1 else features[feat_idx]
            
            if feat_value > 0.7 and fraud_score > 0.5:
                alerts.append({
                    'type': f'Feature_{feat_idx}_anomaly',
                    'confidence': float(feat_value),
                    'contribution': float(self.feature_importance[feat_idx])
                })
        
        return alerts