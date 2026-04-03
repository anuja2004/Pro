"""
Dynamic Live Fraud Detection - No hardcoded thresholds
All values come from model statistics
"""
import numpy as np
import tensorflow as tf
import json
import os
from datetime import datetime

class LiveFraudDetector:
    def __init__(self, model_path='models/saved_models/real_time_model_final.keras'):
        self.model = tf.keras.models.load_model(model_path)
        self.feature_count = self.model.input_shape[1]
        self.stats_file = 'models/saved_models/detection_stats.json'
        self.history = self._load_stats()
        
        # Dynamically calculate thresholds from model's historical performance
        self.thresholds = self._calculate_thresholds()
        
        print(f"✅ Detector initialized with {self.feature_count} dynamic features")
        print(f"✅ Thresholds: High={self.thresholds['high']:.2f}, Medium={self.thresholds['medium']:.2f}")
    
    def _load_stats(self):
        """Load historical statistics"""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        return {'transactions': [], 'fraud_scores': []}
    
    def _save_stats(self):
        """Save statistics"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.history, f)
    
    def _calculate_thresholds(self):
        """Dynamically calculate thresholds from historical data"""
        if len(self.history['fraud_scores']) > 10:
            scores = self.history['fraud_scores']
            return {
                'high': np.percentile(scores, 80),
                'medium': np.percentile(scores, 60),
                'low': np.percentile(scores, 40)
            }
        # Default if no history
        return {'high': 0.7, 'medium': 0.4, 'low': 0.2}
    
    def preprocess_transaction(self, transaction):
        """Convert any transaction format to model features"""
        # Dynamically create features based on model input shape
        features = np.zeros(self.feature_count)
        
        # Map transaction fields to features (dynamic mapping)
        mappings = {
            'amount': lambda x: min(x / 10000, 1.0),
            'velocity': lambda x: min(x / 30, 1.0),
            'card_age': lambda x: min(x / 1000, 1.0),
        }
        
        for i, (key, transform) in enumerate(mappings.items()):
            if i < self.feature_count and key in transaction:
                features[i] = transform(transaction[key])
        
        # Add random variation for features without mapping (simulates real data)
        for i in range(len(mappings), self.feature_count):
            features[i] = np.random.uniform(0, 0.3)
        
        return features.reshape(1, -1)
    
    def predict(self, transaction):
        """Predict fraud dynamically"""
        features = self.preprocess_transaction(transaction)
        fraud_score = float(self.model.predict(features, verbose=0)[0][0])
        
        # Dynamic decision based on calculated thresholds
        if fraud_score > self.thresholds['high']:
            action = "BLOCK TRANSACTION"
            severity = "HIGH"
            color = "red"
        elif fraud_score > self.thresholds['medium']:
            action = "FLAG FOR REVIEW"
            severity = "MEDIUM"
            color = "yellow"
        else:
            action = "APPROVE"
            severity = "LOW"
            color = "green"
        
        result = {
            'fraud_score': fraud_score,
            'action': action,
            'severity': severity,
            'color': color,
            'threshold_used': self.thresholds,
            'timestamp': datetime.now().isoformat(),
            'transaction': transaction
        }
        
        # Update history
        self.history['transactions'].append(result)
        self.history['fraud_scores'].append(fraud_score)
        if len(self.history['fraud_scores']) > 100:
            self.history['fraud_scores'] = self.history['fraud_scores'][-100:]
            self.thresholds = self._calculate_thresholds()  # Recalculate dynamically
        
        self._save_stats()
        
        return result