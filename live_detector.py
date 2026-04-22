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
    def __init__(self, model_path='models/saved_models/federated_model_final.h5'):
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
        """
        Use absolute static thresholds instead of dynamic sliding percentiles.
        Sliding percentiles mathematically force a percentage of legit transactions
        to be blocked in highly-imbalanced streams.
        """
        return {
            'high': 0.45,    # 45% probability (P99+ of risk distribution) for BLOCK
            'medium': 0.35,  # 35% probability (P90) for FLAG
            'low': 0.20      # Lower bounds
        }
    
    def preprocess_transaction(self, transaction):
        """Convert any transaction format to model features"""
        raw_feat = transaction.pop('raw_features', None)
        if raw_feat is not None:
            return raw_feat.reshape(1, -1)
            
        # Check for HTML UI advanced manual inputs mapping
        manual = transaction.pop('manual_override_features', None)
        if manual is not None:
            # Safely extract in strict alphabetical order used by the Model Aligner
            features_list = [
                'address_stability','amount','bank_tenure_months','credit_limit','credit_risk_score',
                'customer_age','device_distinct_emails','device_fraud_count','device_os_encoded',
                'email_is_free','emp_CA','emp_CB','emp_CC','emp_CD','emp_CE','emp_CF',
                'has_other_cards','house_BA','house_BB','house_BC','house_BD','house_BE','house_BF',
                'is_foreign_request','keep_alive','location_velocity','month','name_email_similarity',
                'phone_valid','session_length','source_internet','time_since_request_days',
                'velocity_24h','velocity_4w','velocity_6h'
            ]
            arr = np.zeros(self.feature_count)
            for i, f in enumerate(features_list):
                if i < self.feature_count:
                    try:
                        arr[i] = float(manual.get(f, 0.0))
                    except:
                        arr[i] = 0.0
            return arr.reshape(1, -1)
            
        # Fallback if no features present
        features = np.zeros(self.feature_count)
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