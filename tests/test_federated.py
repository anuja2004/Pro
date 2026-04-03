"""
Unit tests for federated learning modules
"""
import unittest
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.federated.client import FederatedClient
from src.federated.model import create_fraud_model

class TestFederated(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        self.features = np.random.randn(n_samples, n_features)
        self.labels = np.random.randint(0, 2, n_samples)
        
        self.data = pd.DataFrame(
            np.column_stack([self.features, self.labels]),
            columns=[f'f{i}' for i in range(n_features)] + ['fraud_label']
        )
        
        self.feature_cols = [f'f{i}' for i in range(n_features)]
        
        self.config = {
            'model': {
                'layer1_units': 32,
                'layer2_units': 16,
                'layer3_units': 8,
                'dropout1': 0.3,
                'dropout2': 0.2,
                'dropout3': 0.1,
                'l2_lambda': 0.001
            },
            'training': {
                'batch_size': 16,
                'client_learning_rate': 0.001
            }
        }
    
    def test_federated_client(self):
        """Test FederatedClient initialization"""
        client = FederatedClient(
            client_id='test_client',
            data=self.data,
            feature_columns=self.feature_cols
        )
        
        self.assertEqual(client.client_id, 'test_client')
        self.assertEqual(client.features.shape, (100, 10))
        
        # Test dataset creation
        dataset = client.get_tf_dataset(batch_size=8)
        self.assertIsInstance(dataset, tf.data.Dataset)
    
    def test_model_creation(self):
        """Test model creation"""
        model = create_fraud_model(len(self.feature_cols), self.config)
        self.assertIsInstance(model, tf.keras.Model)
        
        # Test forward pass
        test_input = tf.random.normal((4, len(self.feature_cols)))
        output = model(test_input)
        self.assertEqual(output.shape, (4, 1))

if __name__ == '__main__':
    unittest.main()
