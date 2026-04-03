"""
Unit tests for model evaluation
"""
import unittest
import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import fraud_metrics

class TestModel(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        # Perfect predictions
        self.y_true_1 = np.array([0, 0, 1, 1])
        self.y_pred_1 = np.array([0, 0, 1, 1])
        
        # Imperfect predictions
        self.y_true_2 = np.array([0, 0, 1, 1, 0, 1])
        self.y_pred_2 = np.array([0, 1, 1, 0, 0, 1])
        
        # Probabilities
        self.y_proba = np.array([0.1, 0.8, 0.9, 0.2, 0.3, 0.7])
    
    def test_fraud_metrics_perfect(self):
        """Test metrics with perfect predictions"""
        metrics = fraud_metrics(self.y_true_1, self.y_pred_1)
        
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)
    
    def test_fraud_metrics_imperfect(self):
        """Test metrics with imperfect predictions"""
        metrics = fraud_metrics(self.y_true_2, self.y_pred_2, self.y_proba)
        
        self.assertLess(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertGreaterEqual(metrics['recall'], 0)
        self.assertIn('auc_roc', metrics)
    
    def test_confusion_matrix_values(self):
        """Test confusion matrix values"""
        metrics = fraud_metrics(self.y_true_2, self.y_pred_2)
        
        self.assertEqual(metrics['true_positives'], 2)
        self.assertEqual(metrics['false_positives'], 1)
        self.assertEqual(metrics['true_negatives'], 2)
        self.assertEqual(metrics['false_negatives'], 1)

if __name__ == '__main__':
    unittest.main()
