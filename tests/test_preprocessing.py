"""
Unit tests for preprocessing modules
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.preprocess_baf import preprocess_baf
from src.preprocessing.preprocess_ieee import preprocess_ieee
from src.preprocessing.feature_aligner import align_features

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing"""
        # Sample BAF data
        self.baf_sample = pd.DataFrame({
            'fraud_bool': [0, 0, 1],
            'intended_balcon_amount': [100, 200, -50],
            'proposed_credit_limit': [1000, 2000, 1500],
            'days_since_request': [0.01, 0.02, 0.03],
            'month': [1, 1, 2],
            'customer_age': [30, 40, 25],
            'zip_count_4w': [100, 200, 300],
            'velocity_6h': [1000, 2000, 3000],
            'velocity_24h': [5000, 6000, 7000],
            'velocity_4w': [10000, 20000, 30000],
            'device_fraud_count': [0, 0, 1],
            'device_distinct_emails_8w': [1, 1, 2],
            'device_os': ['linux', 'windows', 'android'],
            'name_email_similarity': [0.5, 0.6, 0.7],
            'email_is_free': [1, 1, 0],
            'phone_home_valid': [1, 0, 1],
            'phone_mobile_valid': [1, 1, 1],
            'bank_months_count': [12, 24, 6],
            'credit_risk_score': [100, 200, 150],
            'has_other_cards': [1, 1, 0],
            'current_address_months_count': [12, 24, 6],
            'employment_status': ['CA', 'CB', 'CC'],
            'housing_status': ['BC', 'BA', 'BB'],
            'session_length_in_minutes': [10, 15, 20],
            'keep_alive_session': [1, 1, 0],
            'foreign_request': [0, 0, 1],
            'source': ['INTERNET', 'INTERNET', 'TELEAPP']
        })
        
        # Sample IEEE data
        self.ieee_sample = pd.DataFrame({
            'is_fraud': [0, 0, 1],
            'amt': [100, 200, 500],
            'trans_date_trans_time': ['2020-06-21 12:14:25', 
                                      '2020-06-21 13:14:25',
                                      '2020-06-22 14:14:25'],
            'dob': ['1968-03-19', '1990-01-17', '1970-10-21'],
            'lat': [33.9659, 40.3207, 40.6729],
            'long': [-80.9355, -110.436, -73.5365],
            'merch_lat': [33.9864, 39.4505, 40.4958],
            'merch_long': [-81.2007, -109.9604, -74.1961],
            'city_pop': [333497, 302, 34496]
        })
    
    def test_preprocess_baf(self):
        """Test BAF preprocessing"""
        result = preprocess_baf(self.baf_sample)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('fraud_label', result.columns)
        self.assertGreater(len(result.columns), 10)
    
    def test_preprocess_ieee(self):
        """Test IEEE preprocessing"""
        result = preprocess_ieee(self.ieee_sample)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('fraud_label', result.columns)
        self.assertIn('distance_km', result.columns)
    
    def test_align_features(self):
        """Test feature alignment"""
        baf_proc = preprocess_baf(self.baf_sample)
        ieee_proc = preprocess_ieee(self.ieee_sample)
        
        baf_aligned, ieee_aligned, features = align_features(baf_proc, ieee_proc)
        
        self.assertEqual(len(baf_aligned.columns), len(ieee_aligned.columns))
        self.assertGreater(len(features), 0)

if __name__ == '__main__':
    unittest.main()
