"""
Federated Learning Client Implementation
"""
import tensorflow as tf
import numpy as np
from src.utils.metrics import fraud_metrics

class FederatedClient:
    """Represents a client in federated learning"""
    
    def __init__(self, client_id, data, feature_columns, batch_size=32):
        """
        Initialize a federated client
        
        Args:
            client_id: Unique identifier for the client
            data: DataFrame with features and 'fraud_label' column
            feature_columns: List of feature column names
            batch_size: Batch size for training
        """
        self.client_id = client_id
        self.data = data
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        
        # Separate features and labels
        self.features = data[feature_columns].values.astype(np.float32)
        self.labels = data['fraud_label'].values.astype(np.float32)
        
        # Calculate class weights for imbalance
        self.fraud_ratio = np.sum(self.labels) / len(self.labels)
        self.class_weight = {0: 1.0, 1: (1.0 / self.fraud_ratio) * 0.5}
        
        print(f"✅ Client {client_id} initialized with {len(data)} samples")
        print(f"   Fraud rate: {self.fraud_ratio:.4f}")
    
    # def get_tf_dataset(self, batch_size=None):
    #     """Create a balanced TensorFlow dataset"""
        
    #     if batch_size is None:
    #         batch_size = self.batch_size
        
    #     # Create dataset
    #     dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
        
    #     # Handle class imbalance with sampling
    #     fraud_indices = np.where(self.labels == 1)[0]
    #     legit_indices = np.where(self.labels == 0)[0]
        
    #     # If no fraud samples, return regular dataset
    #     if len(fraud_indices) == 0:
    #         return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    #     # Create separate datasets for fraud and legit
    #     fraud_dataset = tf.data.Dataset.from_tensor_slices(
    #         (self.features[fraud_indices], self.labels[fraud_indices])
    #     )
    #     legit_dataset = tf.data.Dataset.from_tensor_slices(
    #         (self.features[legit_indices], self.labels[legit_indices])
    #     )
        
    #     # Balance the datasets
    #     fraud_dataset = fraud_dataset.repeat()
    #     legit_dataset = legit_dataset.repeat()
        
    #     # Sample with 50/50 probability
    #     balanced_dataset = tf.data.Dataset.sample_from_datasets(
    #         [fraud_dataset, legit_dataset],
    #         weights=[0.5, 0.5],
    #         stop_on_empty_dataset=True
    #     )
        
    #     return balanced_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_tf_dataset(self, batch_size=None):
        """Create a balanced TensorFlow dataset with better sampling"""
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # Handle class imbalance with better sampling
        fraud_indices = np.where(self.labels == 1)[0]
        legit_indices = np.where(self.labels == 0)[0]
        
        # If no fraud samples, return regular dataset
        if len(fraud_indices) == 0:
            dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
            return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Create separate datasets for fraud and legit
        fraud_dataset = tf.data.Dataset.from_tensor_slices(
            (self.features[fraud_indices], self.labels[fraud_indices])
        ).repeat()
        
        legit_dataset = tf.data.Dataset.from_tensor_slices(
            (self.features[legit_indices], self.labels[legit_indices])
        ).repeat()
        
        # Sample with 70/30 probability (favor fraud samples more)
        balanced_dataset = tf.data.Dataset.sample_from_datasets(
            [fraud_dataset, legit_dataset],
            weights=[0.7, 0.3],  # 70% fraud, 30% legitimate
            stop_on_empty_dataset=True
        )
        
        return balanced_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def evaluate(self, model):
        """Evaluate model on client data"""
        
        # Create test dataset
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.features, self.labels)
        ).batch(self.batch_size * 2)
        
        # Get predictions
        y_pred_proba = model.predict(test_dataset)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = fraud_metrics(self.labels, y_pred, y_pred_proba)
        
        return metrics
    
    def get_statistics(self):
        """Get client statistics"""
        return {
            'client_id': self.client_id,
            'n_samples': len(self.data),
            'fraud_count': int(np.sum(self.labels)),
            'legit_count': int(len(self.labels) - np.sum(self.labels)),
            'fraud_rate': float(self.fraud_ratio),
            'n_features': len(self.feature_columns)
        }
