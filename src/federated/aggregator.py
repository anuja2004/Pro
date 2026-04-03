"""
Custom federated averaging aggregator
"""
import numpy as np
import tensorflow as tf

class FederatedAveraging:
    """
    Implements federated averaging algorithm (FedAvg)
    """
    
    def __init__(self):
        self.global_weights = None
    
    def aggregate(self, client_weights, client_sizes):
        """
        Aggregate client models using weighted average
        
        Args:
            client_weights: List of model weights from clients
            client_sizes: Number of samples from each client
            
        Returns:
            Aggregated global weights
        """
        if not client_weights:
            return None
        
        total_samples = sum(client_sizes)
        
        # Initialize aggregated weights with zeros
        aggregated_weights = [
            np.zeros_like(weights) for weights in client_weights[0]
        ]
        
        # Weighted average
        for client_weight, n_samples in zip(client_weights, client_sizes):
            weight = n_samples / total_samples
            for i, layer_weights in enumerate(client_weight):
                aggregated_weights[i] += weight * layer_weights
        
        self.global_weights = aggregated_weights
        return aggregated_weights
    
    def secure_aggregation(self, client_weights, client_sizes, noise_scale=0.01):
        """
        Secure aggregation with differential privacy
        
        Args:
            client_weights: List of model weights from clients
            client_sizes: Number of samples from each client
            noise_scale: Scale of Gaussian noise for privacy
            
        Returns:
            Noisy aggregated weights
        """
        aggregated = self.aggregate(client_weights, client_sizes)
        
        # Add Gaussian noise for differential privacy
        noisy_weights = []
        for layer_weights in aggregated:
            noise = np.random.normal(
                0, noise_scale, layer_weights.shape
            )
            noisy_weights.append(layer_weights + noise)
        
        return noisy_weights
