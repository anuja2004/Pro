"""
Differential privacy utilities for federated learning
- Simplified version with no external dependencies
"""
import numpy as np

def add_differential_privacy(gradients, noise_multiplier=1.1, l2_norm_clip=0.1):
    """
    Add differential privacy to gradients
    
    Args:
        gradients: List of gradient arrays
        noise_multiplier: Scale of noise to add
        l2_norm_clip: Maximum L2 norm for clipping
        
    Returns:
        Noisy, clipped gradients
    """
    private_gradients = []
    
    for grad in gradients:
        if grad is None:
            private_gradients.append(grad)
            continue
        
        # Convert to numpy if it's a tensor
        if hasattr(grad, 'numpy'):
            grad = grad.numpy()
        
        # Clip gradients
        grad_norm = np.linalg.norm(grad.flatten())
        if grad_norm > l2_norm_clip:
            grad = grad * (l2_norm_clip / grad_norm)
        
        # Add Gaussian noise
        noise = np.random.normal(
            0, 
            noise_multiplier * l2_norm_clip,
            grad.shape
        )
        
        private_gradients.append(grad + noise)
    
    return private_gradients

class DPMechanism:
    """Differential privacy mechanism for federated learning"""
    
    def __init__(self, epsilon=1.0, delta=1e-5, sensitivity=1.0):
        """
        Initialize DP mechanism
        
        Args:
            epsilon: Privacy budget
            delta: Failure probability
            sensitivity: Sensitivity of the query
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
    
    def gaussian_noise_scale(self):
        """Calculate Gaussian noise scale for (ε,δ)-DP"""
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_gaussian_noise(self, value):
        """Add Gaussian noise to value"""
        noise_scale = self.gaussian_noise_scale()
        
        # Handle different input types
        if isinstance(value, (int, float)):
            noise = np.random.normal(0, noise_scale)
            return value + noise
        elif isinstance(value, np.ndarray):
            noise = np.random.normal(0, noise_scale, value.shape)
            return value + noise
        else:
            # Assume it's a tensor
            import tensorflow as tf
            noise = tf.random.normal(value.shape, stddev=noise_scale)
            return value + noise
    
    def laplace_noise_scale(self):
        """Calculate Laplace noise scale for ε-DP"""
        return self.sensitivity / self.epsilon
    
    def add_laplace_noise(self, value):
        """Add Laplace noise to value"""
        noise_scale = self.laplace_noise_scale()
        
        if isinstance(value, (int, float)):
            noise = np.random.laplace(0, noise_scale)
            return value + noise
        elif isinstance(value, np.ndarray):
            noise = np.random.laplace(0, noise_scale, value.shape)
            return value + noise
        else:
            # Tensor handling
            import tensorflow as tf
            noise = tf.random.laplace(value.shape, loc=0.0, scale=noise_scale)
            return value + noise
