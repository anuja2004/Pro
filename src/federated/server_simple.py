"""
Simplified federated server - No TensorFlow Federated dependency
"""
import numpy as np
import tensorflow as tf
from src.federated.model import create_fraud_model
from src.utils.logger import setup_logger
import os
import pickle
import json
from datetime import datetime

logger = setup_logger('federated_server_simple')

class SimpleFederatedServer:
    """Simplified federated learning server (no TFF dependency)"""
    
    def __init__(self, feature_count, config):
        """
        Initialize federated server
        
        Args:
            feature_count: Number of input features
            config: Configuration dictionary
        """
        self.feature_count = feature_count
        self.config = config
        self.global_model = create_fraud_model(feature_count, config)
        self.history = {
            'round': [],
            'baf_loss': [],
            'ieee_loss': [],
            'baf_accuracy': [],
            'ieee_accuracy': [],
            'baf_precision': [],
            'ieee_precision': [],
            'baf_recall': [],
            'ieee_recall': []
        }
        
        logger.info(f"✅ Simple federated server initialized with {feature_count} features")
        logger.info(f"   Model architecture: {config['model']['layer1_units']}->{config['model']['layer2_units']}->{config['model']['layer3_units']}")
    
    def aggregate_models(self, client_models, client_sizes):
        """
        Federated averaging: weight models by client data size (FedAvg)
        
        Args:
            client_models: List of trained client models
            client_sizes: Number of samples from each client
            
        Returns:
            Aggregated model weights
        """
        total_samples = sum(client_sizes)
        
        # Get weights from first client to know structure
        aggregated_weights = client_models[0].get_weights()
        
        # Initialize with zeros
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] = aggregated_weights[i] * 0
        
        # Weighted average
        for model, n_samples in zip(client_models, client_sizes):
            weight = n_samples / total_samples
            model_weights = model.get_weights()
            
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] += weight * model_weights[i]
        
        return aggregated_weights
    
    def secure_aggregation(self, client_models, client_sizes, noise_scale=0.01):
        """
        Secure aggregation with differential privacy
        
        Args:
            client_models: List of trained client models
            client_sizes: Number of samples from each client
            noise_scale: Scale of Gaussian noise for privacy
            
        Returns:
            Noisy aggregated weights
        """
        # First do normal aggregation
        aggregated_weights = self.aggregate_models(client_models, client_sizes)
        
        # Add Gaussian noise for differential privacy
        noisy_weights = []
        for layer_weights in aggregated_weights:
            if layer_weights is not None:
                noise = np.random.normal(
                    0, noise_scale, layer_weights.shape
                )
                noisy_weights.append(layer_weights + noise)
            else:
                noisy_weights.append(layer_weights)
        
        return noisy_weights
    
    def evaluate_on_client(self, client, model=None):
        """
        Evaluate model on a specific client
        
        Args:
            client: FederatedClient object
            model: Model to evaluate (uses global model if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model is None:
            model = self.global_model
        
        # Get evaluation data
        eval_data = client.get_tf_dataset(batch_size=128).take(20)
        
        # Evaluate
        results = model.evaluate(eval_data, verbose=0)
        
        # Get predictions for additional metrics
        all_features = client.features
        all_labels = client.labels
        
        # Take a subset for faster evaluation
        subset_size = min(1000, len(all_labels))
        indices = np.random.choice(len(all_labels), subset_size, replace=False)
        
        X_subset = all_features[indices]
        y_subset = all_labels[indices]
        
        # Get predictions
        y_pred_proba = model.predict(X_subset, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_subset, y_pred, zero_division=0)
        recall = recall_score(y_subset, y_pred, zero_division=0)
        f1 = f1_score(y_subset, y_pred, zero_division=0)
        
        return {
            'loss': results[0],
            'accuracy': results[1],
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, clients, rounds=10, use_privacy=False):
        """
        Run federated training
        
        Args:
            clients: List of FederatedClient objects
            rounds: Number of federated rounds
            use_privacy: Whether to use differential privacy
            
        Returns:
            global_model: Trained global model
            history: Training history
        """
        logger.info(f"Starting federated training for {rounds} rounds")
        logger.info(f"Privacy enabled: {use_privacy}")
        
        for round_num in range(1, rounds + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Round {round_num}/{rounds}")
            logger.info(f"{'='*50}")
            
            client_models = []
            client_sizes = []
            
            # Local training on each client
            for client_idx, client in enumerate(clients):
                # Get client statistics
                stats = client.get_statistics()
                client_sizes.append(stats['n_samples'])
                
                logger.info(f"\n📱 Training on {client.client_id} ({stats['n_samples']} samples, fraud rate: {stats['fraud_rate']:.4f})")
                
                # Create a copy of global model for this client
                local_model = tf.keras.models.clone_model(self.global_model)
                local_model.set_weights(self.global_model.get_weights())
                local_model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.config['training']['client_learning_rate']
                    ),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                )
                
                # Train on client data
                train_data = client.get_tf_dataset(batch_size=self.config['training']['batch_size'])
                
                # Determine number of steps per epoch
                steps_per_epoch = min(50, len(client.labels) // self.config['training']['batch_size'])
                
                # Train the model
                history = local_model.fit(
                    train_data,
                    epochs=self.config['training'].get('local_epochs', 1),
                    steps_per_epoch=steps_per_epoch,
                    verbose=0
                )
                
                client_models.append(local_model)
                
                # Log training results
                train_loss = history.history['loss'][0]
                train_acc = history.history['accuracy'][0]
                logger.info(f"   ✅ Local training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Federated averaging
            if use_privacy:
                logger.info("\n🔒 Applying secure aggregation with differential privacy...")
                aggregated_weights = self.secure_aggregation(client_models, client_sizes)
            else:
                logger.info("\n🔄 Applying federated averaging...")
                aggregated_weights = self.aggregate_models(client_models, client_sizes)
            
            self.global_model.set_weights(aggregated_weights)
            
            # Evaluate on both clients
            logger.info("\n📊 Evaluating global model:")
            for client in clients:
                metrics = self.evaluate_on_client(client)
                
                # Store in history
                if client.client_id == 'bank_account_fraud':
                    self.history['baf_loss'].append(metrics['loss'])
                    self.history['baf_accuracy'].append(metrics['accuracy'])
                    self.history['baf_precision'].append(metrics['precision'])
                    self.history['baf_recall'].append(metrics['recall'])
                    logger.info(f"   {client.client_id}:")
                    logger.info(f"      Loss: {metrics['loss']:.4f}")
                    logger.info(f"      Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"      Precision: {metrics['precision']:.4f}")
                    logger.info(f"      Recall: {metrics['recall']:.4f}")
                    logger.info(f"      F1: {metrics['f1']:.4f}")
                else:
                    self.history['ieee_loss'].append(metrics['loss'])
                    self.history['ieee_accuracy'].append(metrics['accuracy'])
                    self.history['ieee_precision'].append(metrics['precision'])
                    self.history['ieee_recall'].append(metrics['recall'])
                    logger.info(f"   {client.client_id}:")
                    logger.info(f"      Loss: {metrics['loss']:.4f}")
                    logger.info(f"      Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"      Precision: {metrics['precision']:.4f}")
                    logger.info(f"      Recall: {metrics['recall']:.4f}")
                    logger.info(f"      F1: {metrics['f1']:.4f}")
            
            self.history['round'].append(round_num)
            
            # Save checkpoint every 5 rounds
            if round_num % 5 == 0:
                self.save_checkpoint(round_num)
        
        logger.info("\n" + "="*50)
        logger.info("🎉 FEDERATED LEARNING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
        return self.global_model, self.history
    
    def save_checkpoint(self, round_num):
        """Save model checkpoint"""
        os.makedirs('models/saved_models', exist_ok=True)
        filename = f"models/saved_models/federated_model_round{round_num}.h5"
        self.global_model.save(filename)
        logger.info(f"💾 Checkpoint saved: {filename}")
    
    def save_model(self, filepath):
        """Save final model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.global_model.save(filepath)
        logger.info(f"💾 Model saved: {filepath}")
    
    def save_history(self, filepath):
        """Save training history"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                if value and isinstance(value[0], (np.float32, np.float64)):
                    serializable_history[key] = [float(v) for v in value]
                else:
                    serializable_history[key] = value
            else:
                serializable_history[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        logger.info(f"💾 History saved: {filepath}")
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            rounds = self.history['round']
            
            # Loss plot
            if self.history.get('baf_loss') and self.history.get('ieee_loss'):
                axes[0].plot(rounds, self.history['baf_loss'], 'b-o', 
                            label='Bank Account Client', linewidth=2, markersize=4)
                axes[0].plot(rounds, self.history['ieee_loss'], 'r-s', 
                            label='Credit Card Client', linewidth=2, markersize=4)
                axes[0].set_xlabel('Federated Round', fontsize=12)
                axes[0].set_ylabel('Loss', fontsize=12)
                axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
                axes[0].legend(fontsize=10)
                axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            if self.history.get('baf_accuracy') and self.history.get('ieee_accuracy'):
                axes[1].plot(rounds, self.history['baf_accuracy'], 'b-o', 
                            label='Bank Account Client', linewidth=2, markersize=4)
                axes[1].plot(rounds, self.history['ieee_accuracy'], 'r-s', 
                            label='Credit Card Client', linewidth=2, markersize=4)
                axes[1].set_xlabel('Federated Round', fontsize=12)
                axes[1].set_ylabel('Accuracy', fontsize=12)
                axes[1].set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
                axes[1].legend(fontsize=10)
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
                logger.info(f"📊 Plot saved: {save_path}/training_history.png")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Skipping plot.")