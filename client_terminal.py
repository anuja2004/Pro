#!/usr/bin/env python3
"""
Real-Time Federated Learning Client
Run this in Terminal 2 (Bank) and Terminal 3 (Card)
"""
import socket
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import sys
import os
import pandas as pd
import struct

# Add your existing src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.preprocess_baf import preprocess_baf
from src.preprocessing.preprocess_ieee import preprocess_ieee

HOST = 'localhost'
PORT = 5001  # Changed to 5001 to avoid conflict
BUFFER_SIZE = 8192  # 8KB chunks

class RealTimeFederatedClient:
    def __init__(self, client_id, dataset_path):
        self.client_id = client_id
        self.dataset_path = dataset_path
        self.model = None
        self.current_round = 0
        self.feature_count = 35  # Will be updated after loading data
        
        # Set color based on client type
        self.color = '\033[92m' if 'bank' in client_id.lower() else '\033[93m'
        
        self.print_status("🚀 Initializing...")
        self.load_data()
    
    def send_large_message(self, sock, data):
        """Send large message in chunks"""
        try:
            # Convert to JSON string
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')
            
            # Send size first (4 bytes)
            size = len(json_bytes)
            sock.sendall(struct.pack('!I', size))
            
            # Send in chunks
            sent = 0
            while sent < size:
                chunk = json_bytes[sent:sent + BUFFER_SIZE]
                sock.sendall(chunk)
                sent += len(chunk)
                
            self.print_status(f"📤 Sent {size} bytes")
        except Exception as e:
            self.print_status(f"❌ Error sending: {e}")
            raise
    
    def receive_large_message(self, sock):
        """Receive large message in chunks"""
        try:
            # Receive size first
            size_data = sock.recv(4)
            if not size_data:
                return None
            size = struct.unpack('!I', size_data)[0]
            
            if size > 10 * 1024 * 1024:  # 10MB limit
                self.print_status(f"❌ Message too large: {size} bytes")
                return None
            
            # Receive in chunks
            received = 0
            json_bytes = b''
            while received < size:
                chunk = sock.recv(min(BUFFER_SIZE, size - received))
                if not chunk:
                    return None
                json_bytes += chunk
                received += len(chunk)
            
            # Parse JSON
            return json.loads(json_bytes.decode('utf-8'))
        except Exception as e:
            self.print_status(f"❌ Error receiving: {e}")
            return None
    
    def load_data(self):
        """Load and prepare data"""
        self.print_status(f"📂 Loading data from {self.dataset_path}...")
        
        try:
            # Check if file exists
            if not os.path.exists(self.dataset_path):
                self.print_status(f"❌ File not found: {self.dataset_path}")
                sys.exit(1)
            
            # Load dataset
            df = pd.read_csv(self.dataset_path)
            self.print_status(f"📊 Original dataset shape: {df.shape}")
            
            # Use subset for faster demo
            
            self.print_status(f"📊 Using subset: {df.shape}")
            
            # Preprocess based on client type
            if 'bank' in self.client_id.lower():
                processed = preprocess_baf(df)
                self.print_status("🏦 Bank account data preprocessed")
            else:
                processed = preprocess_ieee(df)
                self.print_status("💳 Credit card data preprocessed")
            
            self.print_status(f"📊 Processed shape: {processed.shape}")
            
            # Get features and labels
            if 'fraud_label' in processed.columns:
                self.features = processed.drop('fraud_label', axis=1).values.astype(np.float32)
                self.labels = processed['fraud_label'].values.astype(np.float32)
                self.print_status("✅ Found 'fraud_label' column")
            elif 'is_fraud' in processed.columns:
                self.features = processed.drop('is_fraud', axis=1).values.astype(np.float32)
                self.labels = processed['is_fraud'].values.astype(np.float32)
                self.print_status("✅ Found 'is_fraud' column")
            else:
                # If no label column, assume last column is label
                self.features = processed.iloc[:, :-1].values.astype(np.float32)
                self.labels = processed.iloc[:, -1].values.astype(np.float32)
                self.print_status("⚠️ Using last column as label")
            
            # Handle NaN values
            nan_count = np.isnan(self.features).sum()
            if nan_count > 0:
                self.print_status(f"⚠️ Found {nan_count} NaN values, filling with 0")
                self.features = np.nan_to_num(self.features)
            
            # Get original feature count
            original_features = self.features.shape[1]
            self.print_status(f"🔢 Original features: {original_features}")
            
            # STANDARDIZE TO 35 FEATURES
            target_features = 35
            
            if original_features > target_features:
                # Take first 35 features
                self.features = self.features[:, :target_features]
                self.feature_count = target_features
                self.print_status(f"🔧 Trimmed features from {original_features} to {target_features}")
            elif original_features < target_features:
                # Pad with zeros
                padding = np.zeros((self.features.shape[0], target_features - original_features))
                self.features = np.hstack([self.features, padding])
                self.feature_count = target_features
                self.print_status(f"🔧 Padded features from {original_features} to {target_features}")
            else:
                self.feature_count = original_features
                self.print_status(f"✅ Features already {target_features}")
            
            # Calculate fraud rate
            fraud_rate = np.sum(self.labels) / len(self.labels)
            
            self.print_status(f"✅ Loaded {len(self.labels)} samples")
            self.print_status(f"📊 Fraud rate: {fraud_rate:.4f} ({int(np.sum(self.labels))} fraud cases)")
            self.print_status(f"🔢 Final features: {self.feature_count}")
            
        except Exception as e:
            self.print_status(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def create_model(self):
        """Create local model matching server architecture"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(self.feature_count,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            self.print_status(f"✅ Created model with {self.feature_count} input features")
            return model
            
        except Exception as e:
            self.print_status(f"❌ Error creating model: {e}")
            raise
    
    def get_balanced_batch(self, batch_size=32):
        """70/30 balanced batch"""
        fraud_indices = np.where(self.labels == 1)[0]
        legit_indices = np.where(self.labels == 0)[0]
        
        if len(fraud_indices) == 0:
            idx = np.random.choice(len(self.labels), batch_size)
            return self.features[idx], self.labels[idx]
        
        # 70% fraud, 30% legitimate
        n_fraud = int(batch_size * 0.7)    ##
        n_legit = batch_size - n_fraud
        
        # Sample with replacement if not enough fraud cases
        fraud_idx = np.random.choice(fraud_indices, n_fraud, replace=len(fraud_indices) < n_fraud)
        legit_idx = np.random.choice(legit_indices, n_legit, replace=len(legit_indices) < n_legit)
        
        batch_idx = np.concatenate([fraud_idx, legit_idx])
        np.random.shuffle(batch_idx)
        
        return self.features[batch_idx], self.labels[batch_idx]
    
    def train_local(self, global_weights, round_num):
        """Train one round with threshold tuning"""
        self.current_round = round_num
        self.print_status(f"🏋️  Training round {round_num}...")
        
        try:
            # Set weights
            converted_weights = [np.array(w, dtype=np.float32) for w in global_weights]
            
            if self.model is None:
                self.model = self.create_model()
            
            self.model.set_weights(converted_weights)
            
            # Training
            total_loss = 0
            total_acc = 0
            n_batches = 30
            
            for i in range(n_batches):
                X, y = self.get_balanced_batch(32)
                history = self.model.train_on_batch(X, y, return_dict=True)
                total_loss += history['loss']
                total_acc += history['accuracy']
                
                if (i + 1) % 10 == 0:
                    self.print_status(f"   Batch {i+1}/{n_batches} - Loss: {history['loss']:.4f}")
            
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            
            # Create balanced evaluation set
            fraud_indices = np.where(self.labels == 1)[0]
            legit_indices = np.where(self.labels == 0)[0]
            
            if len(fraud_indices) > 0:
                n_fraud_eval = min(100, len(fraud_indices))
                n_legit_eval = n_fraud_eval
                
                fraud_idx = np.random.choice(fraud_indices, n_fraud_eval, replace=False)
                legit_idx = np.random.choice(legit_indices, n_legit_eval, replace=False)
                
                X_val = np.vstack([self.features[fraud_idx], self.features[legit_idx]])
                y_val = np.hstack([self.labels[fraud_idx], self.labels[legit_idx]])
                
                shuffle_idx = np.random.permutation(len(y_val))
                X_val = X_val[shuffle_idx]
                y_val = y_val[shuffle_idx]
            else:
                X_val, y_val = self.get_balanced_batch(100)
            
            # ========== THRESHOLD TUNING ==========
            y_pred_proba = self.model.predict(X_val, verbose=0)
            
            # Try multiple thresholds
            thresholds = [0.3, 0.4, 0.5]
            best_recall = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (y_pred_proba > threshold).astype(int)
                tp = np.sum((y_val == 1) & (y_pred == 1))
                fn = np.sum((y_val == 1) & (y_pred == 0))
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                if recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold
            
            # Use best threshold
            y_pred_best = (y_pred_proba > best_threshold).astype(int)
            tp = np.sum((y_val == 1) & (y_pred_best == 1))
            fn = np.sum((y_val == 1) & (y_pred_best == 0))
            fp = np.sum((y_val == 0) & (y_pred_best == 1))
            tn = np.sum((y_val == 0) & (y_pred_best == 0))
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            metrics = {
                'loss': float(avg_loss),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall)
            }
            
            self.print_status(f"✅ Best threshold: {best_threshold}")
            self.print_status(f"   Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}")
            
            if tp + fn > 0:
                self.print_status(f"   📊 Caught {tp}/{tp+fn} fraud cases ({recall*100:.1f}%)")
            
            new_weights = self.model.get_weights()
            weights_list = [w.tolist() for w in new_weights]
            
            return weights_list, metrics
            
        except Exception as e:
            self.print_status(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            dummy_weights = self.model.get_weights() if self.model else []
            dummy_metrics = {'loss': 0.5, 'accuracy': 0.5, 'precision': 0, 'recall': 0}
            return [w.tolist() for w in dummy_weights], dummy_metrics
    
    def print_status(self, message):
        """Print with color and timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{self.color}[{timestamp}] [{self.client_id}]\033[0m {message}")
    
    def run(self):
        """Connect to server and train"""
        sock = None
        try:
            # Connect to server
            self.print_status(f"🔌 Connecting to server at {HOST}:{PORT}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)  # 30 second timeout
            sock.connect((HOST, PORT))
            self.print_status("🔗 Connected to server")
            
            # Send client info
            info = {
                'client_id': self.client_id,
                'dataset_size': len(self.labels),
                'feature_count': self.feature_count
            }
            self.print_status(f"📤 Sending client info: {info}")
            self.send_large_message(sock, info)
            
            # Training loop
            round_num = 0
            while True:
                # Receive model from server
                self.print_status("📡 Waiting for model from server...")
                message = self.receive_large_message(sock)
                
                if not message:
                    self.print_status("⚠️ Connection closed by server")
                    break
                
                if message.get('type') == 'error':
                    self.print_status(f"❌ Server error: {message.get('message')}")
                    break
                
                if message['type'] == 'model':
                    round_num = message['round']
                    global_weights = message['weights']
                    
                    self.print_status(f"📥 Received model for round {round_num}")
                    
                    # Create model if needed
                    if self.model is None:
                        self.model = self.create_model()
                    
                    # Verify dimensions match
                    expected_shape = self.model.get_weights()[0].shape[1]
                    received_shape = len(global_weights[0]) if global_weights else 0
                    
                    if expected_shape != self.feature_count:
                        self.print_status(f"⚠️ Dimension mismatch: Model expects {expected_shape}, data has {self.feature_count}")
                        # Recreate model with correct dimensions
                        self.model = self.create_model()
                    
                    # Train locally
                    new_weights, metrics = self.train_local(global_weights, round_num)
                    
                    # Send weights back to server
                    update = {
                        'type': 'weights',
                        'weights': new_weights,
                        'metrics': metrics,
                        'round': round_num
                    }
                    self.send_large_message(sock, update)
                    self.print_status(f"📤 Sent weights to server for round {round_num}")
                    
                elif message['type'] == 'complete':
                    self.print_status("🎉 Training completed successfully!")
                    break
                    
        except KeyboardInterrupt:
            self.print_status("👋 Disconnecting...")
        except ConnectionRefusedError:
            self.print_status(f"❌ Could not connect to server at {HOST}:{PORT}. Is the server running?")
        except Exception as e:
            self.print_status(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if sock:
                sock.close()
                self.print_status("🔌 Disconnected from server")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\n❌ Missing arguments!")
        print("\nUsage: python client_terminal.py <client_id> <dataset_path>")
        print("\nExamples:")
        print("  Terminal 2: python client_terminal.py bank_account_fraud data/raw/bank_account_fraud.csv")
        print("  Terminal 3: python client_terminal.py credit_card_fraud data/raw/ieee_fraud_detection.csv")
        print("\nMake sure the server is running in Terminal 1 first!")
        sys.exit(1)
    
    client_id = sys.argv[1]
    dataset_path = sys.argv[2]
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("\nAvailable datasets in data/raw/:")
        if os.path.exists('data/raw'):
            for file in os.listdir('data/raw'):
                print(f"  - data/raw/{file}")
        else:
            print("  No datasets found. Create data/raw/ directory and add your CSV files.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Federated Learning Client: {client_id}")
    print(f"{'='*60}\n")
    
    client = RealTimeFederatedClient(client_id, dataset_path)
    client.run()