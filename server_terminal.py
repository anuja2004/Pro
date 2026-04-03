#!/usr/bin/env python3
"""
Real-Time Federated Learning Server
Run this in Terminal 1
"""
import socket
import json
import threading
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import sys
import struct
import time

HOST = 'localhost'
PORT = 5001
MAX_CLIENTS = 2
BUFFER_SIZE = 8192  # 8KB chunks

class RealTimeFederatedServer:
    def __init__(self):
        self.clients = {}
        self.client_weights = {}
        self.client_sizes = {}
        self.client_features = {}
        self.current_round = 0
        self.global_model = None
        self.feature_count = None
        self.lock = threading.Lock()
        self.rounds_complete = 0
        self.max_rounds = 20
        self.clients_ready = threading.Condition()
        self.expected_clients = MAX_CLIENTS
        
        print("\033[96m" + "="*60 + "\033[0m")
        print("\033[96m🚀 REAL-TIME FEDERATED LEARNING SERVER\033[0m")
        print("\033[96m" + "="*60 + "\033[0m")
        print(f"📡 Server starting on {HOST}:{PORT}")
        print(f"👥 Waiting for {MAX_CLIENTS} clients...\n")
        
    def send_large_message(self, sock, data):
        """Send large message in chunks"""
        try:
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
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False
        except Exception as e:
            print(f"\033[91mError sending: {e}\033[0m")
            return False
    
    def receive_large_message(self, sock):
        """Receive large message in chunks"""
        try:
            # Receive size first
            size_data = sock.recv(4)
            if not size_data:
                return None
            size = struct.unpack('!I', size_data)[0]
            
            if size > 10 * 1024 * 1024:  # 10MB limit
                print(f"\033[91mMessage too large: {size} bytes\033[0m")
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
            
            return json.loads(json_bytes.decode('utf-8'))
        except socket.timeout:
            return None
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:
            print(f"\033[91mError receiving: {e}\033[0m")
            return None
    
    def create_initial_model(self):
        """Create initial model with correct feature count"""
        if self.feature_count is None:
            return
        
        self.global_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.feature_count,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.global_model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print(f"\033[96m✅ Model created with {self.feature_count} features\033[0m")
    
    def broadcast(self, message, sender="SERVER"):
        """Print with color"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\033[96m[{timestamp}] [{sender}]\033[0m {message}")
    
    def handle_client(self, client_socket, addr):
        """Handle individual client"""
        client_id = None
        client_socket.settimeout(60)
        
        try:
            # Receive client info
            client_info = self.receive_large_message(client_socket)
            if not client_info:
                self.broadcast("❌ Failed to receive client info", "SERVER")
                return
            
            client_id = client_info['client_id']
            dataset_size = client_info['dataset_size']
            client_features = client_info.get('feature_count', 35)
            
            self.broadcast(f"📥 Received connection from {client_id}", "SERVER")
            self.broadcast(f"   • Dataset size: {dataset_size} samples", "SERVER")
            self.broadcast(f"   • Features: {client_features}", "SERVER")
            
            with self.lock:
                if self.feature_count is None:
                    self.feature_count = client_features
                    self.create_initial_model()
                    self.broadcast(f"📊 Using {self.feature_count} features from {client_id}", "SERVER")
                
                if self.feature_count != client_features:
                    error_msg = f"Feature mismatch! Server has {self.feature_count}, client has {client_features}"
                    self.broadcast(f"❌ {error_msg}", "SERVER")
                    error_response = {'type': 'error', 'message': error_msg}
                    self.send_large_message(client_socket, error_response)
                    return
                
                self.clients[client_id] = client_socket
                self.client_sizes[client_id] = dataset_size
                self.client_features[client_id] = client_features
            
            self.broadcast(f"✅ {client_id} fully connected! ({len(self.clients)}/{MAX_CLIENTS} clients)", "SERVER")
            
            # Send initial model
            if self.global_model is not None:
                weights = [w.tolist() for w in self.global_model.get_weights()]
                model_msg = {
                    'type': 'model',
                    'round': 0,
                    'weights': weights
                }
                if self.send_large_message(client_socket, model_msg):
                    self.broadcast(f"📤 Sent initial model to {client_id}", "SERVER")
            
            # Wait for all clients
            with self.clients_ready:
                while len(self.clients) < self.expected_clients:
                    self.clients_ready.wait(timeout=1)
            
            # Training loop
            while self.current_round < self.max_rounds:
                update = self.receive_large_message(client_socket)
                
                if not update:
                    break
                
                if update['type'] == 'weights':
                    round_num = update.get('round', self.current_round + 1)
                    
                    with self.lock:
                        self.client_weights[client_id] = {
                            'weights': [np.array(w, dtype=np.float32) for w in update['weights']],
                            'size': dataset_size,
                            'metrics': update['metrics']
                        }
                    
                    self.broadcast(f"📥 Received weights from {client_id} - Loss: {update['metrics']['loss']:.4f}", "SERVER")
                    
                    # Check if all clients sent weights
                    with self.lock:
                        if len(self.client_weights) == len(self.clients):
                            self.aggregate_and_distribute()
            
        except Exception as e:
            self.broadcast(f"❌ Error with {client_id}: {e}", "SERVER")
        finally:
            with self.lock:
                if client_id and client_id in self.clients:
                    del self.clients[client_id]
                if client_id and client_id in self.client_weights:
                    del self.client_weights[client_id]
            client_socket.close()
            self.broadcast(f"🔌 {client_id} disconnected", "SERVER")
    
    def aggregate_and_distribute(self):
        """Federated averaging"""
        self.current_round += 1
        self.rounds_complete += 1
        
        print("\n" + "\033[96m" + "="*60 + "\033[0m")
        print(f"\033[96m🔄 ROUND {self.current_round} AGGREGATION\033[0m")
        print("\033[96m" + "="*60 + "\033[0m")
        
        total_samples = sum(self.client_sizes.values())
        first_client = list(self.client_weights.values())[0]
        num_layers = len(first_client['weights'])
        
        new_weights = []
        for layer_idx in range(num_layers):
            weighted_sum = None
            for client_id, data in self.client_weights.items():
                weight_array = data['weights'][layer_idx]
                client_size = self.client_sizes[client_id]
                
                if weighted_sum is None:
                    weighted_sum = weight_array * client_size
                else:
                    weighted_sum += weight_array * client_size
            
            avg_weight = weighted_sum / total_samples
            new_weights.append(avg_weight)
        
        self.global_model.set_weights(new_weights)
        
        # Show results
        print("\n📊 ROUND RESULTS:")
        for client_id, data in self.client_weights.items():
            m = data['metrics']
            color = '\033[92m' if 'bank' in client_id else '\033[93m'
            print(f"{color}   {client_id}: Loss={m['loss']:.4f}, Acc={m['accuracy']:.4f}, Recall={m['recall']:.4f}\033[0m")
        
        # Send updated model
        weights_list = [w.tolist() for w in new_weights]
        model_msg = {
            'type': 'model',
            'round': self.current_round,
            'weights': weights_list
        }
        
        print("\n📤 Sending updated model to clients:")
        for client_id, client_socket in list(self.clients.items()):
            if self.send_large_message(client_socket, model_msg):
                print(f"\033[96m   ✅ Sent to {client_id}\033[0m")
            else:
                print(f"\033[91m   ❌ Failed to send to {client_id}\033[0m")
                # Remove dead client
                with self.lock:
                    if client_id in self.clients:
                        del self.clients[client_id]
        
        self.client_weights = {}
        
        if self.rounds_complete >= self.max_rounds:
            print("\n" + "\033[96m" + "="*60 + "\033[0m")
            print("\033[96m🎉 TRAINING COMPLETE!\033[0m")
            print("\033[96m" + "="*60 + "\033[0m")
            
            complete_msg = {'type': 'complete'}
            for client_id, client_socket in list(self.clients.items()):
                self.send_large_message(client_socket, complete_msg)
            
            self.save_model()
    
    def save_model(self):
        """Save final model"""
        os.makedirs('models/saved_models', exist_ok=True)
        self.global_model.save('models/saved_models/real_time_model_final.keras')
        print("\033[96m💾 Model saved: models/saved_models/real_time_model_final.keras\033[0m")
    
    def start(self):
        """Start server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((HOST, PORT))
            server.listen(MAX_CLIENTS)
            
            def check_clients():
                while True:
                    with self.clients_ready:
                        if len(self.clients) >= self.expected_clients:
                            self.broadcast(f"\n🎯 All {MAX_CLIENTS} clients connected! Starting training...\n", "SERVER")
                            self.clients_ready.notify_all()
                    time.sleep(1)
            
            client_checker = threading.Thread(target=check_clients, daemon=True)
            client_checker.start()
            
            self.broadcast(f"📡 Server ready on {HOST}:{PORT}", "SERVER")
            self.broadcast(f"👥 Waiting for {MAX_CLIENTS} clients...\n", "SERVER")
            
            while True:
                client_socket, addr = server.accept()
                thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                thread.daemon = True
                thread.start()
                
        except KeyboardInterrupt:
            print("\n\033[96m👋 Server shutting down...\033[0m")
        finally:
            server.close()

if __name__ == "__main__":
    server = RealTimeFederatedServer()
    server.start()