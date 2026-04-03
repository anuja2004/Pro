"""
Main execution script - No TensorFlow Federated dependency
"""
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from src.preprocessing.feature_aligner import align_features
from src.federated.client import FederatedClient
from src.federated.server_simple import SimpleFederatedServer as FederatedServer
from src.utils.logger import setup_logger
from src.utils.visualization import plot_training_history

# Setup logging
logger = setup_logger('federated_fraud_detection')

def load_config(config_path='config/hyperparameters.yaml'):
    """Load configuration"""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml'):
                config = yaml.safe_load(f)
            else:
                import json
                config = json.load(f)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_path} not found. Using default config.")
        # Default configuration
        return {
            'model': {
                'layer1_units': 64,
                'layer2_units': 32,
                'layer3_units': 16,
                'dropout1': 0.3,
                'dropout2': 0.2,
                'dropout3': 0.1,
                'l2_lambda': 0.001
            },
            'training': {
                'batch_size': 32,
                'client_learning_rate': 0.001,
                'server_learning_rate': 1.0,
                'clients_per_round': 2,
                'local_epochs': 1
            }
        }

def load_data(quick_test=False):
    """Load and preprocess datasets"""
    logger.info("📂 Loading datasets...")
    
    try:
        if quick_test:
            # Try to load samples first, then fall back to full data with limit
            try:
                baf_df = pd.read_csv('data/samples/baf_sample.csv')
                logger.info(f"✅ Loaded BAF sample: {baf_df.shape}")
            except FileNotFoundError:
                baf_df = pd.read_csv('data/raw/bank_account_fraud.csv').head(1000)
                logger.info(f"✅ Loaded BAF first 1000 rows: {baf_df.shape}")
            
            try:
                ieee_df = pd.read_csv('data/samples/ieee_sample.csv')
                logger.info(f"✅ Loaded IEEE sample: {ieee_df.shape}")
            except FileNotFoundError:
                ieee_df = pd.read_csv('data/raw/ieee_fraud_detection.csv').head(1000)
                logger.info(f"✅ Loaded IEEE first 1000 rows: {ieee_df.shape}")
        else:
            # Load full datasets
            baf_df = pd.read_csv('data/raw/bank_account_fraud.csv')
            ieee_df = pd.read_csv('data/raw/ieee_fraud_detection.csv')
            logger.info(f"✅ Loaded full BAF dataset: {baf_df.shape}")
            logger.info(f"✅ Loaded full IEEE dataset: {ieee_df.shape}")
        
        return baf_df, ieee_df
    
    except FileNotFoundError as e:
        logger.error(f"❌ Dataset not found: {e}")
        logger.error("Please place your dataset files in data/raw/")
        logger.error("Expected files: bank_account_fraud.csv and ieee_fraud_detection.csv")
        
        # Create dummy data for testing
        logger.warning("Creating dummy data for testing...")
        return create_dummy_data()

def create_dummy_data():
    """Create dummy data for testing when real data isn't available"""
    np.random.seed(42)
    
    # Create dummy BAF data
    n_samples_baf = 500
    baf_df = pd.DataFrame({
        'fraud_bool': np.random.choice([0, 1], n_samples_baf, p=[0.98, 0.02]),
        'intended_balcon_amount': np.random.randn(n_samples_baf) * 100,
        'proposed_credit_limit': np.random.randint(100, 5000, n_samples_baf),
        'days_since_request': np.random.rand(n_samples_baf),
        'month': np.random.randint(1, 13, n_samples_baf),
        'customer_age': np.random.randint(18, 80, n_samples_baf),
        'zip_count_4w': np.random.randint(1, 100, n_samples_baf),
        'velocity_6h': np.random.randn(n_samples_baf) * 1000,
        'velocity_24h': np.random.randn(n_samples_baf) * 2000,
        'velocity_4w': np.random.randn(n_samples_baf) * 5000,
        'device_fraud_count': np.random.randint(0, 5, n_samples_baf),
        'device_distinct_emails_8w': np.random.randint(1, 10, n_samples_baf),
        'device_os': np.random.choice(['windows', 'linux', 'android', 'ios'], n_samples_baf),
        'name_email_similarity': np.random.rand(n_samples_baf),
        'email_is_free': np.random.choice([0, 1], n_samples_baf),
        'phone_home_valid': np.random.choice([0, 1], n_samples_baf),
        'phone_mobile_valid': np.random.choice([0, 1], n_samples_baf),
        'bank_months_count': np.random.randint(1, 120, n_samples_baf),
        'credit_risk_score': np.random.randint(50, 250, n_samples_baf),
        'has_other_cards': np.random.choice([0, 1], n_samples_baf),
        'current_address_months_count': np.random.randint(1, 240, n_samples_baf),
        'employment_status': np.random.choice(['CA', 'CB', 'CC', 'CD'], n_samples_baf),
        'housing_status': np.random.choice(['BA', 'BB', 'BC', 'BD'], n_samples_baf),
        'session_length_in_minutes': np.random.rand(n_samples_baf) * 30,
        'keep_alive_session': np.random.choice([0, 1], n_samples_baf),
        'foreign_request': np.random.choice([0, 1], n_samples_baf),
        'source': np.random.choice(['INTERNET', 'TELEAPP'], n_samples_baf)
    })
    
    # Create dummy IEEE data
    n_samples_ieee = 500
    ieee_df = pd.DataFrame({
        'is_fraud': np.random.choice([0, 1], n_samples_ieee, p=[0.98, 0.02]),
        'amt': np.random.randn(n_samples_ieee) * 100,
        'trans_date_trans_time': pd.date_range('2020-01-01', periods=n_samples_ieee, freq='H'),
        'dob': pd.date_range('1970-01-01', periods=n_samples_ieee, freq='Y'),
        'lat': np.random.uniform(25, 50, n_samples_ieee),
        'long': np.random.uniform(-120, -70, n_samples_ieee),
        'merch_lat': np.random.uniform(25, 50, n_samples_ieee),
        'merch_long': np.random.uniform(-120, -70, n_samples_ieee),
        'city_pop': np.random.randint(100, 1000000, n_samples_ieee)
    })
    
    logger.info(f"✅ Created dummy BAF data: {baf_df.shape}")
    logger.info(f"✅ Created dummy IEEE data: {ieee_df.shape}")
    
    return baf_df, ieee_df

def create_clients(baf_df, ieee_df, config):
    """Create federated learning clients"""
    from src.preprocessing.preprocess_baf import preprocess_baf
    from src.preprocessing.preprocess_ieee import preprocess_ieee
    
    logger.info("🔄 Preprocessing datasets...")
    
    # Preprocess each dataset
    baf_processed = preprocess_baf(baf_df)
    ieee_processed = preprocess_ieee(ieee_df)
    
    # Align features
    baf_aligned, ieee_aligned, common_features = align_features(
        baf_processed, ieee_processed
    )
    
    logger.info(f"📊 Common features: {len(common_features)}")
    logger.info(f"📊 BAF aligned shape: {baf_aligned.shape}")
    logger.info(f"📊 IEEE aligned shape: {ieee_aligned.shape}")
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    baf_aligned.to_csv('data/processed/baf_processed.csv', index=False)
    ieee_aligned.to_csv('data/processed/ieee_processed.csv', index=False)
    
    # Create clients
    client_baf = FederatedClient(
        client_id='bank_account_fraud',
        data=baf_aligned,
        feature_columns=common_features,
        batch_size=config['training']['batch_size']
    )
    
    client_ieee = FederatedClient(
        client_id='credit_card_fraud',
        data=ieee_aligned,
        feature_columns=common_features,
        batch_size=config['training']['batch_size']
    )
    
    return [client_baf, client_ieee], common_features

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Federated Learning for Fraud Detection')
    parser.add_argument('--rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--privacy', action='store_true', help='Enable differential privacy')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--quick', action='store_true', help='Quick test with small data')
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     FEDERATED LEARNING FOR FRAUD DETECTION                ║
    ║     Bank Account Fraud + Credit Card Fraud                ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📊 Configuration:")
    print(f"   • Rounds: {args.rounds}")
    print(f"   • Privacy: {'YES' if args.privacy else 'NO'}")
    print(f"   • Quick test: {'YES' if args.quick else 'NO'}")
    print(f"   • Save model: {'YES' if args.save_model else 'NO'}")
    print()
    
    # Load configuration
    config = load_config()
    
    # Load data
    baf_df, ieee_df = load_data(quick_test=args.quick)
    
    # Create clients
    clients, feature_columns = create_clients(baf_df, ieee_df, config)
    
    # Initialize federated server
    server = FederatedServer(
        feature_count=len(feature_columns),
        config=config
    )
    
    # Run federated learning
    logger.info(f"🚀 Starting federated learning for {args.rounds} rounds...")
    final_model, history = server.train(
        clients=clients,
        rounds=args.rounds,
        use_privacy=args.privacy
    )
    
    # Save results
    if args.save_model:
        os.makedirs('models/saved_models', exist_ok=True)
        os.makedirs('models/history', exist_ok=True)
        
        server.save_model('models/saved_models/federated_model_final.h5')
        server.save_history('models/history/training_history.json')
        logger.info("💾 Model and history saved")
    
    # Plot results
    os.makedirs('outputs/figures', exist_ok=True)
    try:
        plot_training_history(history, save_path='outputs/figures/')
        logger.info("📊 Training plots saved to outputs/figures/")
    except Exception as e:
        logger.warning(f"Could not plot results: {e}")
    
    logger.info("✅ Federated learning completed successfully!")
    
    # Print final summary
    print("\n" + "="*60)
    print("🎯 FEDERATED LEARNING SUMMARY")
    print("="*60)
    
    if history.get('baf_accuracy'):
        print(f"\n📈 Bank Account Client - Final Accuracy: {history['baf_accuracy'][-1]:.4f}")
    if history.get('ieee_accuracy'):
        print(f"📈 Credit Card Client - Final Accuracy: {history['ieee_accuracy'][-1]:.4f}")
    
    return final_model, history

if __name__ == "__main__":
    final_model, history = main()