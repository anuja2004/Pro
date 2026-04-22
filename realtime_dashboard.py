"""
Real-Time Fraud Detection Dashboard
NOW USING REAL TRANSACTIONS FROM YOUR IEEE DATASET
"""
import time
import random
import threading
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from live_detector import LiveFraudDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fraud_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

streaming_active = False

# Load REAL transaction data
print("\n📂 Loading REAL transaction data...")
df = pd.read_csv('data/raw/ieee_fraud_detection.csv')

# Load the exact corresponding ML Features created by federated learning preprocessing
print("🔬 Loading exact ML feature vectors...")
ml_features_df = pd.read_csv('data/processed/ieee_processed.csv')
if 'fraud_label' in ml_features_df.columns:
    ml_features_matrix = ml_features_df.drop(columns=['fraud_label']).to_numpy()
else:
    ml_features_matrix = ml_features_df.to_numpy()

print(f"✅ Loaded {len(df)} real transactions")
print(f"   Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

# Store all real transactions in a list for streaming
print("⏳ Optimizing data format for streaming... (this will be fast)")

# Vectorized pandas operations for lightning fast preprocessing
trans_date = pd.to_datetime(df['trans_date_trans_time'])

# Construct a new dataframe with just the columns we need
processed_df = pd.DataFrame({
    'id': 'TX' + df['trans_num'].fillna(pd.Series(df.index)).astype(str).str[:8],
    'amount': df['amt'].astype(float),
    'merchant': df['merchant'].fillna('Unknown').astype(str).str[:30],
    'category': df['category'].fillna('Unknown').astype(str),
    'is_fraud': df['is_fraud'].astype(int),
    'hour': trans_date.dt.hour,
    'is_night': (trans_date.dt.hour < 6) | (trans_date.dt.hour > 22),
    'timestamp': trans_date.dt.strftime('%H:%M:%S'),
    'date': trans_date.dt.strftime('%Y-%m-%d'),
    'city': df['city'].fillna('Unknown').astype(str),
    'state': df['state'].fillna('Unknown').astype(str)
})

# Convert to list of dictionaries instantly
real_transactions = processed_df.to_dict('records')

print(f"✅ Prepared {len(real_transactions)} transactions for streaming")
print(f"   Will stream in chronological order\n")

print("👤 Building historical user profiles...")
df['cc_num'] = df['cc_num'].astype(str)
user_means = df.groupby('cc_num')['amt'].mean()
last_indices = df.groupby('cc_num').tail(1).index

historical_profiles = {
    str(df.loc[idx, 'cc_num']): {
        'ml_features': ml_features_matrix[idx],
        'mean_amt': user_means[str(df.loc[idx, 'cc_num'])],
        'base_row': df.loc[idx].to_dict()
    }
    for idx in last_indices
}
max_amt = float(df['amt'].max())
print(f"✅ Generated {len(historical_profiles)} unique user profiles\n")

# Keep track of current position
current_index = 0
transactions = []
current_stats = {
    'total': 0,
    'fraud': 0,
    'legit': 0,
    'fraud_rate': 0
}

# Initialize Federated Learning Model
print("\n🤖 Initializing Federated ML Model...")
ml_detector = LiveFraudDetector()

def get_ml_reasons(transaction, percentage):
    """Fallback reasons for ML since it's a black box"""
    if percentage < 40:
        return []
        
    reasons = ["🤖 Neural Network detected complex anomalous pattern"]
    if transaction['amount'] > 500:
        reasons.append(f"💰 Unusually high amount (${transaction['amount']:.2f})")
    if transaction.get('category', '') in ['misc_net', 'misc_pos', 'travel', 'gas_transport']:
        reasons.append(f"📦 High-risk category: {transaction['category']}")
    if transaction.get('is_night', False):
        reasons.append(f"🌙 Unusual transaction time ({transaction.get('hour', 0)}:00)")
    
    return reasons

def process_transaction(transaction):
    """Process a REAL transaction using the Federated ML Model"""
    global current_stats
    
    # Send transaction to ML model
    ml_result = ml_detector.predict(transaction)
    
    # Map ML decision to UI
    raw_score = ml_result['fraud_score']
    thresholds = ml_result['threshold_used']
    thresh_high = thresholds['high']
    thresh_med = thresholds['medium']
    
    # Scale raw probability to UI percentage
    if raw_score >= thresh_high:
        pct = 75 + ((raw_score - thresh_high) / max(0.001, 1.0 - thresh_high)) * 24
        decision = 'BLOCK'
        icon = '🔴'
        color = '#ff4444'
        risk_level = 'high'
    elif raw_score >= thresh_med:
        pct = 40 + ((raw_score - thresh_med) / max(0.001, thresh_high - thresh_med)) * 34
        decision = 'FLAG'
        icon = '🟡'
        color = '#ffaa00'
        risk_level = 'medium'
    else:
        pct = 5 + ((raw_score) / max(0.001, thresh_med)) * 34
        decision = 'APPROVE'
        icon = '🟢'
        color = '#00c851'
        risk_level = 'low'
        
    percentage = min(round(pct, 1), 99.9)
    reasons = get_ml_reasons(transaction, percentage)
    
    # Create result
    result = {
        'transaction': transaction,
        'fraud_score': percentage,
        'decision': decision,
        'risk_level': risk_level,
        'icon': icon,
        'color': color,
        'reasons': reasons,
        'actual_fraud': transaction['is_fraud'],
        'processed_time': datetime.now().strftime('%H:%M:%S')
    }
    
    # Update stats
    current_stats['total'] += 1
    if decision in ['BLOCK', 'FLAG']:
        current_stats['fraud'] += 1
        
        # Log detected frauds to CSV
        log_file = 'data/processed/detected_frauds.csv'
        file_exists = os.path.isfile(log_file)
        try:
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Transaction_ID', 'Processed_Time', 'Amount', 'Merchant', 'Category', 'AI_Decision', 'AI_Fraud_Probability_Pct', 'Is_Actually_Fraud'])
                writer.writerow([
                    transaction['id'],
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    transaction['amount'],
                    transaction['merchant'],
                    transaction.get('category', ''),
                    decision,
                    percentage,
                    transaction['is_fraud']
                ])
        except Exception as e:
            print(f"Failed to log fraud to CSV: {e}")
            
    else:
        current_stats['legit'] += 1
    current_stats['fraud_rate'] = round((current_stats['fraud'] / current_stats['total']) * 100, 1)
    
    # Store in history (keep last 50)
    transactions.insert(0, result)
    if len(transactions) > 50:
        transactions.pop()
    
    return result

def stream_real_transactions():
    """Background thread to stream REAL transactions in order"""
    global current_index, streaming_active
    
    while current_index < len(real_transactions) and streaming_active:
        try:
            # Get next real transaction
            transaction = real_transactions[current_index]
            
            # Inject exact neural-network features for this row
            transaction['raw_features'] = ml_features_matrix[current_index]
            
            current_index += 1
            
            # Process the transaction
            result = process_transaction(transaction)
            
            # Send to browser
            socketio.emit('new_transaction', result)
            socketio.emit('update_stats', current_stats)
            
            # Wait before next transaction (simulate real-time)
            socketio.sleep(2)
        except Exception as e:
            print(f"❌ Error in streaming thread: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if streaming_active:
        # When all transactions are done, send completion message
        socketio.emit('stream_complete', {'message': 'All transactions processed!'})
        streaming_active = False

# Flask Routes
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    return jsonify(current_stats)

@app.route('/api/history')
def get_history():
    return jsonify(transactions[:20])

@app.route('/api/remaining')
def get_remaining():
    remaining = len(real_transactions) - current_index
    return jsonify({'remaining': remaining, 'total': len(real_transactions), 'current': current_index})

@app.route('/api/users')
def get_users():
    """Return a small list of sample users, intentionally including our specific known fraud victim"""
    sample_users = sorted(list(historical_profiles.keys()))[:10]
    
    # Guarantee the specific fraud victim we analyzed is in the list
    fraud_victim_cc = "3560725013359375"
    if fraud_victim_cc not in sample_users and fraud_victim_cc in historical_profiles:
        sample_users.insert(0, fraud_victim_cc)
        
    return jsonify({'users': sample_users})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fraud for custom transaction based on history"""
    data = request.json
    
    account_id = str(data.get('account_id', ''))
    user_profile = historical_profiles.get(account_id)
    amount = float(data.get('amount', 0))
    
    # Generate the baseline features using the user's history
    if user_profile:
        base_features = user_profile['ml_features'].copy()
        
        # Scale amount exactly like preprocessing (Amount is index 1 for IEEE)
        base_features[1] = amount / max_amt
        
        # Recalculate dynamic 'credit_risk_score' (index 4 for IEEE)
        # Formula: (amount / (mean_amt + 1e-8)).clip(0,10) / 10
        user_mean = user_profile['mean_amt']
        risk_score = min(amount / (user_mean + 1e-8), 10.0) / 10.0
        base_features[4] = risk_score
        
    else:
        # Fallback empty array
        base_features = np.zeros(ml_detector.feature_count)
        
    now = datetime.now()
    transaction = {
        'id': f"MANUAL_{now.strftime('%H%M%S')}",
        'amount': amount,
        'merchant': data.get('merchant', 'Custom'),
        'category': data.get('category', 'Custom'),
        'timestamp': now.strftime('%H:%M:%S'),
        'date': now.strftime('%Y-%m-%d'),
        'is_fraud': 0,  # Unknown for manual tests
        'city': 'Manual Base',
        'hour': now.hour,
        'is_night': now.hour < 6 or now.hour > 22,
        'raw_features': base_features
    }
    
    result = process_transaction(transaction)
    return jsonify(result)

@socketio.on('connect')
def handle_connect():
    emit('update_stats', current_stats)
    emit('history', transactions[:20])
    emit('stream_status', {'remaining': len(real_transactions) - current_index, 'total': len(real_transactions)})

@socketio.on('start_stream')
def handle_start_stream():
    """Start streaming real transactions when client requests"""
    global streaming_active
    if not streaming_active:
        streaming_active = True
        socketio.start_background_task(stream_real_transactions)
        emit('stream_started', {'message': 'Started streaming real transactions'})

@socketio.on('stop_stream')
def handle_stop_stream():
    """Stop streaming"""
    global streaming_active
    streaming_active = False
    emit('stream_stopped', {'message': 'Streaming paused.'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔴 REAL-TIME FRAUD DETECTION DASHBOARD")
    print("="*60)
    print(f"\n📂 USING REAL DATA: {len(real_transactions)} transactions")
    print(f"   Fraud cases: {sum(1 for t in real_transactions if t['is_fraud'])}")
    print("\n📡 Starting server...")
    print("🌐 Open browser: http://localhost:5003")
    print("▶️  Click 'Start Streaming' to begin real transaction playback")
    print("\n⚠️  Note: Transactions will play in chronological order\n")
    
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, use_reloader=False)