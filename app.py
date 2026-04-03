"""
Flask Web App - Fully dynamic, no hardcoded values
"""
from flask import Flask, render_template, request, jsonify
from live_detector import LiveFraudDetector
import random
from datetime import datetime

app = Flask(__name__)
detector = LiveFraudDetector()
transaction_history = []

@app.route('/')
def index():
    """Dynamic dashboard"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Dynamic prediction endpoint"""
    data = request.get_json()
    
    transaction = {
        'id': f"TX{len(transaction_history)+1:05d}",
        'amount': float(data.get('amount', 0)),
        'merchant': data.get('merchant', 'Unknown'),
        'velocity': int(data.get('velocity', 1)),
        'card_age': int(data.get('card_age', 365)),
        'is_foreign': data.get('is_foreign', False),
        'is_night': data.get('is_night', False),
        'timestamp': datetime.now().isoformat()
    }
    
    result = detector.predict(transaction)
    transaction_history.append({'transaction': transaction, 'result': result})
    
    return jsonify(result)

@app.route('/api/random')
def random_transaction():
    """Generate random transaction dynamically"""
    transaction = {
        'id': f"RND{random.randint(1000, 9999)}",
        'amount': round(random.uniform(10, 5000), 2),
        'merchant': random.choice(['Coffee Shop', 'Amazon', 'Walmart', 'Foreign Exchange', 'Casino']),
        'velocity': random.randint(1, 25),
        'card_age': random.randint(1, 1000),
        'is_foreign': random.choice([True, False]),
        'is_night': random.choice([True, False]),
        'timestamp': datetime.now().isoformat()
    }
    
    result = detector.predict(transaction)
    transaction_history.append({'transaction': transaction, 'result': result})
    
    return jsonify(result)

@app.route('/api/stats')
def stats():
    """Get dynamic statistics"""
    if not transaction_history:
        return jsonify({'total': 0, 'blocked': 0, 'flagged': 0, 'approved': 0})
    
    total = len(transaction_history)
    blocked = sum(1 for t in transaction_history if 'BLOCK' in t['result']['action'])
    flagged = sum(1 for t in transaction_history if 'FLAG' in t['result']['action'])
    
    return jsonify({
        'total': total,
        'blocked': blocked,
        'flagged': flagged,
        'approved': total - blocked - flagged,
        'avg_score': sum(t['result']['fraud_score'] for t in transaction_history) / total if total > 0 else 0
    })

@app.route('/api/history')
def history():
    """Get transaction history"""
    return jsonify(transaction_history[-20:])

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 DYNAMIC FRAUD DETECTION WEB APP")
    print("="*60)
    print(f"\n📡 Model: {detector.feature_count} features dynamically loaded")
    print(f"📊 Thresholds: High={detector.thresholds['high']:.2f}, Medium={detector.thresholds['medium']:.2f}")
    print("\n🔗 Open: http://localhost:5002")
    
    app.run(host='0.0.0.0', port=5002, debug=True)