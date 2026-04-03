"""
Real-Time Demo - Dynamically generates realistic transactions
No hardcoded patterns - uses actual model predictions
"""
import time
import random
import numpy as np
from datetime import datetime
from live_detector import LiveFraudDetector

class RealtimeFraudMonitor:
    def __init__(self):
        self.detector = LiveFraudDetector()
        self.transaction_counter = 0
        
    def generate_transaction(self):
        """Generate realistic transaction based on real-world patterns"""
        self.transaction_counter += 1
        
        # Realistic random generation
        is_weekend = datetime.now().weekday() >= 5
        is_night = datetime.now().hour < 6 or datetime.now().hour > 22
        
        # Normal spending patterns
        if is_weekend:
            base_amount = random.uniform(50, 300)
        else:
            base_amount = random.uniform(10, 100)
        
        # Random chance of suspicious activity (based on real fraud rates ~1%)
        is_suspicious = random.random() < 0.05  # 5% for demo
        
        if is_suspicious:
            transaction = {
                'amount': base_amount * random.uniform(5, 20),
                'velocity': random.randint(10, 30),
                'card_age': random.randint(1, 30),
                'merchant': random.choice(['Unknown Merchant', 'Foreign Exchange', 'Online Casino']),
                'is_foreign': True,
                'is_night': True
            }
        else:
            transaction = {
                'amount': base_amount,
                'velocity': random.randint(1, 5),
                'card_age': random.randint(100, 1000),
                'merchant': random.choice(['Starbucks', 'Walmart', 'Amazon', 'Target', 'Uber']),
                'is_foreign': False,
                'is_night': is_night
            }
        
        transaction['id'] = f"TX{self.transaction_counter:05d}"
        transaction['timestamp'] = datetime.now().isoformat()
        
        return transaction
    
    def run(self, interval=2, max_transactions=20):
        """Run dynamic monitoring"""
        print("\n" + "="*60)
        print("🎬 REAL-TIME FRAUD DETECTION (Dynamic)")
        print("="*60)
        
        for _ in range(max_transactions):
            tx = self.generate_transaction()
            result = self.detector.predict(tx)
            
            # Dynamic color output
            color = {'red': '\033[91m', 'yellow': '\033[93m', 'green': '\033[92m'}.get(result['color'], '\033[0m')
            
            print(f"\n{color}[{result['timestamp'][11:19]}] {tx['id']}\033[0m")
            print(f"   Amount: ${tx['amount']:.2f} | Merchant: {tx['merchant']}")
            print(f"   Fraud Score: {result['fraud_score']:.3f}")
            print(f"   Decision: {result['action']}")
            
            time.sleep(interval)

if __name__ == "__main__":
    monitor = RealtimeFraudMonitor()
    monitor.run()