"""
Preprocessing for Bank Account Fraud (BAF) dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_baf(df):
    """
    Preprocess Bank Account Fraud dataset
    
    Args:
        df: Raw BAF DataFrame
        
    Returns:
        Processed DataFrame with unified features
    """
    print("📊 Preprocessing BAF dataset...")
    
    processed = pd.DataFrame()
    
    # 1. Target variable
    processed['fraud_label'] = df['fraud_bool']
    
    # 2. Monetary features
    processed['amount'] = df['intended_balcon_amount'].abs()  # Handle negatives
    processed['credit_limit'] = df['proposed_credit_limit']
    
    # 3. Time features
    processed['time_since_request_days'] = df['days_since_request']
    processed['month'] = df['month']
    
    # 4. Customer age
    processed['customer_age'] = df['customer_age']
    
    # 5. Location velocity (normalized)
    processed['location_velocity'] = df['zip_count_4w'] / 1000
    
    # 6. Transaction velocity (log transform to handle skewness)
    processed['velocity_6h'] = np.log1p(df['velocity_6h'].clip(lower=0))
    processed['velocity_24h'] = np.log1p(df['velocity_24h'].clip(lower=0))
    processed['velocity_4w'] = np.log1p(df['velocity_4w'].clip(lower=0))
    
    # 7. Device features
    processed['device_fraud_count'] = df['device_fraud_count']
    processed['device_distinct_emails'] = df['device_distinct_emails_8w']
    
    # Encode device OS
    le = LabelEncoder()
    processed['device_os_encoded'] = le.fit_transform(df['device_os'].fillna('unknown'))
    
    # 8. Identity features
    processed['name_email_similarity'] = df['name_email_similarity']
    processed['email_is_free'] = df['email_is_free']
    
    # 9. Phone validation
    processed['phone_valid'] = ((df['phone_home_valid'] == 1) | 
                                (df['phone_mobile_valid'] == 1)).astype(int)
    
    # 10. Banking features
    processed['bank_tenure_months'] = df['bank_months_count']
    processed['credit_risk_score'] = df['credit_risk_score']
    processed['has_other_cards'] = df['has_other_cards']
    
    # 11. Address stability
    processed['address_stability'] = df['current_address_months_count'] / 100
    
    # 12. Employment status (one-hot encode)
    employment_dummies = pd.get_dummies(df['employment_status'], prefix='emp')
    for col in employment_dummies.columns:
        processed[col] = employment_dummies[col]
    
    # 13. Housing status (one-hot encode)
    housing_dummies = pd.get_dummies(df['housing_status'], prefix='house')
    for col in housing_dummies.columns:
        processed[col] = housing_dummies[col]
    
    # 14. Session features
    processed['session_length'] = df['session_length_in_minutes']
    processed['keep_alive'] = df['keep_alive_session']
    
    # 15. Request source
    processed['is_foreign_request'] = df['foreign_request']
    processed['source_internet'] = (df['source'] == 'INTERNET').astype(int)
    
    # Fill any missing values
    processed = processed.fillna(0)
    
    print(f"✅ BAF preprocessing complete: {processed.shape}")
    print(f"   Features created: {len(processed.columns)}")
    
    return processed
