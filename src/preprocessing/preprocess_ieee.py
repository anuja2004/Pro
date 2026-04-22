"""
Preprocessing for IEEE-CIS Credit Card Transaction dataset
Maps IEEE columns to the SAME feature names as BAF dataset so that
the feature_aligner can find many common features instead of just 1.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between cardholder and merchant in km"""
    R = 6371
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    except:
        return 0

def preprocess_ieee(df):
    """
    Convert IEEE transaction data to features using the SAME names as BAF
    so the feature aligner can find 35 common features.
    """
    print("📊 Preprocessing IEEE transaction dataset...")

    processed = pd.DataFrame()
    df = df.copy()

    # ─── TARGET ───────────────────────────────────────────────────────────────
    processed['fraud_label'] = df['is_fraud']

    # ─── MONETARY ─────────────────────────────────────────────────────────────
    # BAF: amount, credit_limit
    processed['amount'] = (df['amt'] / df['amt'].max()).clip(0, 1)
    # credit_limit: wealthier population areas tend to have higher credit limits
    # Using log-normalised city population as a reasonable proxy (avoids circular = amount)
    processed['credit_limit'] = np.log1p(df['city_pop'] / df['city_pop'].max()).clip(0, 1)

    # ─── TIME ─────────────────────────────────────────────────────────────────
    # BAF: time_since_request_days, month
    df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    ref_date = df['trans_datetime'].min()
    processed['time_since_request_days'] = (
        (df['trans_datetime'] - ref_date).dt.total_seconds() / 86400
    ).clip(0, 365) / 365
    processed['month'] = df['trans_datetime'].dt.month / 12.0

    # ─── CUSTOMER ─────────────────────────────────────────────────────────────
    # BAF: customer_age
    df['dob'] = pd.to_datetime(df['dob'])
    processed['customer_age'] = (
        (df['trans_datetime'] - df['dob']).dt.days / 365
    ).clip(18, 90)

    # ─── VELOCITY ─────────────────────────────────────────────────────────────
    # BAF: velocity_6h, velocity_24h, velocity_4w, location_velocity
    card_counts = df.groupby('cc_num').size().reset_index(name='_card_tx_count')
    df = df.merge(card_counts, on='cc_num', how='left')
    processed['velocity_6h']  = np.log1p(df['_card_tx_count'] / 6).clip(0, 5)
    processed['velocity_24h'] = np.log1p(df['_card_tx_count'] / 4).clip(0, 5)
    processed['velocity_4w']  = np.log1p(df['_card_tx_count'] / 2).clip(0, 5)
    processed['location_velocity'] = np.log1p(df['city_pop'] / 1000).clip(0, 1)

    # ─── DEVICE ───────────────────────────────────────────────────────────────
    # BAF: device_fraud_count, device_distinct_emails, device_os_encoded
    # device_fraud_count: MUST NOT use is_fraud — that leaks the label into features!
    # Setting to 0 (not available in IEEE raw data)
    processed['device_fraud_count']    = 0                        # not available — label leak fixed
    processed['device_distinct_emails'] = 0                       # not available
    processed['device_os_encoded']     = 0                        # not available

    # ─── IDENTITY ─────────────────────────────────────────────────────────────
    # BAF: name_email_similarity, email_is_free
    processed['name_email_similarity'] = 0.5                      # neutral proxy
    processed['email_is_free']         = 0                        # not available

    # ─── PHONE ────────────────────────────────────────────────────────────────
    # BAF: phone_valid
    processed['phone_valid'] = 1                                   # assume valid

    # ─── BANKING ──────────────────────────────────────────────────────────────
    # BAF: bank_tenure_months, credit_risk_score, has_other_cards
    processed['bank_tenure_months'] = (
        (df['trans_datetime'] - df['dob']).dt.days / 30
    ).clip(0, 600) / 600
    # credit_risk_score: ratio of this transaction's amount vs the user's mean spending.
    # A sudden large transaction relative to history is a genuine risk signal.
    # Avoids the circular 1-amount proxy which taught the model nothing real.
    user_mean_amt = df.groupby('cc_num')['amt'].transform('mean')
    processed['credit_risk_score'] = (df['amt'] / (user_mean_amt + 1e-8)).clip(0, 10) / 10
    processed['has_other_cards']   = 0                            # not available

    # ─── ADDRESS ──────────────────────────────────────────────────────────────
    # BAF: address_stability
    processed['address_stability'] = np.log1p(df['city_pop'] / 1000).clip(0, 1)

    # ─── EMPLOYMENT (one-hot) ─────────────────────────────────────────────────
    # BAF has emp_CA … emp_CF  (6 categories from employment_status)
    for col in ['emp_CA', 'emp_CB', 'emp_CC', 'emp_CD', 'emp_CE', 'emp_CF']:
        processed[col] = 0

    # ─── HOUSING (one-hot) ────────────────────────────────────────────────────
    # BAF has house_BA … house_BF  (6 categories from housing_status)
    for col in ['house_BA', 'house_BB', 'house_BC', 'house_BD', 'house_BE', 'house_BF']:
        processed[col] = 0

    # ─── SESSION ──────────────────────────────────────────────────────────────
    # BAF: session_length, keep_alive
    hour = df['trans_datetime'].dt.hour
    processed['session_length'] = hour / 24.0
    processed['keep_alive']     = ((hour >= 0) & (hour < 6)).astype(float)

    # ─── REQUEST ──────────────────────────────────────────────────────────────
    # BAF: is_foreign_request, source_internet
    print("   Calculating geographic distances...")
    distances = np.array([
        min(haversine_distance(row['lat'], row['long'],
                               row['merch_lat'], row['merch_long']) / 500, 1.0)
        for _, row in df[['lat', 'long', 'merch_lat', 'merch_long']].iterrows()
    ])
    processed['is_foreign_request'] = (distances > 0.5).astype(float)
    processed['source_internet']    = (distances > 0.2).astype(float)  # proxy: distance implies online

    # ─── CLEAN UP ─────────────────────────────────────────────────────────────
    processed = processed.fillna(0)

    print(f"✅ IEEE preprocessing complete: {processed.shape}")
    print(f"   Fraud rate in this data: {processed['fraud_label'].mean():.4f}")
    print(f"   Features created: {len(processed.columns) - 1}")

    return processed