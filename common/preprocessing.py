import numpy as np

def build_features(amount, hour, txn_type, balance_delta):
    amount_log = np.log1p(amount)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    return np.column_stack([
        amount_log,
        hour_sin,
        hour_cos,
        txn_type,
        balance_delta,
    ])
