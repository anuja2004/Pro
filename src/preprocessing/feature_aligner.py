"""
Feature alignment between different datasets
"""
import pandas as pd
import numpy as np

def align_features(baf_df, ieee_df):
    """
    Align features between BAF and IEEE datasets
    
    Args:
        baf_df: Processed BAF DataFrame
        ieee_df: Processed IEEE DataFrame
        
    Returns:
        baf_aligned, ieee_aligned, common_features
    """
    print("🔄 Aligning features between datasets...")
    
    # Get common columns (excluding target)
    baf_cols = set(baf_df.columns) - {'fraud_label'}
    ieee_cols = set(ieee_df.columns) - {'fraud_label'}
    
    common_features = list(baf_cols.intersection(ieee_cols))
    
    print(f"   BAF features: {len(baf_cols)}")
    print(f"   IEEE features: {len(ieee_cols)}")
    print(f"   Common features: {len(common_features)}")
    
    # Ensure both datasets have the same columns
    baf_aligned = baf_df[common_features + ['fraud_label']].copy()
    ieee_aligned = ieee_df[common_features + ['fraud_label']].copy()
    
    # Handle any missing columns in IEEE
    for col in common_features:
        if col not in ieee_aligned.columns:
            ieee_aligned[col] = 0
    
    # Handle any missing columns in BAF
    for col in common_features:
        if col not in baf_aligned.columns:
            baf_aligned[col] = 0
    
    # Ensure column order is the same
    feature_cols = sorted(common_features)
    baf_aligned = baf_aligned[feature_cols + ['fraud_label']]
    ieee_aligned = ieee_aligned[feature_cols + ['fraud_label']]
    
    # Normalize numerical features to similar scales
    for col in feature_cols:
        if col in ['fraud_label']:
            continue
            
        # Get non-NaN values from both datasets
        baf_values = baf_aligned[col].dropna()
        ieee_values = ieee_aligned[col].dropna()
        
        if len(baf_values) > 0 and len(ieee_values) > 0:
            # Calculate global min/max
            all_values = pd.concat([baf_values, ieee_values])
            min_val = all_values.min()
            max_val = all_values.max()
            
            if max_val > min_val:
                # Min-max scale both datasets
                baf_aligned[col] = (baf_aligned[col] - min_val) / (max_val - min_val)
                ieee_aligned[col] = (ieee_aligned[col] - min_val) / (max_val - min_val)
    
    print(f"✅ Feature alignment complete")
    print(f"   BAF aligned shape: {baf_aligned.shape}")
    print(f"   IEEE aligned shape: {ieee_aligned.shape}")
    
    return baf_aligned, ieee_aligned, feature_cols
