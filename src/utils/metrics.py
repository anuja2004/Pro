"""
Custom metrics for fraud detection evaluation
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report as sk_classification_report
)

def fraud_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive fraud detection metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # AUC-ROC if probabilities provided
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def classification_report(y_true, y_pred, y_pred_proba=None):
    """
    Generate detailed classification report
    """
    basic_metrics = fraud_metrics(y_true, y_pred, y_pred_proba)
    
    report = "\n" + "="*50 + "\n"
    report += "CLASSIFICATION REPORT\n"
    report += "="*50 + "\n"
    
    report += f"\nAccuracy:  {basic_metrics['accuracy']:.4f}"
    report += f"\nPrecision: {basic_metrics['precision']:.4f}"
    report += f"\nRecall:    {basic_metrics['recall']:.4f}"
    report += f"\nF1-Score:  {basic_metrics['f1']:.4f}"
    
    if 'auc_roc' in basic_metrics:
        report += f"\nAUC-ROC:   {basic_metrics['auc_roc']:.4f}"
    
    report += "\n\nConfusion Matrix:"
    report += f"\n  TN: {basic_metrics['true_negatives']}"
    report += f"\n  FP: {basic_metrics['false_positives']}"
    report += f"\n  FN: {basic_metrics['false_negatives']}"
    report += f"\n  TP: {basic_metrics['true_positives']}"
    
    report += "\n" + "="*50
    
    return report
